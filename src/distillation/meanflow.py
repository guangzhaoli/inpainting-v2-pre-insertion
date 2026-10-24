"""MeanFlow initialization and conditional DMD on a shared frozen Qwen backbone.

G, the frozen fused teacher, and the fake-score model own separate LoRAs.
Only G owns the trainable endpoint-time embedding. Teacher/fake geometry uses
the frozen teacher companion. Fake optimizer steps finish BEFORE a live G graph
is built; adapter selection is restored to G before activation recomputation.
"""

from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from peft import get_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer

from src.distillation.backends import is_adapter_parameter, is_lora_parameter
from src.distillation.meanflow_math import (
    add_noise,
    anchor_step,
    dmd_surrogate,
    expand_time,
    flow_step,
    meanflow_loss,
    synchronize_gradients,
)
from src.distillation.opd import QwenImageEditOPDStrategy, _load_component

FAKE_WEIGHTS = "fake_score_lora.safetensors"
FAKE_OPTIMIZER = "fake_score_optimizer.pt"
STRATEGY_STATE = "meanflow_training_state.json"


def forward_map(trainer, state, start, end, condition):
    """Uses the existing Qwen conditioning path, including clean control times."""
    batch = state.shape[0]
    start = torch.as_tensor(start, device=state.device, dtype=torch.float32).expand(
        batch
    )
    end = torch.as_tensor(end, device=state.device, dtype=torch.float32).expand(batch)
    conditioned = dict(condition, _flow_map_end=end)
    with (
        trainer.accelerator.autocast()
        if trainer.accelerator is not None
        else nullcontext()
    ):
        return QwenImageEditOPDStrategy._predict_raw_velocity(
            trainer, state, start, conditioned
        )


def sigma_schedule(trainer, steps, sequence_length):
    key = (steps, sequence_length, str(trainer._dit_device()))
    cache = getattr(trainer, "_flow_map_sigma_cache", None)
    if cache is None:
        cache = trainer._flow_map_sigma_cache = {}
    if key in cache:
        return cache[key]
    if steps == 1:
        # Qwen's shift_terminal stretching divides by 1-last_sigma, which is
        # zero for a single input sigma=1. A one-step flow map has exactly
        # these endpoints and needs no scheduler transformation.
        with torch.inference_mode(False), torch.no_grad():
            cache[key] = torch.tensor([1.0, 0.0], device=trainer._dit_device())
        return cache[key]
    # A validation call may enter under inference_mode. Cache normal tensors
    # so later training can save these constants for backward.
    with torch.inference_mode(False), torch.no_grad():
        timesteps, _ = trainer.prepare_predict_timesteps(steps, sequence_length)
        sigma = trainer.scheduler.sigmas.detach().float().cpu().clone()
    if sigma.numel() != steps + 1 or not bool(torch.all(sigma[:-1] > sigma[1:])):
        raise ValueError("MeanFlow requires a strictly descending T+1 sigma schedule")
    if not torch.allclose(timesteps.float().cpu() / 1000, sigma[:-1], atol=1e-5):
        raise ValueError("Qwen MeanFlow requires timestep/1000 == sigma")
    if not torch.isclose(sigma[0], sigma.new_tensor(1.0)) or not torch.isclose(
        sigma[-1], sigma.new_tensor(0.0)
    ):
        raise ValueError("MeanFlow schedule must run from sigma=1 to sigma=0")
    with torch.inference_mode(False), torch.no_grad():
        cache[key] = sigma.to(trainer._dit_device())
    return cache[key]


def initial_latent(trainer, condition):
    return trainer.prepare_initial_latents(
        batch_size=condition["control_latents"].shape[0],
        num_channels_latents=trainer._dit_config().in_channels // 4,
        height=int(condition["height"]),
        width=int(condition["width"]),
        dtype=trainer.weight_dtype,
        device=trainer._dit_device(),
    )


def apply_anchor(condition, generated, noise, sigma):
    source = condition.get("anchor_source_latents")
    weight = condition.get("anchor_edit_weight")
    if (source is None) != (weight is None):
        raise ValueError("Both anchor source and edit weight are required")
    if source is None:
        return generated
    return anchor_step(
        generated,
        source.to(generated),
        noise.to(generated),
        sigma,
        weight.to(generated),
    )


@torch.no_grad()
def sample_flow_map(trainer, condition):
    """Standalone and periodic inference share the exact same flow-map sampler."""
    if condition.get("reference_cfg_embeddings") is not None or condition.get(
        "negative_prompt"
    ):
        raise ValueError(
            "Flow-map inference requires the guidance-absorbed single branch"
        )
    if float(condition.get("true_cfg_scale", 1.0)) != 1.0:
        raise ValueError("Flow-map inference requires true_cfg_scale=1")
    condition = dict(condition)
    for key in (
        "control_latents",
        "prompt_embeds",
        "anchor_source_latents",
        "anchor_edit_weight",
    ):
        if condition.get(key) is not None:
            condition[key] = condition[key].to(
                device=trainer._dit_device(), dtype=trainer.weight_dtype
            )
    condition["prompt_embeds_mask"] = condition["prompt_embeds_mask"].to(
        device=trainer._dit_device(), dtype=torch.int64
    )
    condition["_opd_txt_seq_lens"] = (
        condition["prompt_embeds_mask"].sum(1).cpu().tolist()
    )
    condition = QwenImageEditOPDStrategy._cache_student_geometry(trainer, condition)
    latent = initial_latent(trainer, condition)
    noise = latent.detach().clone()
    sigmas = sigma_schedule(
        trainer, int(condition["num_inference_steps"]), latent.shape[1]
    )
    for start, end in zip(sigmas[:-1], sigmas[1:]):
        average = forward_map(trainer, latent, start, end, condition)
        latent = apply_anchor(
            condition, flow_step(latent, average, start, end), noise, end
        )
        latent = latent.to(dtype=trainer.weight_dtype)
    return latent


class QwenImageEditMeanFlowDMDStrategy(QwenImageEditOPDStrategy):
    # Multiple trainable adapters are activated serially. Explicit, bucketed
    # all-reduce avoids DDP unused-parameter hooks tied to mutable PEFT state.
    manual_gradient_sync = True

    def __init__(self, config, **kwargs):
        # Reuse OPD conditioning helpers, but not its routed trajectory
        # objective, metrics, or DDP-specific batch-size restrictions.
        self.config = config
        self.teachers = {teacher.name: teacher for teacher in config.teachers}
        self.backend = _load_component(config.teacher_backend)
        if config.meanflow is None:
            raise ValueError("distillation.meanflow is required")
        self.options = config.meanflow
        self.fake_optimizer = None
        self.fake_update_count = 0

    def setup(self, trainer):
        if len(self.config.teachers) != 1:
            raise ValueError("Use exactly one fused teacher for MeanFlow/DMD")
        if trainer.accelerator.mixed_precision not in {"no", "bf16"}:
            raise ValueError("MeanFlow/DMD runtime supports bf16 or no mixed precision")
        self.backend.setup(trainer, self.config.teachers)
        self.backend.assert_frozen()
        self.teacher = self.config.teachers[0]
        self.teacher_adapter = self.teacher.lora.adapter_name
        self.generator_adapter = trainer.adapter_name
        self.fake_adapter = self.options.fake_adapter
        self.trainer = trainer
        from src.utils.tools import calculate_sha256_file

        fingerprint = [None]
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                weight = Path(self.teacher.lora.pretrained_weight)
                if weight.is_dir():
                    weight = Path(trainer._resolve_resume_lora_weight(str(weight)))
                files = {"lora": weight}
                geometry = self.backend._geometry_weight_path(self.teacher)
                if geometry:
                    files["geometry"] = Path(geometry)
                fingerprint[0] = {
                    "files": {
                        name: calculate_sha256_file(str(path))
                        for name, path in files.items()
                    }
                }
            except Exception as exc:
                fingerprint[0] = {"error": str(exc)}
        if dist.is_initialized():
            dist.broadcast_object_list(fingerprint, src=0)
        if "error" in fingerprint[0]:
            raise ValueError(
                "Cannot fingerprint fused teacher: " + fingerprint[0]["error"]
            )
        self.teacher_fingerprint = fingerprint[0]["files"]
        if self.options.stage == "dmd":
            fake_config = trainer.config.model_copy(deep=True)
            fake_config.model.lora = self.teacher.lora.model_copy(deep=True)
            fake_config.model.lora.adapter_name = self.fake_adapter
            if trainer.config.resume:
                path = Path(trainer.config.resume) / FAKE_WEIGHTS
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing fake-score resume weights: {path}"
                    )
                fake_config.model.lora.pretrained_weight = str(path)
            trainer.__class__.load_pretrain_lora_model(
                trainer.dit, fake_config, self.fake_adapter, stage="fit"
            )
        self._cache_parameter_groups(trainer)
        self._active_role = None
        self._activate("generator")

    def _cache_parameter_groups(self, trainer):
        model = trainer._unwrap_dit()
        self._adapter_layers = tuple(
            layer for layer in model.modules() if isinstance(layer, BaseTunerLayer)
        )
        self._lora_trainability = []
        generator, fake = [], []
        for name, parameter in model.named_parameters():
            if not is_lora_parameter(name):
                continue
            is_generator = is_adapter_parameter(name, self.generator_adapter)
            is_fake = self.options.stage == "dmd" and is_adapter_parameter(
                name, self.fake_adapter
            )
            self._lora_trainability.append((parameter, is_generator or is_fake))
            if is_generator:
                generator.append(parameter)
            elif is_fake:
                fake.append(parameter)
        self._extra_generator_modules = [model.flow_map_time_embedding]
        if getattr(model, "mask_geometry_tokenizer", None) is not None:
            self._extra_generator_modules.append(model.mask_geometry_tokenizer)
        for module in self._extra_generator_modules:
            generator.extend(module.parameters())
        self._generator_parameters = tuple(generator)
        self._fake_parameters = tuple(fake)

    def configure_trainability(self, trainer):
        for parameter, trainable in self._lora_trainability:
            parameter.requires_grad_(trainable)
        for module in self._extra_generator_modules:
            module.requires_grad_(True)

    def _activate(self, role):
        adapter = {
            "generator": self.generator_adapter,
            "teacher": self.teacher_adapter,
            "fake": self.fake_adapter,
        }[role]
        if self._active_role == role:
            return
        for layer in self._adapter_layers:
            layer.set_adapter(adapter)
        # PEFT mutates requires_grad on set_adapter. Restore the stable training
        # set without touching existing accumulated G gradients.
        self.configure_trainability(self.trainer)
        self._active_role = role

    @contextmanager
    def role(self, role):
        self._activate(role)
        try:
            yield
        finally:
            self._activate("generator")

    def generator_parameters(self, trainer):
        return self._generator_parameters

    def fake_parameters(self, trainer):
        return self._fake_parameters

    def after_prepare(self, trainer):
        # Match DDP's initial parameter synchronization without registering its
        # hooks on the shared multi-role module.
        if dist.is_available() and dist.is_initialized():
            for p in trainer._unwrap_dit().parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)
        if self.options.stage == "dmd":
            self.fake_optimizer = torch.optim.AdamW(
                self.fake_parameters(trainer),
                lr=self.options.fake_lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                foreach=trainer._dit_device().type == "cuda",
            )

    def _signature(self):
        return {
            "version": 2,
            "options": self.options.model_dump(mode="json"),
            "teacher": self.teacher.model_dump(mode="json"),
            "flow_map": self.trainer.config.model.flow_map.model_dump(mode="json"),
            "teacher_fingerprint": self.teacher_fingerprint,
        }

    def consume_metrics(self, accelerator):
        # BaseTrainer aggregates the meanflow/* per-step metrics. Do not emit
        # the ordinary OPD strategy's unused route/trajectory counters.
        return {}

    def save_state(self, trainer, folder):
        folder = Path(folder)
        if self.options.stage == "dmd":
            weights = get_peft_model_state_dict(
                trainer._unwrap_dit(), adapter_name=self.fake_adapter
            )
            safetensors.torch.save_file(
                {k: v.detach().cpu().contiguous() for k, v in weights.items()},
                str(folder / FAKE_WEIGHTS),
            )
            torch.save(self.fake_optimizer.state_dict(), folder / FAKE_OPTIMIZER)
        state = dict(self._signature(), fake_update_count=self.fake_update_count)
        (folder / STRATEGY_STATE).write_text(json.dumps(state, indent=2))

    def restore_state(self, trainer, folder):
        folder = Path(folder)
        state = json.loads((folder / STRATEGY_STATE).read_text())
        for key, value in self._signature().items():
            if state.get(key) != value:
                raise ValueError(f"MeanFlow/DMD resume contract changed: {key}")
        self.fake_update_count = int(state["fake_update_count"])
        if self.options.stage == "dmd":
            self.fake_optimizer.load_state_dict(
                torch.load(
                    folder / FAKE_OPTIMIZER, map_location="cpu", weights_only=True
                )
            )
        self._activate("generator")

    def _instantaneous(self, state, sigma, condition):
        conditioned = dict(condition)
        conditioned.pop("_flow_map_end", None)
        with self.trainer.accelerator.autocast():
            return self._predict_raw_velocity(
                self.trainer, state, sigma * 1000, conditioned
            )

    def _required_condition_policies(self, teacher_names=None):
        # Config validation guarantees the same single-branch prompt policy.
        # Irrelevant teacher CFG defaults must not trigger a second encoding.
        guidance = self.config.student_guidance
        return {self._condition_key(guidance): guidance}

    def _prepare(self, trainer, batch):
        batch = trainer.prepare_opd_batch(dict(batch))
        if batch.get("source_image") is not None:
            if batch["source_image"].shape != batch["tar_image"].shape:
                raise ValueError("source_image must share processed target geometry")
            batch["tar_image"] = batch["source_image"]
        masks = None
        settings = trainer.config.predict.latent_anchoring
        mode = getattr(settings.mode, "value", settings.mode)
        if self.options.anchoring == "deployment" and mode != "legacy":
            from src.utils.latent_anchoring import build_latent_anchoring_masks

            masks = build_latent_anchoring_masks(
                batch["tar_mask_cond"],
                mode=mode,
                mask_threshold=settings.mask_threshold,
                interaction_dilation_px=settings.interaction_dilation_px,
                transition_width_px=settings.transition_width_px,
            )
            batch["source_hole_mask"] = masks.free
        # encode_prompt owns text-encoder offloading; avoid nested .to() walks.
        with torch.no_grad():
            conditions = self._prepare_conditions(trainer, batch)
        raw = conditions[self._condition_key(self.config.student_guidance)]
        frozen = self.backend._tokenize_teacher_geometry(
            self.teacher.name, trainer, raw
        )
        generator = self._prepare_student_condition(trainer, raw)
        if masks is not None:
            # Never encode the unmasked GT object as an anchor. Custom datasets
            # may supply a true clean source; otherwise anchor the masked context.
            source = batch.get("source_image")
            if source is None:
                source = batch["tar_image"] * (
                    1 - batch["source_hole_mask"].unsqueeze(1)
                )
            with torch.no_grad():
                shapes = trainer._get_image_shapes(raw, source.shape[0])
                if batch.get("source_image") is None and all(
                    len(shape) > 1 and shape[0] == shape[1] for shape in shapes
                ):
                    # The first control is exactly this masked canvas, already
                    # deterministically encoded with VAE argmax in conditioning.
                    anchor = raw["control_latents"][:, : math.prod(shapes[0][0])]
                else:
                    anchor = trainer.encode_image_latents(
                        trainer.preprocess_image_for_vae_encoder(
                            source, input_range="0-1"
                        ),
                        source.shape[0],
                        trainer.num_channels_latents,
                        trainer.weight_dtype,
                    )
                weight = trainer._prepare_latent_anchoring_weight(
                    edit_weight=masks.edit_weight,
                    free_mask=masks.free,
                    batch_size=source.shape[0],
                    num_channels_latents=trainer.num_channels_latents,
                    height_image=int(raw["height"]),
                    width_image=int(raw["width"]),
                    soft_transition=mode == "expanded_soft",
                )
            for value in (generator, frozen):
                value["anchor_source_latents"] = anchor.detach().to(
                    device=trainer._dit_device(), dtype=trainer.weight_dtype
                )
                value["anchor_edit_weight"] = weight.detach().to(
                    device=trainer._dit_device(), dtype=trainer.weight_dtype
                )
        return batch, generator, frozen

    @torch.no_grad()
    def _teacher_sample(self, condition):
        trainer = self.trainer
        latent = initial_latent(trainer, condition)
        noise = latent.clone()
        sigma = sigma_schedule(trainer, self.options.teacher_steps, latent.shape[1])
        with self.role("teacher"):
            for start, end in zip(sigma[:-1], sigma[1:]):
                v = self._instantaneous(latent, start, condition)
                latent = apply_anchor(
                    condition, flow_step(latent, v, start, end), noise, end
                )
                latent = latent.to(dtype=trainer.weight_dtype)
        return latent.detach()

    def _generate(self, condition, *, with_grad):
        trainer = self.trainer
        self._activate("generator")
        latent = initial_latent(trainer, condition)
        noise = latent.detach().clone()
        sigma = sigma_schedule(trainer, self.options.student_steps, latent.shape[1])
        if self.options.rollout == "shortcut":
            # Choose a real deployment interval, and learn a three-segment
            # shortcut chain as in AnyFlow. Zero-length segments are omitted.
            index = int(torch.randint(self.options.student_steps, ()).item())
            indices = sorted({0, index, index + 1, self.options.student_steps})
            intervals = [
                (sigma[a], sigma[b]) for a, b in zip(indices[:-1], indices[1:])
            ]
        else:
            intervals = list(zip(sigma[:-1], sigma[1:]))
        cached_condition = None
        for index, (start, end) in enumerate(intervals):
            grad = with_grad and (
                self.options.gradient_mode == "full" or index == len(intervals) - 1
            )
            if not grad:
                latent = latent.detach()
                if cached_condition is None:
                    cached_condition = self._cache_student_geometry(trainer, condition)
            with torch.enable_grad() if grad else torch.no_grad():
                u = forward_map(
                    trainer, latent, start, end, condition if grad else cached_condition
                )
                latent = apply_anchor(
                    condition, flow_step(latent, u, start, end), noise, end
                )
                latent = latent.to(dtype=trainer.weight_dtype)
        return latent

    def _forward_loss(self, condition, target):
        device, batch_size = target.device, target.shape[0]
        eps = self.options.finite_difference_epsilon
        times = torch.rand(2, batch_size, device=device) * (1 - 4 * eps) + 2 * eps
        t, r = times.max(0).values, times.min(0).values
        diagonal = (
            torch.rand(batch_size, device=device) < self.options.diagonal_probability
        )
        r = torch.where(diagonal, t, r)
        noise = torch.randn_like(target)
        self._activate("generator")
        return meanflow_loss(
            lambda x, a, b: forward_map(self.trainer, x, a, b, condition),
            target,
            noise,
            t,
            r,
            epsilon=eps,
            adaptive_power=self.options.adaptive_loss_power,
        )

    def backward_step(self, trainer, batch):
        self.trainer = trainer
        batch, condition, score_condition = self._prepare(trainer, batch)
        options = self.options
        metrics = {}
        total = torch.zeros((), device=trainer._dit_device())
        target = None
        if options.forward_weight > 0:
            target = self._teacher_sample(score_condition)
        if (
            target is not None
            and options.teacher_score_probe_interval
            and trainer.global_step % options.teacher_score_probe_interval == 0
        ):
            with torch.no_grad(), self.role("teacher"):
                tau = (
                    torch.rand(target.shape[0], device=target.device)
                    * (options.tau_max - options.tau_min)
                    + options.tau_min
                )
                noise = torch.randn_like(target)
                probe = self._instantaneous(
                    add_noise(target, noise, tau), tau, score_condition
                )
                metrics["meanflow/teacher_score_probe_mse"] = (
                    (probe.float() - (noise.float() - target.float())).square().mean()
                )
        # F updates complete before ANY generator autograd graph is constructed.
        if options.stage == "dmd":
            clean = self._generate(condition, with_grad=False).detach()
            fake_losses = []
            for _ in range(options.fake_updates):
                tau = (
                    torch.rand(clean.shape[0], device=clean.device)
                    * (options.tau_max - options.tau_min)
                    + options.tau_min
                )
                noise = torch.randn_like(clean)
                noisy = add_noise(clean, noise, tau)
                with self.role("fake"):
                    self.fake_optimizer.zero_grad(set_to_none=True)
                    prediction = self._instantaneous(noisy, tau, score_condition)
                    loss_fake = (
                        (prediction.float() - (noise.float() - clean.float()))
                        .square()
                        .mean()
                    )
                    # Accelerator.backward divides by G accumulation; F has its
                    # own per-batch update, so undo that division explicitly.
                    trainer.accelerator.backward(
                        loss_fake * trainer.config.train.gradient_accumulation_steps
                    )
                    synchronize_gradients(self.fake_parameters(trainer))
                    torch.nn.utils.clip_grad_norm_(
                        self.fake_parameters(trainer),
                        trainer.config.train.max_grad_norm,
                    )
                    self.fake_optimizer.step()
                    self.fake_optimizer.zero_grad(set_to_none=True)
                self.fake_update_count += 1
                fake_losses.append(loss_fake.detach())
            metrics["meanflow/fake_loss"] = torch.stack(fake_losses).mean()

        if target is not None:
            loss_forward = (
                self._forward_loss(condition, target) * options.forward_weight
            )
            trainer.accelerator.backward(loss_forward)
            total += loss_forward.detach()
            metrics["meanflow/forward_loss"] = loss_forward.detach()

        if options.stage == "dmd":
            generated = self._generate(condition, with_grad=True)
            tau = (
                torch.rand(generated.shape[0], device=generated.device)
                * (options.tau_max - options.tau_min)
                + options.tau_min
            )
            noisy = add_noise(generated.detach(), torch.randn_like(generated), tau)
            with torch.no_grad(), self.role("teacher"):
                real_v = self._instantaneous(noisy, tau, score_condition)
            with torch.no_grad(), self.role("fake"):
                fake_v = self._instantaneous(noisy, tau, score_condition)
            loss_dmd, direction = dmd_surrogate(
                generated,
                noisy,
                real_v,
                fake_v,
                tau,
                normalize=options.normalize_dmd,
            )
            # Restoring G BEFORE backward is essential for gradient checkpoint
            # recomputation on the shared PEFT backbone.
            self._activate("generator")
            loss = options.dmd_weight * loss_dmd
            metrics["meanflow/dmd_surrogate"] = loss_dmd.detach()
            metrics["meanflow/dmd_gradient_rms"] = direction.square().mean().sqrt()
            if options.keep_weight > 0:
                # Local x0 comparison from the SAME noisy state. Both estimates
                # use the already queried teacher denoiser as the detached
                # target for G's average-velocity map to zero.
                target_x0 = (noisy - expand_time(tau, noisy) * real_v.float()).detach()
                predicted_x0 = forward_map(
                    trainer, noisy, tau, torch.zeros_like(tau), condition
                )
                predicted_x0 = noisy - expand_time(tau, noisy) * predicted_x0.float()
                from src.loss.edit_mask_loss import MaskEditLoss, map_mask_to_latent

                mask = map_mask_to_latent(
                    batch["tar_mask"].to(generated.device),
                    vae_scale_factor=trainer.vae_scale_factor,
                    expected_seq_len=generated.shape[1],
                )
                keep = MaskEditLoss(
                    forground_weight=options.foreground_weight,
                    background_weight=options.background_weight,
                    mode="region_balanced",
                )(mask, predicted_x0, target_x0)
                loss = loss + options.keep_weight * keep
                metrics["meanflow/keep_loss"] = keep.detach()
            trainer.accelerator.backward(loss)
            total += loss.detach()
        self._activate("generator")
        metrics["meanflow/fake_updates"] = total.new_tensor(self.fake_update_count)
        metrics["meanflow/student_steps"] = total.new_tensor(options.student_steps)
        trainer._step_loss_metrics = metrics
        return total
