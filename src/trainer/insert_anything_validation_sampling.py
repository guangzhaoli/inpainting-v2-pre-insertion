import csv
import json
import logging
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from src.models.complementary_canvas import complementary_canvas_enabled
from src.utils.tools import instantiate_class
from src.utils.visualization import (
    binarize_mask,
    build_strict_mask_composite,
    ensure_mask,
    ensure_rgb,
    make_infer_row,
    make_mask_overlay,
    save_mask,
    stack_rows,
)
from src.utils.adaptive_crop import (
    crop_pil_pair,
    restore_pil_output,
)
from src.utils.adaptive_crop_anchoring import change_gated_paste_back, prepare_adaptive_crop_for_inference
from src.utils.inference_metadata import (
    build_effective_inference_metadata,
    build_inference_config_metadata,
    resolve_guidance_embeds_enabled,
    validate_inference_mode_inputs,
)
from src.utils.latent_anchoring import (
    ensure_latent_anchoring_config,
    latent_anchoring_metadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationSample:
    index: int
    ref_image: Image.Image
    ref_mask: Image.Image
    tar_image: Image.Image
    tar_mask: Image.Image
    prompt: Optional[str]
    category: str
    stem: str


class DatasetValidationSource:
    def __init__(self, dataset):
        required = ("_get_sample_info", "_read_image", "_read_mask")
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                "Insert Anything validation requires a dataset exposing "
                f"{required}, missing: {missing}"
            )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def get_prompt(self, index: int) -> Optional[str]:
        return self.dataset._get_sample_info(index).get("prompt")

    def get_sample(self, index: int) -> ValidationSample:
        sample_info = self.dataset._get_sample_info(index)
        stem = str(sample_info.get("stem") or self._stem_from_payload(sample_info, index))
        return ValidationSample(
            index=index,
            ref_image=_array_to_rgb_pil(self.dataset._read_image(sample_info["ref_image"])),
            ref_mask=_array_to_mask_pil(self.dataset._read_mask(sample_info["ref_mask"])),
            tar_image=_array_to_rgb_pil(self.dataset._read_image(sample_info["tar_image"])),
            tar_mask=_array_to_mask_pil(self.dataset._read_mask(sample_info["tar_mask"])),
            prompt=sample_info.get("prompt"),
            category=str(sample_info.get("category") or "validation"),
            stem=stem,
        )

    @staticmethod
    def _stem_from_payload(sample_info: Dict[str, Any], index: int) -> str:
        value = sample_info.get("tar_image") or sample_info.get("ref_image")
        if isinstance(value, str):
            return Path(value).stem
        if isinstance(value, dict) and value.get("path"):
            return Path(value["path"]).stem
        return f"sample_{index:06d}"


class ManifestValidationSource:
    def __init__(self, rows: Sequence[Dict[str, Any]], base_dir: Path):
        if not rows:
            raise ValueError("validation_data must contain at least one sample")
        self.rows = list(rows)
        self.base_dir = base_dir

    def __len__(self) -> int:
        return len(self.rows)

    def get_prompt(self, index: int) -> Optional[str]:
        return self.rows[index].get("prompt")

    def get_sample(self, index: int) -> ValidationSample:
        row = self.rows[index]
        required = ("ref_image", "ref_mask", "tar_image", "tar_mask")
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise ValueError(f"validation sample {index} is missing required keys: {missing}")

        stem = str(row.get("stem") or Path(str(row["tar_image"])).stem or f"sample_{index:06d}")
        return ValidationSample(
            index=index,
            ref_image=_read_rgb_image(row["ref_image"], self.base_dir),
            ref_mask=_read_mask_image(row["ref_mask"], self.base_dir),
            tar_image=_read_rgb_image(row["tar_image"], self.base_dir),
            tar_mask=_read_mask_image(row["tar_mask"], self.base_dir),
            prompt=row.get("prompt"),
            category=str(row.get("category") or "validation"),
            stem=stem,
        )


class InsertAnythingValidationSampler:
    def __init__(self, config, train_dataloader, accelerator):
        self.config = config
        self.sampling_config = config.logging.sampling
        self.accelerator = accelerator
        validate_inference_mode_inputs(
            config,
            prompt="",
            negative_prompt=self.sampling_config.negative_prompt,
        )
        self.source = self._build_source(train_dataloader)
        self.sample_indices = self._select_sample_indices()
        for sample_index in self.sample_indices:
            validate_inference_mode_inputs(
                config,
                prompt=self.source.get_prompt(sample_index),
                negative_prompt=self.sampling_config.negative_prompt,
            )

    def _validation_latent_anchoring_metadata(self) -> Dict[str, Any]:
        """Validation and deployment use the same primary-output contract."""

        configured = ensure_latent_anchoring_config(self.config)
        configured_metadata = latent_anchoring_metadata(configured)
        return {
            "configured": configured_metadata,
            "effective": configured_metadata,
            "forced_legacy_for_validation": False,
            "configured_mode_overridden": False,
        }

    def _build_source(self, train_dataloader):
        validation_data = self.sampling_config.validation_data
        if validation_data is None:
            dataset = getattr(train_dataloader, "dataset", None)
            if dataset is None:
                raise ValueError("train_dataloader must expose dataset when logging.sampling.validation_data is unset")
            return DatasetValidationSource(dataset)

        if isinstance(validation_data, list):
            return ManifestValidationSource(validation_data, Path.cwd())

        validation_path = Path(os.path.expanduser(os.path.expandvars(str(validation_data))))
        if validation_path.is_file():
            return ManifestValidationSource(_read_manifest_rows(validation_path), validation_path.parent)
        if validation_path.is_dir():
            if _is_direct_category_dir(validation_path):
                init_args = self.config.data.init_args.model_copy(
                    deep=True,
                    update={
                        "dataset_path": str(validation_path.parent),
                        "manifest_path": None,
                        "categories": [validation_path.name],
                    },
                )
                dataset = instantiate_class(self.config.data.class_path, init_args)
                return DatasetValidationSource(dataset)
            init_args = self.config.data.init_args.model_copy(
                deep=True,
                update={
                    "dataset_path": str(validation_path),
                    "manifest_path": None,
                },
            )
            dataset = instantiate_class(self.config.data.class_path, init_args)
            return DatasetValidationSource(dataset)

        raise FileNotFoundError(f"validation_data path does not exist: {validation_path}")

    def _select_sample_indices(self) -> List[int]:
        source_size = len(self.source)
        num_samples = int(self.sampling_config.num_samples)
        if num_samples > source_size:
            raise ValueError(
                f"logging.sampling.num_samples={num_samples} exceeds validation source size={source_size}"
            )
        generator = torch.Generator().manual_seed(int(self.sampling_config.seed))
        return torch.randperm(source_size, generator=generator)[:num_samples].tolist()

    def should_run_validation(self, global_step: int) -> bool:
        return global_step > 0 and global_step % int(self.sampling_config.validation_steps) == 0

    def run_validation_loop(self, global_step: int, trainer):
        accelerator = self.accelerator
        output_dir = self._step_output_dir(global_step)
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        logger.info(
            "Running Insert Anything validation sampling at step %s with %s samples",
            global_step,
            len(self.sample_indices),
        )

        saved_records = []
        row_paths = []
        rank = int(getattr(accelerator, "process_index", 0))
        world = int(getattr(accelerator, "num_processes", 1))
        # DDP forward counts must remain identical. Pad the final round with
        # duplicate work, but save each real sample exactly once.
        rounds = (len(self.sample_indices) + world - 1) // world
        with _preserve_rng_state():
            for round_index in range(rounds):
                order = round_index * world + rank
                is_real_sample = order < len(self.sample_indices)
                actual_order = order if is_real_sample else 0
                sample_index = self.sample_indices[actual_order]
                sample = self.source.get_sample(sample_index)
                sample_seed = self._seed_sample(global_step, actual_order)
                validate_inference_mode_inputs(
                    self.config,
                    prompt=sample.prompt,
                    negative_prompt=self.sampling_config.negative_prompt,
                )
                source = ensure_rgb(sample.tar_image)
                source_mask = ensure_mask(sample.tar_mask)
                if source.size != source_mask.size:
                    raise ValueError(
                        "validation source image and mask must have identical geometry, "
                        f"got image={source.size}, mask={source_mask.size}"
                    )
                adaptive_crop_config = self.config.data.init_args.adaptive_crop
                crop_plan = None
                anchoring = ensure_latent_anchoring_config(self.config)
                anchored = anchoring.mode.value != "legacy"
                # Anchoring and the complementary canvas both keep the latent source outside the edit region.
                latent_preserving = anchored or complementary_canvas_enabled(self.config)
                edge_feather_px = 0
                predict_source = source
                predict_source_mask = source_mask
                if adaptive_crop_config.predict_enabled:
                    crop_plan, crop_compatibility = prepare_adaptive_crop_for_inference(
                        image_size=source.size,
                        mask=source_mask,
                        adaptive_crop_config=adaptive_crop_config,
                        config=self.config,
                        processor_config=self.config.data.init_args.processor,
                    )
                    if latent_preserving and crop_plan.applied:
                        edge_feather_px = int(crop_compatibility.get("feather_width_px", 8))
                    predict_source, predict_source_mask = crop_pil_pair(
                        source,
                        source_mask,
                        crop_plan,
                    )
                generated_images = trainer.predict_loaded_model(
                    image=[predict_source, sample.ref_image],
                    prompt=sample.prompt,
                    ref_mask=sample.ref_mask,
                    src_mask=predict_source_mask,
                    num_inference_steps=int(self.sampling_config.num_inference_steps),
                    true_cfg_scale=float(self.sampling_config.true_cfg_scale),
                    guidance_scale=float(self.sampling_config.guidance_scale),
                    negative_prompt=self.sampling_config.negative_prompt,
                    weight_dtype=trainer.weight_dtype,
                    latent_anchoring_mode=anchoring.mode.value,
                )
                local_generated = ensure_rgb(generated_images[0])
                if crop_plan is not None:
                    raw_restored, final_composite = restore_pil_output(
                        original_source=source,
                        original_mask=source_mask,
                        local_output=local_generated,
                        plan=crop_plan,
                        blend_mode=adaptive_crop_config.blend_mode,
                    )
                else:
                    raw_restored = local_generated.resize(source.size, Image.Resampling.LANCZOS)
                    final_composite = build_strict_mask_composite(
                        source,
                        source_mask,
                        raw_restored,
                        adaptive_crop_config.mask_threshold,
                    )

                primary = final_composite
                if latent_preserving:
                    primary = raw_restored
                    if anchoring.paste_back == "change_gated":
                        primary = change_gated_paste_back(
                            source, source_mask, raw_restored,
                            local_size=local_generated.size, crop_plan=crop_plan, settings=anchoring,
                        )
                    elif edge_feather_px:
                        primary, _ = restore_pil_output(
                            original_source=source, original_mask=source_mask,
                            local_output=local_generated, plan=crop_plan,
                            blend_mode=adaptive_crop_config.blend_mode,
                            edge_feather_px=edge_feather_px,
                        )

                if is_real_sample:
                    record, row_path = self._save_sample_outputs(
                        output_dir=output_dir,
                        order=order,
                        sample=sample,
                        sample_seed=sample_seed,
                        raw_restored=raw_restored,
                        final_composite=final_composite,
                        primary_output=primary,
                        crop_plan=crop_plan,
                        cropped_source=predict_source if crop_plan is not None else None,
                        cropped_mask=predict_source_mask if crop_plan is not None else None,
                        local_generated=local_generated,
                        guidance_embeds_enabled=resolve_guidance_embeds_enabled(
                            self.config,
                            trainer,
                        ),
                    )
                    from src.utils.teacher_validation import teacher_validation_metrics

                    context_width = predict_source.width
                    radius = int(getattr(self.config.loss, "interaction_radius_px", 0)) if hasattr(self.config, "loss") else 0
                    record["metrics"] = teacher_validation_metrics(
                        sample, raw_restored,
                        criterion=getattr(trainer, "region_perceptual_criterion", None),
                        device=getattr(accelerator, "device", "cpu"),
                        exclusion_radius=radius * context_width / local_generated.width,
                    )
                    saved_records.append(record)
                    row_paths.append(row_path)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if world > 1:
            from accelerate.utils import gather_object

            saved_records = gather_object(saved_records)
            saved_records.sort(key=lambda record: record["order"])
            row_paths = [Path(record["paths"]["row"]) for record in saved_records]
        if accelerator.is_main_process:
            metric_names = {key for record in saved_records for key in record.get("metrics", {})}
            metrics = {
                name: float(np.mean([r["metrics"][name] for r in saved_records if name in r.get("metrics", {})]))
                for name in metric_names
            }
            with (output_dir / "metrics.json").open("w") as stream:
                json.dump({"step": global_step, "samples": len(saved_records), "metrics": metrics}, stream, indent=2)
            if hasattr(accelerator, "log"):
                accelerator.log({f"validation/{key}": value for key, value in metrics.items()}, step=global_step)
            all_rows_path = output_dir / "all_rows.png"
            grid_path = output_dir / "grid.png"
            sheet = stack_rows(row_paths, all_rows_path)
            sheet.save(grid_path)
            self._write_metadata(output_dir, global_step, saved_records, all_rows_path, grid_path)
            self._log_tensorboard_grid(trainer, sheet, global_step)
            logger.info("Validation samples saved to %s", output_dir)
        accelerator.wait_for_everyone()

    def _step_output_dir(self, global_step: int) -> Path:
        configured = Path(str(self.sampling_config.output_dir))
        if configured.is_absolute():
            base_dir = configured
        else:
            base_dir = Path(self.config.logging.output_dir) / configured
        return base_dir / f"step-{global_step:08d}"

    def _seed_sample(self, global_step: int, order: int) -> int:
        step_component = 0 if self.sampling_config.fixed_seed_across_steps else int(global_step) * 1_000_003
        seed = int(self.sampling_config.seed) + step_component + order
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def _save_sample_outputs(
        self,
        output_dir: Path,
        order: int,
        sample: ValidationSample,
        sample_seed: int,
        raw_restored: Image.Image,
        final_composite: Image.Image,
        crop_plan=None,
        cropped_source: Optional[Image.Image] = None,
        cropped_mask: Optional[Image.Image] = None,
        local_generated: Optional[Image.Image] = None,
        guidance_embeds_enabled: Optional[bool] = None,
        primary_output: Optional[Image.Image] = None,
    ):
        prefix = f"{order:03d}-idx{sample.index:06d}-{_safe_name(sample.category)}-{_safe_name(sample.stem)}"
        source = ensure_rgb(sample.tar_image)
        source_mask = ensure_mask(sample.tar_mask)
        reference = ensure_rgb(sample.ref_image)
        reference_mask = ensure_mask(sample.ref_mask)
        raw_output = ensure_rgb(raw_restored)
        output = ensure_rgb(primary_output if primary_output is not None else final_composite)
        if raw_output.size != source.size or output.size != source.size:
            raise ValueError("validation restored and final outputs must match the original source size")
        if local_generated is None:
            raise ValueError("validation output requires the direct local model output")
        local_output = ensure_rgb(local_generated)
        output_mask = binarize_mask(
            source_mask,
            self.config.data.init_args.adaptive_crop.mask_threshold,
        )

        paths = {
            "source": output_dir / f"{prefix}_source.png",
            "source_mask": output_dir / f"{prefix}_source_mask.png",
            "source_mask_overlay": output_dir / f"{prefix}_source_mask_overlay.png",
            "reference": output_dir / f"{prefix}_reference.png",
            "reference_mask": output_dir / f"{prefix}_reference_mask.png",
            "reference_mask_overlay": output_dir / f"{prefix}_reference_mask_overlay.png",
            "output_local": output_dir / f"{prefix}_output_local.png",
            "output_restored_raw": output_dir / f"{prefix}_output_restored_raw.png",
            "output": output_dir / f"{prefix}_output.png",
            "output_mask": output_dir / f"{prefix}_output_mask.png",
            "output_mask_overlay": output_dir / f"{prefix}_output_mask_overlay.png",
            "row": output_dir / f"{prefix}_row.png",
        }
        paths["generated"] = paths["output"]
        paths["final_composite"] = paths["output"]
        if self.sampling_config.save_composite:
            paths["composite"] = output_dir / f"{prefix}_composite.png"
            paths["final_composite"] = paths["composite"]
        if crop_plan is not None:
            paths["crop_source"] = output_dir / f"{prefix}_crop_source.png"
            paths["crop_mask"] = output_dir / f"{prefix}_crop_mask.png"

        source.save(paths["source"])
        save_mask(source_mask, paths["source_mask"])
        make_mask_overlay(source, source_mask).save(paths["source_mask_overlay"])
        reference.save(paths["reference"])
        save_mask(reference_mask, paths["reference_mask"])
        make_mask_overlay(reference, reference_mask).save(paths["reference_mask_overlay"])
        local_output.save(paths["output_local"])
        raw_output.save(paths["output_restored_raw"])
        output.save(paths["output"])
        save_mask(output_mask, paths["output_mask"])
        make_mask_overlay(raw_output, output_mask).save(paths["output_mask_overlay"])
        if self.sampling_config.save_composite:
            ensure_rgb(final_composite).save(paths["composite"])
        if crop_plan is not None:
            if cropped_source is None or cropped_mask is None:
                raise ValueError("adaptive crop validation outputs require crop source and mask")
            ensure_rgb(cropped_source).save(paths["crop_source"])
            save_mask(cropped_mask, paths["crop_mask"])
        row = make_infer_row(
            ref_image=reference,
            ref_mask=reference_mask,
            source=source,
            source_mask=source_mask,
            output=raw_output,
            composite=output,
            panel_size=int(self.sampling_config.panel_size),
            output_mask=output_mask,
        )
        row.save(paths["row"])

        record = {
            "order": order,
            "source_index": sample.index,
            "category": sample.category,
            "stem": sample.stem,
            "seed": int(sample_seed),
            "parameters": build_effective_inference_metadata(
                self.config,
                prompt=sample.prompt,
                negative_prompt=self.sampling_config.negative_prompt,
                num_inference_steps=int(self.sampling_config.num_inference_steps),
                true_cfg_scale=float(self.sampling_config.true_cfg_scale),
                guidance_scale=float(self.sampling_config.guidance_scale),
                guidance_embeds_enabled=guidance_embeds_enabled,
                effective_height=local_output.height,
                effective_width=local_output.width,
            ),
            "paths": {key: str(path) for key, path in paths.items()},
            "sizes": {
                "source": list(source.size),
                "prediction_input": list(cropped_source.size) if cropped_source is not None else list(source.size),
                "output_local": list(local_output.size),
                "output_restored_raw": list(raw_output.size),
                "final_composite": list(output.size),
            },
            "adaptive_crop": {
                "config": self.config.data.init_args.adaptive_crop.model_dump(mode="json"),
                "plan": crop_plan.to_dict() if crop_plan is not None else None,
            },
            "latent_anchoring": self._validation_latent_anchoring_metadata(),
        }
        return record, paths["row"]

    def _write_metadata(
        self,
        output_dir: Path,
        global_step: int,
        records: List[Dict[str, Any]],
        all_rows_path: Path,
        grid_path: Path,
    ):
        payload = {
            "global_step": int(global_step),
            "validation_steps": int(self.sampling_config.validation_steps),
            "num_samples": int(self.sampling_config.num_samples),
            "seed": {
                "base": int(self.sampling_config.seed),
                "formula": ("base + sample_order" if self.sampling_config.fixed_seed_across_steps
                            else "base + global_step * 1000003 + sample_order"),
            },
            "config": build_inference_config_metadata(self.config),
            "checkpoint": {
                "mode": "in_memory_training_model",
                "global_step": int(global_step),
                "configured_pretrained_lora": self.config.model.lora.pretrained_weight,
            },
            "requested_sampling": {
                "num_inference_steps": int(self.sampling_config.num_inference_steps),
                "true_cfg_scale": float(self.sampling_config.true_cfg_scale),
                "guidance_scale": float(self.sampling_config.guidance_scale),
                "negative_prompt": self.sampling_config.negative_prompt,
            },
            "panel_size": int(self.sampling_config.panel_size),
            "adaptive_crop": self.config.data.init_args.adaptive_crop.model_dump(mode="json"),
            "latent_anchoring": self._validation_latent_anchoring_metadata(),
            "output_semantics": {
                "output": "Configured primary result in original source coordinates; composite is the strict-mask comparison.",
                "output_local": "Direct model output in processor coordinates.",
                "output_restored_raw": "Model output mapped to original source coordinates before mask compositing.",
            },
            "all_rows": str(all_rows_path),
            "grid": str(grid_path),
            "samples": records,
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _log_tensorboard_grid(self, trainer, grid: Image.Image, global_step: int):
        if getattr(trainer, "accelerator", None) is None:
            return
        for tracker in getattr(trainer.accelerator, "trackers", []):
            writer = getattr(tracker, "writer", None)
            if writer is None:
                continue
            writer.add_image(
                "validation/grid",
                np.asarray(grid.convert("RGB")),
                global_step=global_step,
                dataformats="HWC",
            )
            writer.flush()


def _read_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL row {line_number}: {path}") from exc
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON validation manifest must be a list or an object with a 'samples' list")
        return rows

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))

    raise ValueError(f"Unsupported validation manifest format: {path}")


def _is_direct_category_dir(path: Path) -> bool:
    required = ("ref_image", "ref_mask", "tar_image", "tar_mask")
    return all((path / name).is_dir() for name in required)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Validation file not found: {path}")
    return path


def _read_rgb_image(value: Any, base_dir: Path) -> Image.Image:
    if not isinstance(value, str):
        raise TypeError(f"validation image path must be a string, got {type(value)}")
    with Image.open(_resolve_path(value, base_dir)) as img:
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def _read_mask_image(value: Any, base_dir: Path) -> Image.Image:
    if not isinstance(value, str):
        raise TypeError(f"validation mask path must be a string, got {type(value)}")
    with Image.open(_resolve_path(value, base_dir)) as img:
        return ImageOps.exif_transpose(img).convert("L").copy()


def _array_to_rgb_pil(value: np.ndarray) -> Image.Image:
    if value.ndim != 3:
        raise ValueError(f"RGB image array must have shape [H, W, C], got {value.shape}")
    return Image.fromarray(value.astype(np.uint8)).convert("RGB")


def _array_to_mask_pil(value: np.ndarray) -> Image.Image:
    if value.ndim == 3:
        value = value[:, :, 0]
    if value.ndim != 2:
        raise ValueError(f"mask array must have shape [H, W], got {value.shape}")
    return Image.fromarray(value.astype(np.uint8)).convert("L")


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned[:80] or "sample"


@contextmanager
def _preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
