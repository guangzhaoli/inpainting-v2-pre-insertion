from typing import Any, List, Optional, Union

import PIL
import logging
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils.torch_utils import is_compiled_module

from src.loss.edit_mask_loss import map_mask_to_latent
from src.models.load_model import load_qwenvl, load_transformer, load_vae
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from src.utils.images import (
    calculate_best_resolution,
    make_image_devisible,
    make_image_shape_devisible,
    resize_mask_bhw,
)


class QwenImageEditPlusTrainer(QwenImageEditTrainer):
    """Trainer class based on QwenImageEditPlusTrainer.

    Inherits from QwenImageEditTrainer and adds multi-image conditioning support.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        return QwenImageEditPlusPipeline

    _pack_latents = staticmethod(QwenImageEditPlusPipeline._pack_latents)
    _unpack_latents = staticmethod(QwenImageEditPlusPipeline._unpack_latents)

    def _unwrap_dit(self):
        dit = self.dit
        if self.accelerator is not None:
            dit = self.accelerator.unwrap_model(dit)
        if is_compiled_module(dit):
            dit = dit._orig_mod
        return dit

    def load_model(self, text_encoder_device=None):
        """Load and separate components from QwenImageEditPlusPipeline."""
        logging.info("Loading QwenImageEditPipeline and separating components...")

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
            text_encoder=None,
        )

        self.vae = load_vae(self.config.model.pretrained_model_name_or_path, weight_dtype=self.weight_dtype)
        self.text_encoder = load_qwenvl(
            self.config.model.pretrained_model_name_or_path, weight_dtype=self.weight_dtype,
            attn_implementation=self.config.model.attention_backend,
        )
        logging.info(f"text_encoder device: {self.text_encoder.device}")

        self.dit = load_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
            attention_backend=self.config.model.attention_backend,
            zero_cond_t=self.config.model.zero_cond_t,
        )

        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from diffusers.image_processor import VaeImageProcessor

        self.processor: Qwen2VLProcessor = pipe.processor
        self.tokenizer: Qwen2Tokenizer = pipe.tokenizer
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
        self._cache_scheduler_training_tensors()

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.vae_latent_mean = self.vae.config.latents_mean
        self.vae_latent_std = self.vae.config.latents_std
        self.vae_z_dim = self.vae.config.z_dim

        self.latent_channels = self.vae.config.z_dim
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        self.prompt_template_encode = pipe.prompt_template_encode
        self.prompt_template_encode_start_idx = pipe.prompt_template_encode_start_idx

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.num_channels_latents = self.dit.config.in_channels // 4

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def _normalize_control_keys(self, control_keys: Any) -> set[str]:
        if control_keys is None:
            return set()
        if isinstance(control_keys, torch.Tensor):
            control_keys = control_keys.tolist()
        if isinstance(control_keys, str):
            return {control_keys}
        return {str(control_key) for control_key in control_keys}

    def _pad_hidden_sequences(self, sequences: List[torch.Tensor]):
        max_seq_len = max(sequence.size(0) for sequence in sequences)
        padded = torch.stack(
            [
                torch.cat([sequence, sequence.new_zeros(max_seq_len - sequence.size(0), sequence.size(1))])
                for sequence in sequences
            ]
        )
        attention_mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(sequence.size(0), dtype=torch.long, device=sequence.device),
                        torch.zeros(max_seq_len - sequence.size(0), dtype=torch.long, device=sequence.device),
                    ]
                )
                for sequence in sequences
            ]
        )
        return padded, attention_mask

    def prepare_embeddings(self, batch, stage="fit"):
        # This cache lives for one OPD batch only. Keys use the original tensor
        # identities, retained by the batch/cache, and the actual prompt/images.
        cache = batch.pop("_opd_encoding_cache", None)
        if cache is not None and stage != "opd":
            raise ValueError("frozen condition reuse is only supported for OPD")

        def cached(key, compute):
            if cache is None:
                return compute()
            if key not in cache:
                with torch.no_grad():
                    cache[key] = compute()
            return cache[key]

        if "image" in batch:
            batch["image"] = self.preprocess_image_for_vae_encoder(
                batch["image"], input_range="0-1"
            )
            batch["width"] = batch["image"].shape[4]
            batch["height"] = batch["image"].shape[3]

        batch_size = batch["image"].shape[0] if "image" in batch else batch["control"].shape[0]
        condition_images_per_sample = [[] for _ in range(batch_size)]
        image_cache_keys = []
        control_cache_keys = {}
        vae_only_control_keys = self._normalize_control_keys(batch.get("vae_only_control_keys"))
        if "control" not in batch:
            raise ValueError("prepare_embeddings requires main control image under key 'control'")

        if "control" in batch:
            control_image = batch["control"]
            control_cache_keys["control"] = id(control_image)
            image_cache_keys.append(id(control_image))
            prompt_control = cached(
                ("vl_pixels", id(control_image)),
                lambda: self.preprocess_image_for_text_encoder(
                    control_image, best_resolution=384 * 384, input_range="0-1"
                ),
            )
            for batch_index in range(batch_size):
                condition_images_per_sample[batch_index].append(prompt_control[batch_index])
            batch["control"] = cached(
                ("vae_pixels", id(control_image)),
                lambda: self.preprocess_image_for_vae_encoder(control_image, input_range="0-1"),
            )
            batch["width_control"] = batch["control"].shape[4]
            batch["height_control"] = batch["control"].shape[3]

        if isinstance(batch["n_controls"], int):
            num_additional_controls = batch["n_controls"]
        else:
            num_additional_controls = batch["n_controls"][0]

        for control_index in range(num_additional_controls):
            additional_control_key = f"control_{control_index + 1}"
            if additional_control_key not in batch:
                raise ValueError(
                    f"n_controls={num_additional_controls} requires missing key '{additional_control_key}'"
                )
            additional_control_image = batch[additional_control_key]
            control_cache_keys[additional_control_key] = id(additional_control_image)
            if additional_control_key not in vae_only_control_keys:
                image_cache_keys.append(id(additional_control_image))
                prompt_control = cached(
                    ("vl_pixels", id(additional_control_image)),
                    lambda: self.preprocess_image_for_text_encoder(
                        additional_control_image, best_resolution=384 * 384, input_range="0-1"
                    ),
                )
                for batch_index in range(batch_size):
                    condition_images_per_sample[batch_index].append(prompt_control[batch_index])
            batch[additional_control_key] = cached(
                ("vae_pixels", id(additional_control_image)),
                lambda: self.preprocess_image_for_vae_encoder(additional_control_image, input_range="0-1"),
            )
            batch[f"width_control_{control_index + 1}"] = batch[additional_control_key].shape[4]
            batch[f"height_control_{control_index + 1}"] = batch[additional_control_key].shape[3]

        condition_images = [img for sample_imgs in condition_images_per_sample for img in sample_imgs]
        def encode_text(prompt):
            prompts = (prompt,) if isinstance(prompt, str) else tuple(prompt)
            return cached(
                ("vl_embeddings", prompts, tuple(image_cache_keys)),
                lambda: self.encode_prompt(prompt=prompt, image=condition_images),
            )

        prompt_embeds, prompt_embeds_mask = encode_text(batch["prompt"])

        batch["prompt_embeds_mask"] = prompt_embeds_mask
        batch["prompt_embeds"] = prompt_embeds

        if "negative_prompt" in batch and batch["true_cfg_scale"] > 1:
            negative_prompt_embeds, negative_prompt_embeds_mask = encode_text(batch["negative_prompt"])
            batch["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
            batch["negative_prompt_embeds"] = negative_prompt_embeds

        if "image" in batch:
            image = batch["image"]
            batch_size = image.shape[0]
            image_latents = self.encode_image_latents(
                image,
                batch_size,
                self.num_channels_latents,
                self.weight_dtype,
            )
            batch["image_latents"] = image_latents

        if "control" in batch:
            control = batch["control"]
            batch_size = control.shape[0]
            control_latents = cached(
                ("vae_latents", control_cache_keys["control"]),
                lambda: self.encode_image_latents(control, batch_size, self.num_channels_latents, self.weight_dtype),
            )
            batch["control_latents"] = [control_latents]

        for control_index in range(1, num_additional_controls + 1):
            control_key = f"control_{control_index}"
            control = batch[control_key]
            batch_size = control.shape[0]
            control_latents = cached(
                ("vae_latents", control_cache_keys[control_key]),
                lambda: self.encode_image_latents(control, batch_size, self.num_channels_latents, self.weight_dtype),
            )
            batch["control_latents"].append(control_latents)

        if "control_latents" in batch:
            batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)

        if self.config.loss.mask_loss and "mask" in batch:
            mask = batch["mask"]
            height_image, width_image = batch["height"], batch["width"]
            pixel_mask = resize_mask_bhw(mask, height_image, width_image)
            # InsertAnything uses this Plus override rather than the base
            # QwenImageEditTrainer implementation. Preserve the pixel-space GT
            # mask before packing so region-perceptual loss can consume it.
            if getattr(self.config.loss, "region_perceptual", False):
                batch["pixel_mask"] = pixel_mask
            batch["mask"] = pixel_mask
            batch["mask"] = map_mask_to_latent(
                batch["mask"],
                vae_scale_factor=self.vae_scale_factor,
                expected_seq_len=batch["image_latents"].shape[1],
            )

        if "image_latents" in batch:
            batch_size = batch["image_latents"].shape[0]
            batch["img_shapes"] = self._get_image_shapes(batch, batch_size)

        return batch

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
    ):
        num_images_per_prompt = 1
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if image is not None and len(image) > 0 and isinstance(image[0], torch.Tensor):
            new_images = []
            for img in image:
                image_array = img.permute(1, 2, 0).cpu().numpy().astype("uint8")
                new_images.append(PIL.Image.fromarray(image_array))
            image = new_images

        with self.conditioning_context(), torch.no_grad():
            device = self.text_encoder.device
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt,
                image,
                device,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        dtype = dtype or self.weight_dtype

        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

        if isinstance(image, list) and len(image) > 0:
            num_images_per_sample = len(image) // batch_size
            if len(image) % batch_size != 0:
                raise ValueError(
                    f"Number of images ({len(image)}) must be divisible by batch size ({batch_size}). "
                    f"Expected {batch_size * num_images_per_sample} images."
                )
        else:
            num_images_per_sample = 0

        txt = []
        for sample_index in range(batch_size):
            base_img_prompt = ""
            for image_index in range(num_images_per_sample):
                base_img_prompt += img_prompt_template.format(image_index + 1)
            txt.append(self.prompt_template_encode.format(base_img_prompt + prompt[sample_index]))

        model_inputs = self.processor(
            text=txt,
            images=image if image else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        drop_idx = self.prompt_template_encode_start_idx
        encoder = getattr(self.text_encoder, "model", None)
        if encoder is not None:
            outputs = encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=False,
                return_dict=True,
            )
            final_hidden_states = outputs.last_hidden_state
        else:
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            final_hidden_states = outputs.hidden_states[-1]

        split_hidden_states = self._extract_masked_hidden(final_hidden_states, model_inputs.attention_mask)
        split_hidden_states = [hidden_state[drop_idx:] for hidden_state in split_hidden_states]
        prompt_embeds, encoder_attention_mask = self._pad_hidden_sequences(split_hidden_states)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def prepare_predict_batch_data(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        controls_size: Optional[List[int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Union[None, str, List[str]] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        image_latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        prompt_embeds_mask: torch.Tensor = None,
        use_native_size: bool = False,
        best_resolution_size: bool = False,
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> dict:
        if image is None:
            raise ValueError("image is required")
        if prompt is None:
            raise ValueError("prompt is required")

        if not isinstance(image, list):
            image = [image]
        image = [make_image_devisible(single_image, self.vae_scale_factor) for single_image in image]

        prompt_image = [image[0]]
        additional_controls = [image[1:]] if len(image) > 1 else None

        if isinstance(prompt, str):
            prompt = [prompt]

        self.weight_dtype = weight_dtype

        data = {}
        control = []
        if use_native_size:
            controls_size = [[prompt_image[0].size[1], prompt_image[0].size[0]]]
            if additional_controls:
                controls_size.extend([[control.size[1], control.size[0]] for control in additional_controls[0]])
        elif best_resolution_size and controls_size is None:
            controls_size = [[prompt_image[0].size[1], prompt_image[0].size[0]]]
            if additional_controls:
                controls_size.extend([[control.size[1], control.size[0]] for control in additional_controls[0]])

        if best_resolution_size:
            controls_size = [
                calculate_best_resolution(c_size[0], c_size[1], 1024 * 1024) for c_size in controls_size
            ]

        for img in prompt_image:
            img = self.preprocessor.preprocess({"control": img}, controls_size=controls_size)["control"]
            control.append(img)
        control = torch.stack(control, dim=0)
        data["control"] = control
        data["prompt"] = prompt
        data["height"] = height
        data["width"] = width

        if height is None or width is None:
            height, width = control.shape[2], control.shape[3]
        else:
            width, height = make_image_shape_devisible(width, height, self.vae_scale_factor)

        if additional_controls:
            n_controls = len(additional_controls[0])
            new_controls = {f"control_{i+1}": [] for i in range(n_controls)}

            for controls in additional_controls:
                controls = self.preprocessor.preprocess({"controls": controls}, controls_size=controls_size)["controls"]
                for control_index, control_tensor in enumerate(controls):
                    new_controls[f"control_{control_index+1}"].append(control_tensor)

            for control_index in range(n_controls):
                control_stack = torch.stack(new_controls[f"control_{control_index+1}"], dim=0)
                data[f"control_{control_index+1}"] = control_stack
            data["n_controls"] = n_controls
        else:
            data["n_controls"] = 0

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            if len(negative_prompt) != len(data["prompt"]):
                raise ValueError("the number of negative_prompt should be same of control")
            data["negative_prompt"] = negative_prompt
        data["num_inference_steps"] = num_inference_steps
        data["true_cfg_scale"] = true_cfg_scale
        data["guidance"] = guidance_scale
        return data
