import os
from typing import Any, Dict, List, Optional, Union

import PIL
import logging
import safetensors.torch
import torch
import torch.nn.functional as F  # NOQA
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils.torch_utils import is_compiled_module

from src.loss.edit_mask_loss import map_mask_to_latent
from src.models.load_model import load_qwenvl, load_transformer, load_vae
from src.models.query_prompt import (
    QueryPromptAdapter,
    build_source_type_ids,
    compute_image_token_counts,
    group_image_token_counts,
)
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from src.utils.images import (
    calculate_best_resolution,
    make_image_devisible,
    make_image_shape_devisible,
    resize_mask_bhw,
)


QUERY_PROMPT_FILE_NAME = "query_prompt.safetensors"


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

    def _query_prompt_cfg(self):
        return self.config.model.query_prompt

    def _is_query_prompt_enabled(self) -> bool:
        return bool(self._query_prompt_cfg().enabled)

    def _unwrap_dit(self):
        dit = self.dit
        if self.accelerator is not None:
            dit = self.accelerator.unwrap_model(dit)
        if is_compiled_module(dit):
            dit = dit._orig_mod
        return dit

    def _get_query_prompt_module(self) -> Optional[QueryPromptAdapter]:
        return getattr(self._unwrap_dit(), "query_prompt_adapter", None)

    def _get_query_prompt_weight_path(self) -> Optional[str]:
        pretrained_weight = getattr(self.config.model.lora, "pretrained_weight", None)
        if pretrained_weight is None:
            return None
        weight_dir = pretrained_weight if os.path.isdir(pretrained_weight) else os.path.dirname(pretrained_weight)
        return os.path.join(weight_dir, QUERY_PROMPT_FILE_NAME)

    def save_additional_modules(self, save_folder):
        if not self._is_query_prompt_enabled():
            return

        unwrapped_transformer = self._unwrap_dit()

        module = getattr(unwrapped_transformer, "query_prompt_adapter", None)
        if module is None:
            raise ValueError("query prompt is enabled but query_prompt_adapter is not attached to the transformer")

        save_path = os.path.join(save_folder, QUERY_PROMPT_FILE_NAME)
        state_dict = {key: value.detach().cpu() for key, value in module.state_dict().items()}
        safetensors.torch.save_file(state_dict, save_path)
        torch.cuda.empty_cache()

    def load_additional_modules(self, stage="fit"):
        if not self._is_query_prompt_enabled():
            return

        module = self._get_query_prompt_module()
        if module is None:
            raise ValueError("query prompt is enabled but query_prompt_adapter is not attached to the transformer")

        weight_path = self._get_query_prompt_weight_path()
        if weight_path is None:
            if stage == "predict":
                raise FileNotFoundError(
                    "Query prompt is enabled for prediction but no pretrained LoRA/checkpoint path is configured."
                )
            return
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Query prompt is enabled but companion weights were not found at {weight_path}"
            )

        state_dict = safetensors.torch.load_file(weight_path)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected query prompt keys: {unexpected}")
        if missing:
            raise ValueError(f"Missing query prompt keys: {missing}")

    def load_model(self, text_encoder_device=None):
        """Load and separate components from QwenImageEditPlusPipeline."""
        logging.info("Loading QwenImageEditPipeline and separating components...")

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
        )
        pipe.to("cpu")
        logging.info(f"excution device: {pipe._execution_device}")

        self.vae = load_vae(self.config.model.pretrained_model_name_or_path, weight_dtype=self.weight_dtype)
        self.text_encoder = load_qwenvl(
            self.config.model.pretrained_model_name_or_path, weight_dtype=self.weight_dtype
        )
        logging.info(f"text_encoder device: {self.text_encoder.device}")

        self.dit = load_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
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

        if self._is_query_prompt_enabled():
            cfg = self._query_prompt_cfg()
            text_width = self.text_encoder.config.hidden_size
            self.dit.query_prompt_adapter = QueryPromptAdapter(
                input_dim=text_width,
                output_dim=text_width,
                hidden_dim=cfg.hidden_dim,
                num_query_tokens=cfg.num_query_tokens,
                num_heads=cfg.num_heads,
                num_layers=cfg.qformer_layers,
                num_input_layers=len(cfg.layer_indices),
                max_source_images=cfg.max_source_images,
            ).to(dtype=self.weight_dtype)

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def setup_model_device_train_mode(self, stage="fit"):
        super().setup_model_device_train_mode(stage=stage)
        if stage != "fit" or not self._is_query_prompt_enabled():
            return
        for name, param in self.dit.named_parameters():
            if "query_prompt_adapter" in name:
                param.requires_grad = True

    def _normalize_control_keys(self, control_keys: Any) -> set[str]:
        if control_keys is None:
            return set()
        if isinstance(control_keys, torch.Tensor):
            control_keys = control_keys.tolist()
        if isinstance(control_keys, str):
            return {control_keys}
        return {str(control_key) for control_key in control_keys}

    def _use_query_prompt(self, batch: Dict[str, Any]) -> bool:
        if not self._is_query_prompt_enabled():
            return False
        if self._get_query_prompt_module() is None:
            raise ValueError("query prompt is enabled but query_prompt_adapter is not attached to the transformer")
        return True

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

    def _normalize_query_layer_indices(self, total_hidden_layers: int) -> List[int]:
        normalized = []
        seen = set()
        for layer_index in self._query_prompt_cfg().layer_indices:
            index = layer_index if layer_index >= 0 else total_hidden_layers + layer_index
            if index < 0 or index >= total_hidden_layers:
                raise ValueError(f"Invalid query prompt layer index {layer_index} for {total_hidden_layers} layers")
            if index not in seen:
                normalized.append(index)
                seen.add(index)
        return normalized

    def _get_image_token_id(self) -> int:
        if hasattr(self.processor, "image_token_id") and self.processor.image_token_id is not None:
            return int(self.processor.image_token_id)
        if hasattr(self.tokenizer, "image_token_id") and self.tokenizer.image_token_id is not None:
            return int(self.tokenizer.image_token_id)
        if hasattr(self.text_encoder, "config") and getattr(self.text_encoder.config, "image_token_id", None) is not None:
            return int(self.text_encoder.config.image_token_id)
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or image_token_id < 0:
            raise ValueError(f"Failed to resolve image token id for token {image_token!r}")
        return int(image_token_id)

    def _get_image_token_mask(self, model_inputs) -> torch.Tensor:
        image_token_id = self._get_image_token_id()
        return model_inputs.input_ids == image_token_id

    def _build_query_prompt_inputs(
        self,
        layer_splits: Dict[int, List[torch.Tensor]],
        image_token_splits: List[torch.Tensor],
        token_counts_per_sample: List[List[int]],
        normalized_layer_indices: List[int],
    ):
        per_sample_visual = []
        per_sample_masks = []
        per_sample_source_ids = []
        max_visual_len = 0

        for sample_index, image_token_counts in enumerate(token_counts_per_sample):
            image_mask = image_token_splits[sample_index].bool()
            expected_tokens = sum(image_token_counts)
            actual_tokens = int(image_mask.sum().item())
            if actual_tokens != expected_tokens:
                raise ValueError(
                    f"Image token extraction mismatch for sample {sample_index}: expected {expected_tokens}, got {actual_tokens}"
                )

            source_type_ids = build_source_type_ids(
                image_token_counts,
                max_source_images=self._query_prompt_cfg().max_source_images,
                device=image_mask.device,
            )
            sample_layers = [layer_splits[layer_index][sample_index][image_mask] for layer_index in normalized_layer_indices]
            per_sample_visual.append(sample_layers)
            per_sample_masks.append(
                torch.ones(expected_tokens, dtype=torch.bool, device=image_mask.device)
            )
            per_sample_source_ids.append(source_type_ids)
            max_visual_len = max(max_visual_len, expected_tokens)

        padded_visual_layers = []
        padded_visual_masks = []
        padded_source_ids = []
        for sample_layers, sample_mask, source_type_ids in zip(
            per_sample_visual, per_sample_masks, per_sample_source_ids
        ):
            padded_layers = []
            for layer_tokens in sample_layers:
                pad_len = max_visual_len - layer_tokens.size(0)
                if pad_len > 0:
                    layer_tokens = torch.cat(
                        [layer_tokens, layer_tokens.new_zeros(pad_len, layer_tokens.size(1))], dim=0
                    )
                padded_layers.append(layer_tokens)
            padded_visual_layers.append(torch.stack(padded_layers, dim=0))

            pad_len = max_visual_len - sample_mask.size(0)
            if pad_len > 0:
                sample_mask = torch.cat(
                    [sample_mask, torch.zeros(pad_len, dtype=torch.bool, device=sample_mask.device)], dim=0
                )
                source_type_ids = torch.cat(
                    [source_type_ids, torch.zeros(pad_len, dtype=torch.long, device=source_type_ids.device)], dim=0
                )
            padded_visual_masks.append(sample_mask)
            padded_source_ids.append(source_type_ids)

        visual_layers = torch.stack(padded_visual_layers, dim=0)
        visual_mask = torch.stack(padded_visual_masks, dim=0)
        source_type_ids = torch.stack(padded_source_ids, dim=0)
        return visual_layers, visual_mask, source_type_ids

    def _collect_query_prompt_features(
        self,
        outputs,
        model_inputs,
        drop_idx: int,
        num_images_per_sample: int,
    ):
        if getattr(model_inputs, "image_grid_thw", None) is None:
            raise ValueError("query prompt requires image_grid_thw metadata from the Qwen-VL processor")

        normalized_layer_indices = self._normalize_query_layer_indices(len(outputs.hidden_states))
        layer_splits = {}
        for layer_index in normalized_layer_indices:
            split_hidden = self._extract_masked_hidden(outputs.hidden_states[layer_index], model_inputs.attention_mask)
            layer_splits[layer_index] = [hidden[drop_idx:] for hidden in split_hidden]

        image_token_mask = self._get_image_token_mask(model_inputs).to(model_inputs.attention_mask.device)
        image_token_splits = self._extract_masked_hidden(image_token_mask, model_inputs.attention_mask)
        image_token_splits = [image_tokens[drop_idx:] for image_tokens in image_token_splits]

        merge_size = self.processor.image_processor.merge_size
        token_counts = compute_image_token_counts(model_inputs.image_grid_thw, merge_size)
        token_counts_per_sample = group_image_token_counts(token_counts, num_images_per_sample)

        return self._build_query_prompt_inputs(
            layer_splits=layer_splits,
            image_token_splits=image_token_splits,
            token_counts_per_sample=token_counts_per_sample,
            normalized_layer_indices=normalized_layer_indices,
        )

    def _apply_query_prompt(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        query_visual_layers: torch.Tensor,
        query_visual_mask: torch.Tensor,
        query_source_type_ids: torch.Tensor,
    ):
        query_module = self._get_query_prompt_module()
        if query_module is None:
            raise ValueError("query prompt requested but query_prompt_adapter is not available")

        query_device = next(query_module.parameters()).device
        query_dtype = next(query_module.parameters()).dtype
        query_tokens = query_module(
            query_visual_layers.to(device=query_device, dtype=query_dtype),
            query_visual_mask.to(device=query_device),
            query_source_type_ids.to(device=query_device),
        )
        query_tokens = query_tokens.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        query_mask = torch.ones(
            query_tokens.shape[:2],
            dtype=prompt_embeds_mask.dtype,
            device=prompt_embeds_mask.device,
        )

        prompt_embeds = torch.cat([query_tokens, prompt_embeds], dim=1)
        prompt_embeds_mask = torch.cat([query_mask, prompt_embeds_mask], dim=1)
        return prompt_embeds, prompt_embeds_mask

    def prepare_embeddings(self, batch, stage="fit"):
        if "image" in batch:
            batch["image"] = self.preprocess_image_for_vae_encoder(batch["image"])
            batch["width"] = batch["image"].shape[4]
            batch["height"] = batch["image"].shape[3]

        batch_size = batch["image"].shape[0] if "image" in batch else batch["control"].shape[0]
        condition_images_per_sample = [[] for _ in range(batch_size)]
        vae_only_control_keys = self._normalize_control_keys(batch.get("vae_only_control_keys"))
        if "control" not in batch:
            raise ValueError("prepare_embeddings requires main control image under key 'control'")

        if "control" in batch:
            control_image = batch["control"]
            prompt_control = self.preprocess_image_for_text_encoder(control_image, best_resolution=384 * 384)
            for batch_index in range(batch_size):
                condition_images_per_sample[batch_index].append(prompt_control[batch_index])
            batch["control"] = self.preprocess_image_for_vae_encoder(batch["control"])
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
            if additional_control_key not in vae_only_control_keys:
                prompt_control = self.preprocess_image_for_text_encoder(
                    additional_control_image, best_resolution=384 * 384
                )
                for batch_index in range(batch_size):
                    condition_images_per_sample[batch_index].append(prompt_control[batch_index])
            batch[additional_control_key] = self.preprocess_image_for_vae_encoder(batch[additional_control_key])
            batch[f"width_control_{control_index + 1}"] = batch[additional_control_key].shape[4]
            batch[f"height_control_{control_index + 1}"] = batch[additional_control_key].shape[3]

        condition_images = [img for sample_imgs in condition_images_per_sample for img in sample_imgs]
        use_query_prompt = self._use_query_prompt(batch)
        if use_query_prompt:
            prompt_output = self.encode_prompt(
                prompt=batch["prompt"],
                image=condition_images,
                return_dict=True,
                return_query_features=True,
            )
            prompt_embeds, prompt_embeds_mask = self._apply_query_prompt(
                prompt_output["prompt_embeds"],
                prompt_output["prompt_embeds_mask"],
                prompt_output["query_visual_layers"],
                prompt_output["query_visual_mask"],
                prompt_output["query_source_type_ids"],
            )
        else:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt=batch["prompt"],
                image=condition_images,
            )

        batch["prompt_embeds_mask"] = prompt_embeds_mask
        batch["prompt_embeds"] = prompt_embeds

        if not use_query_prompt and "negative_prompt" in batch and batch["true_cfg_scale"] > 1:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=batch["negative_prompt"],
                image=condition_images,
            )
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
            control_latents = self.encode_image_latents(
                control,
                batch_size,
                self.num_channels_latents,
                self.weight_dtype,
            )
            batch["control_latents"] = [control_latents]

        for control_index in range(1, num_additional_controls + 1):
            control_key = f"control_{control_index}"
            control = batch[control_key]
            batch_size = control.shape[0]
            control_latents = self.encode_image_latents(
                control,
                batch_size,
                self.num_channels_latents,
                self.weight_dtype,
            )
            batch["control_latents"].append(control_latents)

        if "control_latents" in batch:
            batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)

        if self.config.loss.mask_loss and "mask" in batch:
            mask = batch["mask"]
            height_image, width_image = batch["height"], batch["width"]
            batch["mask"] = resize_mask_bhw(mask, height_image, width_image)
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
        return_dict: bool = False,
        return_query_features: bool = False,
    ):
        device = self.text_encoder.device
        num_images_per_prompt = 1
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if image is not None and len(image) > 0 and isinstance(image[0], torch.Tensor):
            new_images = []
            for img in image:
                image_array = img.permute(1, 2, 0).cpu().numpy().astype("uint8")
                new_images.append(PIL.Image.fromarray(image_array))
            image = new_images

        # Qwen-VL stays frozen, but query prompt training still needs regular tensors
        # that autograd can consume downstream.
        with torch.no_grad():
            prompt_output = self._get_qwen_prompt_embeds(
                prompt,
                image,
                device,
                return_dict=(return_dict or return_query_features),
                return_query_features=return_query_features,
            )

        if return_dict or return_query_features:
            prompt_embeds = prompt_output["prompt_embeds"]
            prompt_embeds_mask = prompt_output["prompt_embeds_mask"]
        else:
            prompt_embeds, prompt_embeds_mask = prompt_output

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        if return_dict or return_query_features:
            prompt_output["prompt_embeds"] = prompt_embeds
            prompt_output["prompt_embeds_mask"] = prompt_embeds_mask
            return prompt_output
        return prompt_embeds, prompt_embeds_mask

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        return_dict: bool = False,
        return_query_features: bool = False,
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
        if return_query_features:
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            final_hidden_states = outputs.hidden_states[-1]
        else:
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

        if not (return_dict or return_query_features):
            return prompt_embeds, encoder_attention_mask

        output_dict = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": encoder_attention_mask,
        }

        if return_query_features:
            if num_images_per_sample <= 0:
                raise ValueError("query prompt requires at least one conditioning image per sample")
            query_visual_layers, query_visual_mask, query_source_type_ids = self._collect_query_prompt_features(
                outputs=outputs,
                model_inputs=model_inputs,
                drop_idx=drop_idx,
                num_images_per_sample=num_images_per_sample,
            )
            output_dict.update(
                {
                    "query_visual_layers": query_visual_layers,
                    "query_visual_mask": query_visual_mask,
                    "query_source_type_ids": query_source_type_ids,
                }
            )

        return output_dict

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
