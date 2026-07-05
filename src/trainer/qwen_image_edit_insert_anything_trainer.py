"""
Trainer for the Insert Anything mask_prompt task using QwenImageEdit.
"""

from typing import Dict, List, Optional, Union

import PIL.Image
import torch

from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.utils.images import (
    build_preinserted_composite,
    extract_foreground_from_image_and_mask,
    make_image_devisible,
)


class QwenImageEditInsertAnythingTrainer(QwenImageEditPlusTrainer):
    """
    Insert Anything trainer with a single mask_prompt data contract.
    """

    @staticmethod
    def _empty_prompts(prompt, batch_size: int) -> List[str]:
        if isinstance(prompt, list):
            return [""] * len(prompt)
        return [""] * batch_size

    @staticmethod
    def _mask_to_control_image(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError(f"mask control expects [H, W], [B, H, W], or [B, C, H, W], got {tuple(mask.shape)}")

        if mask.shape[1] == 1:
            mask = mask.repeat(1, 3, 1, 1)
        elif mask.shape[1] != 3:
            raise ValueError(f"mask control expects 1 or 3 channels, got {mask.shape[1]}")

        return mask

    def _pre_insertion_cfg(self):
        return self.config.model.pre_insertion

    def _use_pre_insertion(self) -> bool:
        return bool(self._pre_insertion_cfg().enabled)

    def _reference_cfg_cfg(self):
        return self.config.model.reference_cfg

    def _use_reference_cfg(self) -> bool:
        return bool(self._reference_cfg_cfg().enabled)

    def _use_enhanced_mask_prompt(self) -> bool:
        return bool(self._query_prompt_cfg().enabled) or self._use_pre_insertion()

    def _drop_reference_for_training(self, stage: str) -> bool:
        if stage != "fit" or not self._use_reference_cfg():
            return False
        dropout_rate = self._reference_cfg_cfg().train_dropout_rate
        if dropout_rate <= 0:
            return False
        return bool(torch.rand((), device="cpu").item() < dropout_rate)

    def _force_empty_reference_prompt(self) -> bool:
        return self._use_reference_cfg() and bool(self._reference_cfg_cfg().force_empty_prompt)

    def prepare_embeddings(self, batch: Dict, stage="fit") -> Dict:
        """
        Convert Insert Anything mask_prompt data to the parent Qwen-Edit conditioning format.

        Input:
            - ref_image: [B, C, H, W] in [0, 1]
            - ref_mask: [B, H, W] in [0, 1]
            - tar_image: [B, C, H, W] in [0, 1]
            - tar_mask: [B, H, W] in [0, 1]
            - prompt: List[str]
            - category: List[str]
        """
        required_keys = ["ref_image", "ref_mask", "tar_image", "tar_mask", "prompt", "category"]
        missing = [key for key in required_keys if key not in batch or batch[key] is None]
        if missing:
            raise ValueError(f"InsertAnything mask_prompt batch is missing required keys: {missing}")

        ref_image = batch["ref_image"]
        ref_mask = batch["ref_mask"]
        tar_image = batch["tar_image"]
        tar_mask = batch["tar_mask"]
        batch_size = ref_image.shape[0]

        if tar_image.shape[0] != batch_size or tar_mask.shape[0] != batch_size or ref_mask.shape[0] != batch_size:
            raise ValueError("ref_image, ref_mask, tar_image, and tar_mask batch sizes must match")

        if self._use_pre_insertion():
            pre_insertion_cfg = self._pre_insertion_cfg()
            control = build_preinserted_composite(
                target_image=tar_image,
                target_mask=tar_mask,
                ref_image=ref_image,
                ref_mask=ref_mask,
                training=(stage == "fit"),
                scale_jitter_ratio=pre_insertion_cfg.train_scale_jitter_ratio,
                offset_jitter_ratio=pre_insertion_cfg.train_offset_jitter_ratio,
                paste_mode=pre_insertion_cfg.paste_mode,
            )
        else:
            control = tar_image * (1 - tar_mask.unsqueeze(1))

        use_enhanced_mask_prompt = self._use_enhanced_mask_prompt()
        use_reference_cfg = self._use_reference_cfg()
        if use_reference_cfg and use_enhanced_mask_prompt:
            raise ValueError("reference_cfg requires query_prompt.enabled=false and pre_insertion.enabled=false")
        if use_reference_cfg and "negative_prompt" in batch:
            raise ValueError("reference_cfg cannot be combined with negative_prompt text CFG")

        drop_reference = self._drop_reference_for_training(stage)
        force_empty_reference_prompt = self._force_empty_reference_prompt()
        prompt = (
            self._empty_prompts(batch["prompt"], batch_size)
            if use_enhanced_mask_prompt or force_empty_reference_prompt
            else batch["prompt"]
        )
        mask_control = self._mask_to_control_image(tar_mask) if use_enhanced_mask_prompt else None

        parent_batch = {
            "control": control,
            "prompt": prompt,
            "n_controls": 0,
        }

        if not drop_reference:
            parent_batch["control_1"] = ref_image
            parent_batch["n_controls"] += 1

        if mask_control is not None:
            mask_control_key = f"control_{parent_batch['n_controls'] + 1}"
            parent_batch[mask_control_key] = mask_control
            parent_batch["vae_only_control_keys"] = [mask_control_key]
            parent_batch["n_controls"] += 1

        if stage != "predict":
            parent_batch["image"] = tar_image
            if self.config.loss.mask_loss:
                parent_batch["mask"] = tar_mask

        for key in ("height", "width", "num_inference_steps", "guidance"):
            if key in batch:
                parent_batch[key] = batch[key]

        if "true_cfg_scale" in batch:
            parent_batch["true_cfg_scale"] = 1.0 if use_enhanced_mask_prompt else batch["true_cfg_scale"]
        elif use_enhanced_mask_prompt:
            parent_batch["true_cfg_scale"] = 1.0
        else:
            parent_batch["true_cfg_scale"] = 1.0

        if "negative_prompt" in batch and not use_enhanced_mask_prompt:
            parent_batch["negative_prompt"] = batch["negative_prompt"]

        reference_cfg_parent_batch = None
        reference_cfg_scale = float(batch.get("true_cfg_scale", 1.0))
        if stage == "predict" and use_reference_cfg and reference_cfg_scale > 1.0:
            reference_cfg_parent_batch = {
                "control": control,
                "prompt": prompt,
                "n_controls": 0,
                "true_cfg_scale": 1.0,
            }
            for key in ("height", "width", "num_inference_steps", "guidance"):
                if key in batch:
                    reference_cfg_parent_batch[key] = batch[key]

        parent_batch = super().prepare_embeddings(parent_batch, stage=stage)

        for key in ("height", "width", "num_inference_steps", "guidance"):
            if key in batch:
                parent_batch[key] = batch[key]
        parent_batch["true_cfg_scale"] = (
            1.0 if use_enhanced_mask_prompt else batch.get("true_cfg_scale", parent_batch["true_cfg_scale"])
        )

        if reference_cfg_parent_batch is not None:
            reference_cfg_embeddings = super().prepare_embeddings(reference_cfg_parent_batch, stage=stage)
            for key in ("height", "width", "num_inference_steps", "guidance"):
                if key in batch:
                    reference_cfg_embeddings[key] = batch[key]
            parent_batch["reference_cfg_embeddings"] = reference_cfg_embeddings
            parent_batch["reference_cfg_scale"] = reference_cfg_scale

        parent_batch["category"] = batch["category"]
        return parent_batch

    def prepare_predict_batch_data(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        ref_mask: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        src_mask: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        controls_size: Optional[List[int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Union[None, str, List[str]] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        use_native_size: bool = False,
        best_resolution_size: bool = False,
        weight_dtype=torch.bfloat16,
    ) -> dict:
        if image is None:
            raise ValueError("image is required")
        if not isinstance(image, list):
            image = [image]

        if len(image) < 2:
            raise ValueError("mask_prompt prediction requires [src_image, ref_image] plus src_mask")

        src_image = image[0]
        if src_mask is not None:
            src_mask_img = src_mask if not isinstance(src_mask, list) else src_mask[0]
            ref_image = image[1]
            ref_mask_index = 2
        else:
            if len(image) < 3:
                raise ValueError("mask_prompt prediction requires src_mask or image=[src_image, src_mask, ref_image]")
            src_mask_img = image[1]
            ref_image = image[2]
            ref_mask_index = 3

        if len(image) > ref_mask_index:
            ref_mask_img = image[ref_mask_index]
        elif ref_mask is not None:
            ref_mask_img = ref_mask if not isinstance(ref_mask, list) else ref_mask[0]
        elif ref_image.mode == "RGBA":
            ref_mask_img = ref_image.split()[-1]
        else:
            raise ValueError(
                "ref_mask is required for mask_prompt prediction. Provide it as an image input, ref_mask, or RGBA alpha."
            )

        use_enhanced_mask_prompt = self._use_enhanced_mask_prompt()
        use_reference_cfg = self._use_reference_cfg()
        if use_reference_cfg and use_enhanced_mask_prompt:
            raise ValueError("reference_cfg requires query_prompt.enabled=false and pre_insertion.enabled=false")
        if use_reference_cfg and negative_prompt is not None:
            raise ValueError("reference_cfg cannot be combined with negative_prompt text CFG")

        force_empty_reference_prompt = self._force_empty_reference_prompt()
        if prompt is None:
            if not use_enhanced_mask_prompt and not force_empty_reference_prompt:
                raise ValueError("prompt is required when query_prompt and pre_insertion are both disabled")
            prompt = ""
        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) != 1:
            raise ValueError("mask_prompt prediction expects exactly one prompt for one input image")

        ref_image_np, ref_mask_np = extract_foreground_from_image_and_mask(ref_image, ref_mask_img)
        ref_image = PIL.Image.fromarray(ref_image_np)
        ref_mask_img = PIL.Image.fromarray(ref_mask_np, mode="L")

        src_image = make_image_devisible(src_image, self.vae_scale_factor)
        ref_image = make_image_devisible(ref_image, self.vae_scale_factor)

        processed = self.preprocessor.preprocess(
            {
                "image": src_image,
                "mask": src_mask_img,
                "controls": [ref_image],
                "control_masks": [ref_mask_img],
            }
        )

        src_tensor = processed["image"].unsqueeze(0)
        src_mask_tensor = processed["mask"].unsqueeze(0)
        ref_tensor = processed["controls"][0].unsqueeze(0)
        ref_mask_tensor = processed["control_masks"][0].unsqueeze(0)

        data = {
            "ref_image": ref_tensor,
            "ref_mask": ref_mask_tensor,
            "tar_image": src_tensor,
            "tar_mask": src_mask_tensor,
            "prompt": [""] if use_enhanced_mask_prompt or force_empty_reference_prompt else prompt,
            "category": ["predict"],
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": 1.0 if use_enhanced_mask_prompt else true_cfg_scale,
            "guidance": guidance_scale,
        }

        if negative_prompt is not None and not use_enhanced_mask_prompt:
            data["negative_prompt"] = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if height is None or width is None:
            data["height"] = src_tensor.shape[2]
            data["width"] = src_tensor.shape[3]
        else:
            data["height"] = height
            data["width"] = width

        self.weight_dtype = weight_dtype
        return data
