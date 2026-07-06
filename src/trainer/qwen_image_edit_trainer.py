import torch
import torch.nn.functional as F  # NOQA
import PIL
import gc
import numpy as np
from typing import Optional, Union, List, Tuple
import logging
from tqdm.auto import tqdm
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    QwenImageEditPipeline,
    calculate_dimensions,
    retrieve_latents,
    randn_tensor,
)

from src.loss.edit_mask_loss import map_mask_to_latent
from src.trainer.base_trainer import BaseTrainer
from src.utils.tools import infer_image_tensor
from src.utils.images import (
    calculate_best_resolution,
    image_adjust_best_resolution,
    make_image_devisible,
    make_image_shape_devisible,
    resize_mask_bhw,
)


class QwenImageEditTrainer(BaseTrainer):
    """Trainer class based on QwenImageEditPipeline"""

    def __init__(self, config):
        """
        the image process passed to vae
        import numpy as np
        def customized_process(img: np.ndarray) -> torch.Tensor:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img.unsqueeze(2)
            img = img / 255
            img = img*2 -1
            return img
        image to the prompt encoder should be [B,C,1,H,W] in range [-1,1]
        image to the text encoder
            [B,C,H,W] tensor in range [0,255], uint8, or PIL.Image RGB mode
        """
        super().__init__(config)
        self.config = config

        # Component attributes
        self.vae = None  # AutoencoderKLQwenImage
        self.text_encoder = None  # Qwen2_5_VLForConditionalGeneration (text_encoder)
        self.dit = None  # QwenImageTransformer2DModel
        self.tokenizer = None  # Qwen2Tokenizer
        self.scheduler = None  # FlowMatchEulerDiscreteScheduler
        self._scheduler_train_timesteps = None
        self._scheduler_train_sigmas = None
        self._scheduler_train_tensor_cache = {}

        # Parameters obtained from VAE configuration
        self.vae_scale_factor = None
        self.vae_latent_mean = None
        self.vae_latent_std = None
        self.vae_z_dim = None
        self.adapter_name = config.model.lora.adapter_name

    def get_pipeline_class(self):
        return QwenImageEditPipeline

    # Static methods: directly reference QwenImageEditPipeline methods
    _pack_latents = staticmethod(QwenImageEditPipeline._pack_latents)
    _unpack_latents = staticmethod(QwenImageEditPipeline._unpack_latents)

    @staticmethod
    def _move_tensor(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        kwargs = {"non_blocking": tensor.device.type == "cpu"}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)

    def _cache_scheduler_training_tensors(self):
        if self.scheduler is None:
            raise ValueError("scheduler must be loaded before caching training timesteps")
        if not hasattr(self.scheduler, "timesteps") or not hasattr(self.scheduler, "sigmas"):
            raise ValueError("scheduler must expose timesteps and sigmas for training")
        self._scheduler_train_timesteps = self.scheduler.timesteps.detach().clone()
        self._scheduler_train_sigmas = self.scheduler.sigmas.detach().clone()
        self._scheduler_train_tensor_cache = {}

    def _get_cached_scheduler_training_tensors(self, device: torch.device, dtype: torch.dtype):
        if self._scheduler_train_timesteps is None or self._scheduler_train_sigmas is None:
            self._cache_scheduler_training_tensors()

        key = (device.type, device.index, dtype)
        cached = self._scheduler_train_tensor_cache.get(key)
        if cached is None:
            cached = (
                self._scheduler_train_timesteps.to(device=device),
                self._scheduler_train_sigmas.to(device=device, dtype=dtype),
            )
            self._scheduler_train_tensor_cache[key] = cached
        return cached

    def load_model(self, text_encoder_device=None):
        """Load and separate components from QwenImageEditPipeline"""
        logging.info("Loading QwenImageEditPipeline and separating components...")

        # Load complete model using pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
        )
        pipe.to("cpu")
        logging.info(f"excution device: {pipe._execution_device}")

        # Separate individual components

        from src.models.load_model import load_vae, load_qwenvl

        self.vae = load_vae("Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype)  # use original one
        # same to model constructed from vae self.vae = pipe.vae
        self.text_encoder = load_qwenvl("Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype)  # use original one
        logging.info(f"text_encoder device: {self.text_encoder.device}")
        # self.dit = pipe.transformer this is same as the following, verified
        from src.models.load_model import load_transformer

        self.dit = load_transformer(
            self.config.model.pretrained_model_name_or_path,  # could use quantized version
            weight_dtype=self.weight_dtype,
        )
        # load_transformer is same as pipe.transformer

        from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )

        self.processor: Qwen2VLProcessor = pipe.processor
        self.tokenizer: Qwen2Tokenizer = pipe.tokenizer
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
        self._cache_scheduler_training_tensors()
        # Initialize image processor (for predict method)
        from diffusers.image_processor import VaeImageProcessor

        # Set VAE-related parameters
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.vae_latent_mean = self.vae.config.latents_mean
        self.vae_latent_std = self.vae.config.latents_std
        self.vae_z_dim = self.vae.config.z_dim

        # Attributes copied from original pipeline
        self.latent_channels = self.vae.config.z_dim
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        self.prompt_template_encode = pipe.prompt_template_encode
        self.prompt_template_encode_start_idx = pipe.prompt_template_encode_start_idx

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.num_channels_latents = self.dit.config.in_channels // 4

        # Set models to training/evaluation mode
        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def preprocess_image_for_vae_encoder(self, image):
        """
        the image process passed to vae
        import numpy as np
        def customized_process(img: np.ndarray) -> torch.Tensor:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img.unsqueeze(2)
            img = img / 255
            img = img*2 -1
            return img
        image to the vae encoder should be [B,C,1,H,W] in range [-1,1]
        """
        # suppose image input is [0,1] [B,C,H,W] after dataloader -> [B,C,1,H,W]
        tensor_info = infer_image_tensor(image)
        if tensor_info["layout"] == "HW":
            image = image.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif tensor_info["layout"] == "CHW":
            image = image.unsqueeze(0).unsqueeze(2)
        elif tensor_info["layout"] == "HWC":
            image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        elif tensor_info["layout"] == "BCHW":
            image = image.unsqueeze(2)
        elif tensor_info["layout"] == "BHWC":
            image = image.permute(0, 3, 1, 2).unsqueeze(2)
        else:
            raise ValueError(f"Invalid image layout: {tensor_info['layout']},{image.shape}")


        if tensor_info["range"] == "0-255":
            image = image / 255.0
            image = 2 * image - 1
        elif tensor_info["range"] == "0-1":
            image = 2 * image - 1
        elif tensor_info["range"] == "-1-1":
            pass
        else:
            raise ValueError(f"Invalid image range: {tensor_info['range']}, {image.min()}, {image.max()}")

        return image

    def preprocess_image_for_text_encoder(self, image, best_resolution=None) -> torch.Tensor:
        """
        the image process passed to text encoder
        image to the text encoder
            [B,C,H,W] tensor in range [0,255], uint8, or PIL.Image RGB mode
        """
        tensor_info = infer_image_tensor(image)
        if tensor_info["layout"] == "HW":
            image = image.unsqueeze(0).unsqueeze(0)
        elif tensor_info["layout"] == "CHW":
            image = image.unsqueeze(0)
        elif tensor_info["layout"] == "HWC":
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif tensor_info["layout"] == "BCHW":
            pass
        elif tensor_info["layout"] == "BHWC":
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Invalid image layout: {tensor_info['layout']}")
        if tensor_info["range"] == "0-255":
            pass
        elif tensor_info["range"] == "0-1":
            image = image * 255
        elif tensor_info["range"] == "-1-1":
            image = (image + 1) / 2 * 255
        else:
            raise ValueError(f"Invalid image range: {tensor_info['range']}")
        if best_resolution is not None:
            new_width, new_height = calculate_best_resolution(image.shape[3], image.shape[2], best_resolution)
            image = F.interpolate(image, size=(new_height, new_width), mode="bilinear")
        return image

    def encode_image_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        dtype,
        generator=None,
    ):
        if image is None:
            raise ValueError("image is required for VAE latent encoding")
        if isinstance(generator, list) and len(generator) != image.shape[0]:
            raise ValueError(
                f"Generator list length {len(generator)} must match image batch size {image.shape[0]}."
            )

        device = self.vae.device
        image = self._move_tensor(image, device=device, dtype=dtype)
        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image, generator=generator)
        else:
            image_latents = image

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )

        image_latent_height, image_latent_width = image_latents.shape[3:]
        return self._pack_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
        )

    def prepare_initial_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator=None,
        latents=None,
        device=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        if device is None:
            device = next(self.dit.parameters()).device
        shape = (batch_size, 1, num_channels_latents, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = self._move_tensor(latents, device=device, dtype=dtype)
        return latents

    def setup_model_device_train_mode(self, stage="fit"):
        """Set model device allocation and train mode."""
        if stage not in ["fit", "predict"]:
            raise ValueError(f"stage must be one of ['fit', 'predict'], but got {stage}")
        if stage == "fit":
            if not hasattr(self, "accelerator"):
                raise ValueError("accelerator must be set before setting model devices")

        if stage == "fit":
            # Training mode: need all encoders
            self.vae.to(self.accelerator.device)
            self.text_encoder.to(self.accelerator.device)
            # self.dit.to(self.accelerator.device)
            self.vae.decoder.to("cpu")

            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False)
            self.dit.train()
            for name, param in self.dit.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif stage == "predict":
            # Predict mode: allocate to different GPUs according to configuration
            devices = self.config.predict.devices
            self.vae.to(devices.vae)
            self.text_encoder.to(devices.text_encoder)
            self.dit.to(devices.dit)
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False).eval()

    def prepare_embeddings(self, batch, stage="fit"):
        """
        used in: training and predict
        Qwen-Edit
            - image_latent : target used for training
            - control_latent: concatenated version
            - height_image, width_image
            - height_control, width_control: no batch dim
            - height_control_1, width_control_1: no batch dim
            ...
        for control image, need to create two copies:
            1. processed for vae encoder
            2. processed for text encoder
        for additional control images, only pass to vae encoder
        predict mode:

            - no `image` key
            - extra: negative_prompt_embeds, negative_prompt_embeds_mask

        """

        if "image" in batch:

            batch["image"] = self.preprocess_image_for_vae_encoder(batch["image"])  # B,3,1,H,W
            batch["width"] = batch["image"].shape[4]
            batch["height"] = batch["image"].shape[3]

        if "control" not in batch:
            raise ValueError("prepare_embeddings requires main control image under key 'control'")

        prompt_control = self.preprocess_image_for_text_encoder(batch["control"])  # [B,C,H,W], uint8, range [0,255]
        batch["prompt_control"] = prompt_control
        batch["control"] = self.preprocess_image_for_vae_encoder(
            batch["control"]
        )  # [B,C,1,H,W], float, range [-1,1]
        batch["width_control"] = batch["control"].shape[4]
        batch["height_control"] = batch["control"].shape[3]

        if isinstance(batch["n_controls"], int):
            num_additional_controls = batch["n_controls"]
        else:
            num_additional_controls = batch["n_controls"][0]

        for i in range(num_additional_controls):
            additional_control_key = f"control_{i+1}"
            if additional_control_key not in batch:
                raise ValueError(f"n_controls={num_additional_controls} requires missing key '{additional_control_key}'")
            batch[additional_control_key] = self.preprocess_image_for_vae_encoder(batch[additional_control_key])
            batch[f"width_control_{i+1}"] = batch[additional_control_key].shape[4]
            batch[f"height_control_{i+1}"] = batch[additional_control_key].shape[3]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=batch["prompt"],
            image=batch["prompt_control"],
        )
        batch["prompt_embeds_mask"] = prompt_embeds_mask
        batch["prompt_embeds"] = prompt_embeds

        if "negative_prompt" in batch and batch['true_cfg_scale'] > 1:
            # only for predict stage
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=batch["negative_prompt"],
                image=batch["prompt_control"],
            )
            batch["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
            batch["negative_prompt_embeds"] = negative_prompt_embeds

        # get latents
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

        for i in range(1, num_additional_controls + 1):
            control_key = f"control_{i}"
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

        # Generate img_shapes for training
        if "image_latents" in batch:
            batch_size = batch["image_latents"].shape[0]
            batch["img_shapes"] = self._get_image_shapes(batch, batch_size)

        return batch

    def _postprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Post-process output images"""
        image = image_tensor.cpu().float()
        image = image.squeeze(2).squeeze(0)  # [C,H,W]
        image = (image / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]
        image = image.permute(1, 2, 0).numpy()  # [H,W,C]
        image = (image * 255).astype(np.uint8)
        return image

    def _get_image_shapes(self, embeddings: dict, batch_size: int) -> List[List[Tuple[int, int, int]]]:
        """
        embeddings comes from prepare_embeddings.
        The image shapes is made of, the first one is the target image shape, second is control, and additional controls
        ```python
        img_shapes = [
                [
                    (1, height_image // self.vae_scale_factor // 2, width_image // self.vae_scale_factor // 2),
                    (1, height_control // self.vae_scale_factor // 2, width_control // self.vae_scale_factor // 2),
                    (1, height_control_1 // self.vae_scale_factor // 2, width_control_1 // self.vae_scale_factor // 2),
                ]
            ]
        ```
        hiehgt_image, width_image, height_control, width_control, height_control_1, width_control_1
        """
        if "height" not in embeddings or "width" not in embeddings:
            raise ValueError("height_image and width_image must be in embeddings")
        img_shapes = []
        height_image = embeddings["height"]
        width_image = embeddings["width"]
        img_shapes.append((1, height_image // self.vae_scale_factor // 2, width_image // self.vae_scale_factor // 2))
        if "height_control" in embeddings and "width_control" in embeddings:
            height_control = embeddings["height_control"]
            width_control = embeddings["width_control"]
            img_shapes.append(
                (1, height_control // self.vae_scale_factor // 2, width_control // self.vae_scale_factor // 2)
            )
            num_additional_controls = (
                embeddings["n_controls"] if isinstance(embeddings["n_controls"], int) else embeddings["n_controls"][0]
            )
            for i in range(num_additional_controls):
                additional_control_key = f"control_{i+1}"
                if additional_control_key in embeddings:
                    height_control_i = embeddings[f"height_control_{i+1}"]
                    width_control_i = embeddings[f"width_control_{i+1}"]
                    img_shapes.append(
                        (
                            1,
                            height_control_i // self.vae_scale_factor // 2,
                            width_control_i // self.vae_scale_factor // 2,
                        )
                    )
        return [img_shapes] * batch_size

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        device = self.accelerator.device
        image_latents = self._move_tensor(embeddings["image_latents"], device=device, dtype=self.weight_dtype)
        control_latents = self._move_tensor(embeddings["control_latents"], device=device, dtype=self.weight_dtype)
        prompt_embeds = self._move_tensor(embeddings["prompt_embeds"], device=device, dtype=self.weight_dtype)
        prompt_embeds_mask = self._move_tensor(embeddings["prompt_embeds_mask"], device=device, dtype=torch.int64)
        img_shapes = embeddings["img_shapes"]
        batch_size = image_latents.shape[0]
        if "mask" in embeddings:
            edit_mask = self._move_tensor(embeddings["mask"], device=device, dtype=torch.float32)
        else:
            edit_mask = None

        with torch.no_grad():
            noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)

            # Sample timesteps
            u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            max_timestep_index = self.scheduler.config.num_train_timesteps - 1
            indices = (u * self.scheduler.config.num_train_timesteps).long().clamp(min=0, max=max_timestep_index)
            schedule_timesteps, _ = self._get_cached_scheduler_training_tensors(device, image_latents.dtype)
            timesteps = schedule_timesteps[indices.to(device=device)]

            sigmas = self._get_sigmas(timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * image_latents + sigmas * noise
            packed_input = torch.cat([noisy_model_input, control_latents], dim=1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        model_pred = self.dit(
            hidden_states=packed_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : image_latents.size(1)]
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - image_latents
        # pred shape [2, 4104, 64], target shape [2, 4104, 64]
        loss = self.forward_loss(model_pred, target, weighting, edit_mask)
        return loss

    def _get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """Calculate sigma values for noise scheduler"""
        schedule_timesteps, sigmas = self._get_cached_scheduler_training_tensors(self.accelerator.device, dtype)
        timesteps = self._move_tensor(timesteps, device=self.accelerator.device)
        if sigmas.numel() < schedule_timesteps.numel():
            raise ValueError(
                f"scheduler sigmas must cover all timesteps: {sigmas.numel()} vs {schedule_timesteps.numel()}"
            )
        if schedule_timesteps[0] <= schedule_timesteps[-1]:
            step_indices = torch.searchsorted(schedule_timesteps, timesteps)
        else:
            reversed_timesteps = torch.flip(schedule_timesteps, dims=[0])
            reverse_indices = torch.searchsorted(reversed_timesteps, timesteps)
            step_indices = schedule_timesteps.numel() - 1 - reverse_indices

        if torch.any(step_indices < 0) or torch.any(step_indices >= schedule_timesteps.numel()):
            raise ValueError("timesteps contain values outside the cached scheduler range")
        matched_timesteps = schedule_timesteps[step_indices]
        if not torch.equal(matched_timesteps, timesteps):
            raise ValueError("timesteps must exactly match cached scheduler timesteps")
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator = None):
        # generator is None by default
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i: i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
    ):
        r"""
        get the embedding of prompt and image via qwen_vl. Support batch inference. For batch inference,
        will pad to largest length of prompt in batch.
        It got grad by default, lets add the inference mode first

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = self.text_encoder.device
        num_images_per_prompt = 1  # 固定为1，支持单图像生成
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(image, torch.Tensor):
            # make sure it is identical with diffuser pipelines
            # when remove this, got relative error 0.49% in prompt embeds
            new_images = []
            for i in range(image.shape[0]):
                img = image[i].permute(1, 2, 0).cpu().numpy().astype('uint8')
                img = PIL.Image.fromarray(img)
                new_images.append(img)
            image = new_images

        with torch.inference_mode():
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        encoder = getattr(self.text_encoder, "model", None)
        with torch.cuda.device(model_inputs.input_ids.device):  # 作用域内的 'cuda' 都指向同一张卡
            if encoder is not None:
                outputs = encoder(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    pixel_values=model_inputs.pixel_values,
                    image_grid_thw=model_inputs.image_grid_thw,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state
            else:
                outputs = self.text_encoder(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    pixel_values=model_inputs.pixel_values,
                    image_grid_thw=model_inputs.image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[-1]

        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def prepare_predict_batch_data(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        additional_controls: Optional[List[PIL.Image.Image]] = None,
        controls_size: Optional[List[int]] = None,  # first one is for main control
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
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        Processing logic:
            if height, width is None: choose the width from prompt_image
        if controls_size is None, use same processing as that in the processor in the config
        if use_native_size = True:
            the prompt image and additional controls will use their own size. That is they will not be resized.
        """
        if prompt_image is None:
            raise ValueError("prompt_image is required")
        if prompt is None:
            raise ValueError("prompt is required")
        if additional_controls is None:
            additional_controls = []

        if isinstance(prompt_image, PIL.Image.Image):
            prompt_image = [prompt_image]
        prompt_image = [make_image_devisible(image, self.vae_scale_factor) for image in prompt_image]

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
            controls_size = [calculate_best_resolution(c_size[0], c_size[1], 1024*1024) for c_size in controls_size]
            logging.info(f'controls_size after best resolution  {controls_size}')

        logging.info(f'controls_size for processing {controls_size}')
        for img in prompt_image:
            # for each image, need to make one copy for text_encoder, another for image_encoder
            # convert to [C,H,W] in range [0,1]
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

        logging.info(f'target shape for generation {width}, {height}')

        if additional_controls:
            n_controls = len(additional_controls[0])
            new_controls = {f"control_{i+1}": [] for i in range(n_controls)}
            # [control_1_batch1, control1_batch2, ..], [control2_batch1, control2_batch2, ..]

            for controls in additional_controls:
                controls = self.preprocessor.preprocess({"controls": controls}, controls_size=controls_size)["controls"]
                for i, control in enumerate(controls):
                    new_controls[f"control_{i+1}"].append(control)
            for i in range(n_controls):
                control_stack = torch.stack(new_controls[f"control_{i+1}"], dim=0)
                data[f"control_{i+1}"] = control_stack
            data["n_controls"] = n_controls
        else:
            data['n_controls'] = 0

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

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Sampling from embeddings. Only handle the latent diffusion steps. Output the final latents. Need
        to decode the latents to images.
        """
        num_inference_steps = embeddings["num_inference_steps"]
        true_cfg_scale = embeddings["true_cfg_scale"]
        control_latents = embeddings["control_latents"]
        prompt_embeds = embeddings["prompt_embeds"]
        prompt_embeds_mask = embeddings["prompt_embeds_mask"]
        batch_size = embeddings["control_latents"].shape[0]
        img_shapes = self._get_image_shapes(embeddings, batch_size)
        height_image = embeddings["height"]
        width_image = embeddings["width"]

        negative_prompt = embeddings["negative_prompt"] if "negative_prompt" in embeddings else None

        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        reference_cfg_embeddings = embeddings.get("reference_cfg_embeddings")
        reference_cfg_scale = float(embeddings.get("reference_cfg_scale", 1.0))
        do_reference_cfg = reference_cfg_embeddings is not None and reference_cfg_scale > 1
        if do_true_cfg and do_reference_cfg:
            raise ValueError("text true CFG and reference_cfg cannot be enabled at the same time")
        logging.info(f"do_true_cfg={do_true_cfg}, has_negative_prompt={has_neg_prompt}, true_cfg_scale={true_cfg_scale}")
        logging.info(f"do_reference_cfg={do_reference_cfg}, reference_cfg_scale={reference_cfg_scale}")
        device = self.dit.device

        if do_true_cfg:
            # 清理显存以确保有足够空间进行 CFG
            torch.cuda.empty_cache()
            logging.info(f"negative_prompt: {negative_prompt}")

            # 临时将 positive prompt embeddings 移到 CPU 以节省显存
            negative_prompt_embeds = embeddings["negative_prompt_embeds"]
            negative_prompt_embeds_mask = embeddings["negative_prompt_embeds_mask"]
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device, dtype=torch.int64)

        if do_reference_cfg:
            reference_cfg_control_latents = reference_cfg_embeddings["control_latents"].to(self.weight_dtype).to(device)
            reference_cfg_prompt_embeds = reference_cfg_embeddings["prompt_embeds"].to(
                device, dtype=self.weight_dtype
            )
            reference_cfg_prompt_embeds_mask = reference_cfg_embeddings["prompt_embeds_mask"].to(
                device, dtype=torch.int64
            )
            if reference_cfg_control_latents.shape[0] != batch_size:
                raise ValueError("reference_cfg batch size must match the positive branch batch size")
            reference_cfg_img_shapes = self._get_image_shapes(reference_cfg_embeddings, batch_size)
            reference_cfg_txt_seq_lens = reference_cfg_prompt_embeds_mask.sum(dim=1).tolist()

        logging.info(f"mask shape: {prompt_embeds_mask.shape}, dtype: {prompt_embeds_mask.dtype}")
        logging.info(f"prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")

        # 5. Prepare latent variables
        num_channels_latents = self.dit.config.in_channels // 4

        latents = self.prepare_initial_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height_image,
            width=width_image,
            device=device,
            dtype=self.weight_dtype,
        )
        image_seq_len = latents.shape[1]

        logging.info(f"image latent got grad: {control_latents.requires_grad}")
        logging.info(f"latents shape: {latents.shape}")
        logging.info(f"num_channels_latents: {num_channels_latents}")
        logging.info(f"image-latent shape: {control_latents.shape}")

        logging.info(f"shape of img_shapes: {img_shapes}")
        logging.info(f"self.vae_scale_factor: {self.vae_scale_factor}")
        logging.info(f"height: {height_image}")
        logging.info(f"width: {width_image}")
        logging.info(f"image_seq_len: {image_seq_len}")

        timesteps, num_inference_steps = self.prepare_predict_timesteps(num_inference_steps, image_seq_len)

        self._num_timesteps = len(timesteps)

        # 处理guidance
        guidance_scale = embeddings["guidance"]
        if self.dit.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if do_true_cfg and negative_prompt_embeds_mask is not None
            else None
        )

        # 7. 降噪循环 (遵循原始pipeline逻辑)
        self.scheduler.set_begin_index(0)
        self.attention_kwargs = {}

        # set to proper device
        prompt_embeds = prompt_embeds.to(device, dtype=self.weight_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device, dtype=torch.int64)

        if do_true_cfg:
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device, dtype=torch.int64)
            negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=self.weight_dtype)

        logging.info(f"timesteps: {timesteps}")
        logging.info(f"num_inference_steps: {num_inference_steps}")
        with torch.inference_mode():
            # progress_bar = tqdm(enumerate(timesteps), total=num_inference_steps, desc="Generating")
            # for i, t in progress_bar:
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Generating"):
                # progress_bar.set_postfix({'timestep': f'{t:.1f}'})
                # progress_bar.update()

                self._current_timestep = t
                latents = latents.to(device, dtype=self.weight_dtype)

                latent_model_input = latents
                if control_latents is not None:
                    latent_model_input = torch.cat([latents, control_latents], dim=1)

                # broadcast to batch dimension
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Usecache_context (如果transformer支持)
                with torch.inference_mode():  # 外层关掉梯度 & 减元数据
                    with self.dit.cache_context("cond"):
                        noise_pred = self.dit(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_pred[:, : latents.size(1)]
                if do_reference_cfg:
                    noise_pred_cpu = noise_pred.cpu()
                    torch.cuda.empty_cache()

                    reference_cfg_latent_model_input = latents
                    if reference_cfg_control_latents is not None:
                        reference_cfg_latent_model_input = torch.cat(
                            [latents, reference_cfg_control_latents], dim=1
                        )

                    with torch.inference_mode():
                        with self.dit.cache_context("reference_uncond"):
                            reference_cfg_noise_pred = self.dit(
                                hidden_states=reference_cfg_latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=reference_cfg_prompt_embeds_mask,
                                encoder_hidden_states=reference_cfg_prompt_embeds,
                                img_shapes=reference_cfg_img_shapes,
                                txt_seq_lens=reference_cfg_txt_seq_lens,
                                attention_kwargs=self.attention_kwargs,
                                return_dict=False,
                            )[0]
                            reference_cfg_noise_pred = reference_cfg_noise_pred[:, : latents.size(1)]

                    noise_pred = noise_pred_cpu.to(device)
                    comb_pred = reference_cfg_noise_pred + reference_cfg_scale * (
                        noise_pred - reference_cfg_noise_pred
                    )

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_norm = torch.clamp(noise_norm, min=1e-6)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                    del reference_cfg_noise_pred, comb_pred, cond_norm, noise_norm
                    torch.cuda.empty_cache()

                if do_true_cfg:
                    # 临时释放正面推理结果的显存，避免两次推理同时占用显存
                    noise_pred_cpu = noise_pred.cpu()
                    torch.cuda.empty_cache()
                    with torch.inference_mode():  # 外层关掉梯度 & 减元数据
                        with self.dit.cache_context("uncond"):
                            neg_noise_pred = self.dit(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                encoder_hidden_states=negative_prompt_embeds,
                                img_shapes=img_shapes,
                                txt_seq_lens=negative_txt_seq_lens,
                                attention_kwargs=self.attention_kwargs,
                                return_dict=False,
                            )[0]
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                    # 将正面推理结果移回 GPU 进行合并
                    noise_pred = noise_pred_cpu.to(device)
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_norm = torch.clamp(noise_norm, min=1e-6)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                    # 释放中间结果显存
                    del neg_noise_pred, comb_pred, cond_norm, noise_norm
                    torch.cuda.empty_cache()

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
        self._current_timestep = None
        return latents

    def decode_vae_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 8. decode final latents
        latents = latents.to(self.vae.device, dtype=self.weight_dtype)
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        final_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        # 后处理
        final_image = self.image_processor.postprocess(final_image, output_type="pt")
        return final_image
