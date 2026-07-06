"""
Abstract Base Trainer for all trainer implementations.
Defines the core interface that all trainers must implement.
"""

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, ProjectConfiguration
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import math
import re
import shutil
import random

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy  # transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
import os
import glob
import json
from functools import partial
import numpy as np
from src.utils.sampling import calculate_shift, retrieve_timesteps
from tqdm import tqdm
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
import logging
import PIL
import signal
import importlib

import safetensors.torch
from src.models.quantize import quantize_model_to_fp8
from src.utils.tools import calculate_sha256_file
from src.data.config import Config
from src.utils.model_summary import print_model_summary_table
from src.utils.lora_utils import FpsLogger
from src.utils.tools import get_git_info
from src.utils.tools import instantiate_class
from src.scheduler.custom_flowmatch_scheduler import FlowMatchEulerDiscreteScheduler
from src.utils.lora_utils import classify_lora_weight, filter_lora_state_keys
from src.trainer.constants import LORA_FILE_BASE_NAME

logger = logging.getLogger(__name__)

OPTIMIZER_FILE_NAME = "optimizer.bin"
SCHEDULER_FILE_NAME = "scheduler.bin"
TRAIN_STATE_FILE_NAME = "state.json"
RANDOM_STATE_FILE_TEMPLATE = "random_states_{process_index}.pkl"
CHECKPOINT_TYPE_LORA_ONLY = "lora_only"
PEFT_LORA_FILE_NAME = "model.safetensors"


def collect_lora_linears(root: nn.Module):
    loras = []
    for m in root.modules():
        if isinstance(m, nn.Linear):
            if (
                hasattr(m, "lora_A")
                or hasattr(m, "lora_B")
                or hasattr(m, "lora_embedding_A")
                or hasattr(m, "lora_embedding_B")
                or hasattr(m, "lora_edit")
            ):
                loras.append(m)
    return loras


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations.
    Defines the core interface that all trainers must implement.
    """

    def __init__(self, config: Config):
        """Initialize trainer with configuration."""
        self.config = config
        self.accelerator: Optional[Accelerator] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.start_epoch = 0
        self.num_epochs = self.config.train.num_epochs
        self.num_update_steps_per_epoch = 0
        self.effective_max_train_steps = self.config.train.max_train_steps
        self.batch_in_epoch = 0
        self.resume_batch_in_epoch = 0
        self.used_accelerate_resume = False
        self.scheduler: FlowMatchEulerDiscreteScheduler = None

        # Common attributes that all trainers should have
        self.weight_dtype = self._resolve_weight_dtype(self.config.train.mixed_precision)
        self.batch_size = self.config.data.batch_size
        self.fps_logger = FpsLogger()
        self.quantize = self.config.model.quantize
        self.adapter_name = self.config.model.lora.adapter_name
        self.predict_setted = False

        # 为save_last_checkpoint功能添加属性
        self.training_interrupted = False
        self.log_model_info()
        self.load_preprocessor()
        self.pipeline_class = self.get_pipeline_class()

    @staticmethod
    def _resolve_weight_dtype(mixed_precision: str) -> torch.dtype:
        if mixed_precision == "fp16":
            return torch.float16
        if mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    @abstractmethod
    def get_pipeline_class(self):
        """return the pipeline class to use the classmethod"""
        raise NotImplementedError

    def load_preprocessor(self):
        class_path = self.config.data.init_args.processor.class_path
        init_args = self.config.data.init_args.processor.init_args
        self.preprocessor = instantiate_class(class_path, init_args)

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__}(config={self.config})"
        return msg

    def setup_signal_handlers(self):
        """设置信号处理器来捕获Ctrl+C中断"""

        def signal_handler(signum, frame):
            logging.info("收到中断信号，准备保存最后一个检查点...")
            self.training_interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def log_model_info(self):
        """Log model information."""
        logger.info(f"Batch Size: {self.batch_size}")

    def setup_versioned_logging_dir(self):
        """设置版本化的日志目录"""
        base_output_dir = self.config.logging.output_dir
        project_name = self.config.logging.tracker_project_name
        if not project_name:
            trainer = self.config.trainer
            project_name = trainer.value if hasattr(trainer, "value") else str(trainer)
        else:
            project_name = str(project_name)

        # Treat output_dir as the project directory when it is already named after the project.
        output_dir_name = os.path.basename(os.path.normpath(base_output_dir))
        if output_dir_name == project_name:
            project_dir = base_output_dir
        else:
            project_dir = os.path.join(base_output_dir, project_name)

        # 如果项目目录不存在，直接使用 v0
        if not os.path.exists(project_dir):
            versioned_dir = os.path.join(project_dir, "v0")
            self.config.logging.output_dir = versioned_dir
            logging.info(f"创建新的训练版本目录: {versioned_dir}")
            return

        # 查找现有版本
        existing_versions = []
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and item.startswith("v") and item[1:].isdigit():
                version_num = int(item[1:])
                existing_versions.append((version_num, item_path))

        if existing_versions:
            next_version = max(version_num for version_num, _ in existing_versions) + 1
        else:
            next_version = 0

        # 创建新版本目录
        versioned_dir = os.path.join(project_dir, f"v{next_version}")
        self.config.logging.output_dir = versioned_dir
        logging.info(f"使用训练版本目录: {versioned_dir}")

    def accelerator_prepare(self, train_dataloader):
        """Prepare accelerator"""
        # 根据配置决定是否启用梯度检查点
        if self.config.train.gradient_checkpointing:
            self.dit.enable_gradient_checkpointing()

        # sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

        plug = getattr(self.accelerator.state, "fsdp_plugin", None)
        if plug is not None:
            from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision

            # torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)

            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)

            plug.use_orig_params = True
            plug.ignored_modules = collect_lora_linears(self.dit)
            plug.limit_all_gathers = True
            # plug.forward_prefetch = False
            # plug.backward_prefetch = BackwardPrefetch.BACKWARD_POST
            plug.forward_prefetch = True
            plug.backward_prefetch = BackwardPrefetch.BACKWARD_PRE  # 边回传边预取
            plug.sync_module_states = True  # 主卡广播初始化，避免不一致
            plug.min_num_params = 20_000_000  # 5_000_000
            if self.accelerator.mixed_precision != "no":
                plug_dtype = self._resolve_weight_dtype(self.accelerator.mixed_precision)
                plug.mixed_precision = MixedPrecision(
                    param_dtype=plug_dtype,
                    reduce_dtype=plug_dtype,
                    buffer_dtype=plug_dtype,
                    cast_forward_inputs=False,
                )
            else:
                plug.mixed_precision = None
            # from src.models.transformer_qwenimage import QwenImageTransformerBlock  # 你的类路径
            # plug.auto_wrap_policy = partial(
            #     transformer_auto_wrap_policy,
            #     transformer_layer_cls={QwenImageTransformerBlock},  # 注意：是关键字参数
            # )
            plug.auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=plug.min_num_params)
            # from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            plug.sharding_strategy = ShardingStrategy.FULL_SHARD  # FULL_SHARD   # SHARD_GRAD_OP

            self.dit = self.dit.to("cpu")
            torch.cuda.empty_cache()
            import gc

            gc.collect()

            self.dit, optimizer, train_dataloader = self.accelerator.prepare(
                self.dit, self.optimizer, train_dataloader
            )
        else:
            self.dit = self.dit.to(self.accelerator.device)
            self.dit, optimizer, train_dataloader = self.accelerator.prepare(
                self.dit, self.optimizer, train_dataloader
            )
        self.optimizer = optimizer

        # Trackers already initialized in setup_accelerator
        return train_dataloader

    def merge_lora(self):
        """Merge LoRA weights into base model"""
        self.dit.merge_adapter()
        logging.info("Merged LoRA weights into base model")

    def destroy_models(self):
        import gc

        if hasattr(self, "text_encoder"):
            self.text_encoder.cpu()
            del self.text_encoder
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.cpu()
            del self.text_encoder_2
        if hasattr(self, "vae"):
            self.vae.cpu()
            del self.vae
        if hasattr(self, "vit"):
            self.vit.cpu()
            del self.vit
        torch.cuda.empty_cache()
        gc.collect()

    def setup_validation(self, train_dataloader):
        """Setup validation"""
        if self.config.logging.sampling.enable:
            raise NotImplementedError(
                "logging.sampling.enable is true, but validation sampling is not implemented in this branch."
            )
        self.validation_sampler = None

    def clip_gradients(self):
        """Clip gradients"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self._trainable_params,
                self.config.train.max_grad_norm,
            )

    def training_step(self, batch: dict) -> torch.Tensor:
        """Training step, batch from dataloader"""
        embeddings = self.prepare_embeddings(batch, stage="fit")
        return self._compute_loss(embeddings)

    @abstractmethod
    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Compute loss. returned the flow matching loss tensor"""
        pass

    def forward_loss(self, model_pred, target, weighting=None, edit_mask=None):
        if edit_mask is None:
            if weighting is None:
                loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
            else:
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
        else:
            # shape torch.Size([4, 864, 1216]) torch.Size([4, 4104, 64]) torch.Size([4, 4104, 64]) torch.Size([4, 1, 1])
            loss = self.criterion(edit_mask, model_pred, target, weighting)
        return loss

    def train_epoch(self, epoch, train_dataloader):
        self._set_dataloader_epoch(train_dataloader, epoch)
        skip_batches = self.resume_batch_in_epoch if epoch == self.start_epoch else 0
        self.batch_in_epoch = skip_batches
        accumulated_loss_sum = None
        accumulated_loss_count = None
        if skip_batches > 0:
            logging.info(f"Resuming epoch {epoch}, skipping {skip_batches} already-consumed batches.")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx < skip_batches:
                continue

            if self.global_step >= self.effective_max_train_steps:
                logger.info(
                    f"Reached effective_max_train_steps={self.effective_max_train_steps}, stopping training."
                )
                return True

            # 检查是否收到中断信号
            if self.training_interrupted:
                logger.info("检测到训练中断信号，保存最后检查点后退出本epoch...")
                # 立刻做一次“last”保存（即使不是 checkpointing_steps 整除）
                self.save_checkpoint(epoch, self.global_step, is_last=True)
                return True

            with self.accelerator.accumulate(self.dit):
                loss = self.training_step(batch)
                loss_for_logging = loss.detach().float()
                if loss_for_logging.ndim != 0:
                    loss_for_logging = loss_for_logging.mean()
                if accumulated_loss_sum is None:
                    accumulated_loss_sum = torch.zeros_like(loss_for_logging)
                    accumulated_loss_count = torch.zeros_like(loss_for_logging)
                accumulated_loss_sum = accumulated_loss_sum + loss_for_logging
                accumulated_loss_count = accumulated_loss_count + torch.ones_like(loss_for_logging)
                self.fps_logger.update(
                    batch_size=self.batch_size * self.accelerator.num_processes,
                    num_tokens=None,
                )
                self.accelerator.backward(loss)
                self.clip_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
                self.batch_in_epoch = batch_idx + 1
                gathered_loss_sum = self.accelerator.gather(accumulated_loss_sum).sum()
                gathered_loss_count = self.accelerator.gather(accumulated_loss_count).sum()
                if gathered_loss_count.item() <= 0:
                    raise RuntimeError("Accumulated loss count must be positive before logging.")
                self.train_loss = (gathered_loss_sum / gathered_loss_count).item()
                accumulated_loss_sum = None
                accumulated_loss_count = None
                self.running_loss = 0.9 * self.running_loss + 0.1 * self.train_loss
                self.update_progressbar(
                    logs={
                        "loss": self.train_loss,
                        "smooth_loss": self.running_loss,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "fps": self.fps_logger.total_fps(),
                    }
                )
                self.save_checkpoint(epoch, self.global_step)
                if self.validation_sampler and self.validation_sampler.should_run_validation(self.global_step):
                    self.fps_logger.pause()
                    try:
                        self.validation_sampler.run_validation_loop(
                            global_step=self.global_step,
                            trainer=self,  # 传入trainer实例
                        )
                    finally:
                        self.fps_logger.resume()
                if self.global_step >= self.effective_max_train_steps:
                    logger.info(
                        f"Reached effective_max_train_steps={self.effective_max_train_steps}, stopping training."
                    )
                    return True
        self.resume_batch_in_epoch = 0
        return False

    @staticmethod
    def _set_dataloader_epoch(train_dataloader, epoch: int):
        """Set deterministic epoch on the dataloader or its sampler when supported."""
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
            return

        sampler = getattr(train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            return

        batch_sampler = getattr(train_dataloader, "batch_sampler", None)
        inner_sampler = getattr(batch_sampler, "sampler", None)
        if hasattr(inner_sampler, "set_epoch"):
            inner_sampler.set_epoch(epoch)

    def load_resume_progress(self):
        """Load global_step/epoch metadata for resume."""
        if self.config.resume is not None:
            st = self._load_train_state(self.config.resume)
            self.global_step = st["global_step"]
            self.start_epoch = st["epoch"]
            self.resume_batch_in_epoch = st.get("batch_in_epoch", 0)
        else:
            self.global_step = 0
            self.start_epoch = 0
            self.resume_batch_in_epoch = 0

    def compute_training_schedule(self, train_dataloader):
        """Compute the effective training length using both step and epoch limits."""
        num_batches = max(1, len(train_dataloader))
        self.num_update_steps_per_epoch = max(
            1,
            math.ceil(num_batches / self.config.train.gradient_accumulation_steps),
        )
        epoch_limited_steps = self.config.train.num_epochs * self.num_update_steps_per_epoch
        self.effective_max_train_steps = min(self.config.train.max_train_steps, epoch_limited_steps)
        self.num_epochs = self.config.train.num_epochs
        remaining_steps = max(0, self.effective_max_train_steps - self.global_step)
        logging.info(
            "Training schedule: %s batches/epoch, %s optimizer steps/epoch, epoch_limited_steps=%s, "
            "effective_max_train_steps=%s, remaining_steps=%s, total_epochs=%s",
            num_batches,
            self.num_update_steps_per_epoch,
            epoch_limited_steps,
            self.effective_max_train_steps,
            remaining_steps,
            self.num_epochs,
        )

    def setup_progressbar(self):
        self.train_loss = 0.0
        self.running_loss = 0.0
        self.progress_bar = tqdm(
            range(self.global_step, self.effective_max_train_steps),
            desc="fit",
            disable=(not self.accelerator.is_local_main_process),
        )

    def update_progressbar(self, logs: dict):
        self.accelerator.log(logs, step=self.global_step)
        logs = {
            "loss": f"{logs['loss']:.3f}",
            "smooth_loss": f"{logs['smooth_loss']:.3f}",
            "lr": f"{logs['lr']:.1e}",
            "epoch": logs["epoch"],
            "fps": f"{logs['fps']:.2f}",
        }
        self.progress_bar.update(1)
        self.global_step += 1
        self.progress_bar.set_postfix(logs)

    def fit(self, train_dataloader):
        """Main training loop implementation."""
        self.setup_signal_handlers()
        self.setup_accelerator()
        self.load_model()
        if self.config.resume is not None:
            self._validate_lora_resume_checkpoint(self.config.resume)
            self.config.model.lora.pretrained_weight = self._resolve_resume_lora_weight(self.config.resume)
            logging.info(f"Loaded checkpoint from {self.config.model.lora.pretrained_weight}")
        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit,
                self.config.predict.devices.dit,
            )
        self.__class__.load_pretrain_lora_model(self.dit, self.config, self.adapter_name)
        self.load_additional_modules(stage="fit")

        self.setup_model_device_train_mode(stage="fit")
        self.load_resume_progress()
        self.configure_optimizers()
        self.setup_criterion()
        self.setup_validation(train_dataloader)

        train_dataloader = self.accelerator_prepare(train_dataloader)
        self.compute_training_schedule(train_dataloader)
        self.configure_lr_scheduler()
        self.restore_training_state()
        logging.info("***** Running training *****")
        print_model_summary_table(self.dit)

        self.fps_logger.start()
        self.save_train_config()
        self.setup_progressbar()
        current_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.num_epochs):
            current_epoch = epoch
            stop_training = self.train_epoch(epoch, train_dataloader)
            if self.training_interrupted:
                break
            if stop_training:
                break

        # 保存最后一个检查点
        self.save_checkpoint(current_epoch, self.global_step, is_last=True)

        logging.info(f"FPS: {self.fps_logger.last_fps()}")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def setup_criterion(self):
        if self.config.loss.mask_loss:
            from src.loss.edit_mask_loss import MaskEditLoss

            self.criterion = MaskEditLoss(
                forground_weight=self.config.loss.forground_weight,
                background_weight=self.config.loss.background_weight,
            )
        else:
            self.criterion = nn.MSELoss()
        self.criterion.to(self.accelerator.device)

    def setup_predict(
        self,
    ):
        if self.predict_setted:
            return
        if not hasattr(self, "dit") or self.vae is None:
            logging.info("Loading model...")
            self.load_model()

        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit,
                self.config.predict.devices.dit,
            )

        if self.config.model.lora.pretrained_weight is not None:
            logging.info("load lora from pretrained weight")
            self.load_pretrain_lora_model(self.dit, self.config, self.config.lora_adapter_name, stage="predict")
        self.load_additional_modules(stage="predict")

        self.setup_model_device_train_mode(stage="predict")
        logging.info("setup_model_device_train_mode done")
        if os.environ.get("SKIP_MODEL_SUMMARY", "0") != "1":
            print_model_summary_table(self.dit)
        self.predict_setted = True
        logging.info("setup_predict done")

    @abstractmethod
    def prepare_predict_batch_data(self, *args, **kwargs) -> dict:
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        """
        pass

    def predict(
        self,
        **kwargs,
    ):
        """Inference/prediction method.
        Prepare the data, prepare the embeddings, sample the latents, decode the latents to images.
        """
        self.setup_predict()
        batch = self.prepare_predict_batch_data(**kwargs)
        embeddings: dict = self.prepare_embeddings(batch, stage="predict")
        target_height = embeddings["height"]
        target_width = embeddings["width"]
        latents = self.sampling_from_embeddings(embeddings)
        image = self.decode_vae_latent(latents, target_height, target_width)
        output_type = kwargs.get("output_type", "pil")
        if output_type == "pil":
            image = image.detach().permute(0, 2, 3, 1).float().cpu().numpy()
            image = (image * 255).round().astype("uint8")
            if image.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in image]
            else:
                pil_images = [PIL.Image.fromarray(image) for image in image]

            return pil_images
        return image

    def setup_accelerator(self):
        """Initialize accelerator and logging configuration"""
        # Setup versioned logging directory
        self.setup_versioned_logging_dir()

        # Set logging_dir to the versioned output directory directly
        # Use project_dir as logging_dir to avoid extra subdirectory creation
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.logging.output_dir,
            logging_dir=self.config.logging.output_dir,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            mixed_precision=self.config.train.mixed_precision,
            log_with=self.config.logging.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            dataloader_config=DataLoaderConfiguration(
                non_blocking=self.config.data.non_blocking,
            ),
        )

        # Initialize tracker with empty project name to avoid subdirectory
        if self.config.logging.report_to == "tensorboard":
            # Create a simple config dict with only basic types for TensorBoard
            try:
                simple_config = {
                    "learning_rate": float(self.config.optimizer.init_args.get("lr", 0.0001)),
                    "batch_size": int(self.config.data.batch_size),
                    "max_train_steps": int(self.config.train.max_train_steps),
                    "num_epochs": int(self.config.train.num_epochs),
                    "gradient_accumulation_steps": int(self.config.train.gradient_accumulation_steps),
                    "mixed_precision": str(self.config.train.mixed_precision),
                    "lora_r": int(self.config.model.lora.r),
                    "lora_alpha": int(self.config.model.lora.lora_alpha),
                    "model_name": str(self.config.model.pretrained_model_name_or_path),
                    "checkpointing_steps": int(self.config.train.checkpointing_steps),
                }
                self.accelerator.init_trackers("", config=simple_config)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize tensorboard tracker: {e}") from e
        logging.info(f"Number of devices used in DDP training: {self.accelerator.num_processes}")

        # Set weight data type
        self.weight_dtype = self._resolve_weight_dtype(self.accelerator.mixed_precision)

        # Create output directory
        if self.accelerator.is_main_process and self.config.logging.output_dir is not None:
            os.makedirs(self.config.logging.output_dir, exist_ok=True)

        logging.info(f"Mixed precision: {self.accelerator.mixed_precision}")

    def save_checkpoint(self, epoch, global_step, is_last=False):
        """Save checkpoint"""
        self.fps_logger.pause()
        if not is_last and (self.global_step % self.config.train.checkpointing_steps != 0):
            self.fps_logger.resume()
            return
        save_path = os.path.join(self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}")
        if is_last:
            save_path = os.path.join(self.config.logging.output_dir, f"checkpoint-last-{epoch}-{global_step}-last")

        if self.accelerator.is_main_process:
            logging.info(f"Saving checkpoint to {save_path}")
            os.makedirs(save_path, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self._save_random_state(save_path)

        if self.accelerator.is_main_process:
            self.save_lora(save_path, adapter_name=self.adapter_name)
            self.save_additional_modules(save_path)
            torch.save(self.optimizer.state_dict(), os.path.join(save_path, OPTIMIZER_FILE_NAME))
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(save_path, SCHEDULER_FILE_NAME))

            state_info = {
                "checkpoint_type": CHECKPOINT_TYPE_LORA_ONLY,
                "global_step": global_step,
                "epoch": epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "is_last": is_last,
                "used_accelerate_state": False,
                "num_processes": self.accelerator.num_processes,
                "random_state_pattern": RANDOM_STATE_FILE_TEMPLATE,
            }
            if is_last:
                git_info = get_git_info()
                state_info.update(git_info)
            with open(os.path.join(save_path, TRAIN_STATE_FILE_NAME), "w") as f:
                json.dump(state_info, f)
            self._enforce_checkpoints_total_limit()
        self.accelerator.wait_for_everyone()
        self.fps_logger.resume()

    @staticmethod
    def _torch_load(path: str, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _load_train_state(checkpoint_dir: str) -> dict:
        state_path = os.path.join(checkpoint_dir, TRAIN_STATE_FILE_NAME)
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing training state file: {state_path}")
        with open(state_path) as f:
            return json.load(f)

    def _random_state_path(self, checkpoint_dir: str, process_index: int | None = None) -> str:
        if process_index is None:
            process_index = self.accelerator.process_index if self.accelerator is not None else 0
        return os.path.join(
            checkpoint_dir,
            RANDOM_STATE_FILE_TEMPLATE.format(process_index=process_index),
        )

    def _save_random_state(self, checkpoint_dir: str):
        state = {
            "process_index": self.accelerator.process_index,
            "random_state": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
            "torch_manual_seed": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
        torch.save(state, self._random_state_path(checkpoint_dir))

    def _load_random_state(self, checkpoint_dir: str):
        random_state_path = self._random_state_path(checkpoint_dir)
        if not os.path.exists(random_state_path):
            raise FileNotFoundError(f"Missing random state file for this rank: {random_state_path}")

        state = self._torch_load(random_state_path, map_location="cpu")
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_seed"])
        torch.set_rng_state(state["torch_manual_seed"])
        if torch.cuda.is_available() and "torch_cuda_manual_seed" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_manual_seed"])

    def _validate_lora_resume_checkpoint(self, checkpoint_dir: str):
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint_dir}")
        train_state = self._load_train_state(checkpoint_dir)
        self._resolve_resume_lora_weight(checkpoint_dir)
        saved_num_processes = train_state.get("num_processes")
        if (
            saved_num_processes is not None
            and self.accelerator is not None
            and int(saved_num_processes) != self.accelerator.num_processes
        ):
            logging.warning(
                "Resuming LoRA-only checkpoint with a different process count: saved=%s current=%s. "
                "This is allowed, but data order and RNG continuity may differ.",
                saved_num_processes,
                self.accelerator.num_processes,
            )

        required_files = [
            OPTIMIZER_FILE_NAME,
            SCHEDULER_FILE_NAME,
            os.path.basename(self._random_state_path(checkpoint_dir)),
        ]
        missing = [name for name in required_files if not os.path.exists(os.path.join(checkpoint_dir, name))]
        if missing:
            raise FileNotFoundError(f"Resume checkpoint {checkpoint_dir} is missing required files: {missing}")

    def _enforce_checkpoints_total_limit(self):
        """Keep only the newest regular checkpoints according to train.checkpoints_total_limit."""
        limit = self.config.train.checkpoints_total_limit
        if limit is None:
            return

        checkpoint_pattern = re.compile(r"^checkpoint-(\d+)-(\d+)$")
        checkpoints = []
        output_dir = self.config.logging.output_dir
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Checkpoint output directory does not exist: {output_dir}")

        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if not os.path.isdir(item_path):
                continue
            match = checkpoint_pattern.fullmatch(item)
            if match is None:
                continue
            epoch = int(match.group(1))
            step = int(match.group(2))
            checkpoints.append((step, epoch, item_path))

        excess = len(checkpoints) - limit
        if excess <= 0:
            return

        for _, _, checkpoint_path in sorted(checkpoints)[:excess]:
            logging.info(f"Removing old checkpoint due to checkpoints_total_limit={limit}: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)

    def save_lora(self, save_folder, adapter_name=None):
        """Save LoRA weights"""
        if save_folder.endswith(".safetensors"):
            raise ValueError(f"save_folder must be a directory, got safetensors file path: {save_folder}")
        if self.accelerator is not None:
            unwrapped_transformer = self.accelerator.unwrap_model(self.dit)
        else:
            unwrapped_transformer = self.dit
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod
        adapter_name = self.adapter_name if adapter_name is None else adapter_name

        peft_lora_state_dict = get_peft_model_state_dict(unwrapped_transformer, adapter_name=adapter_name)
        peft_lora_state_dict = {
            key: value.detach().cpu() for key, value in peft_lora_state_dict.items()
        }
        safetensors.torch.save_file(
            peft_lora_state_dict,
            os.path.join(save_folder, PEFT_LORA_FILE_NAME),
        )

        diffusers_lora_state_dict = convert_state_dict_to_diffusers(peft_lora_state_dict)
        self.pipeline_class.save_lora_weights(save_folder, diffusers_lora_state_dict, safe_serialization=True)
        logging.info(f"Saved PEFT and Diffusers LoRA weights to {save_folder}")

    def save_additional_modules(self, save_folder):
        return

    def load_additional_modules(self, stage="fit"):
        return

    def save_train_config(self):
        import yaml

        if self.accelerator is not None and not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        d_json = self.config.model_dump(mode="json", exclude_none=True)
        train_yaml_file = os.path.join(self.config.logging.output_dir, "train_config.yaml")
        os.makedirs(self.config.logging.output_dir, exist_ok=True)
        tmp_train_yaml_file = f"{train_yaml_file}.tmp.{os.getpid()}"
        try:
            with open(tmp_train_yaml_file, "w") as f:
                yaml.dump(d_json, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_train_yaml_file, train_yaml_file)
        finally:
            if os.path.exists(tmp_train_yaml_file):
                os.remove(tmp_train_yaml_file)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        trainable_named_params = [(name, param) for name, param in self.dit.named_parameters() if param.requires_grad]
        lora_layers = [param for _, param in trainable_named_params]
        if len(lora_layers) == 0:
            raise ValueError("No trainable parameters found. Check LoRA/query_prompt configuration before training.")

        # Log how many parameters are in training and show one example
        if (getattr(self, "accelerator", None) is None) or self.accelerator.is_main_process:
            total_elements = sum(p.numel() for p in lora_layers)
            logging.info(f"Trainable parameters: {len(lora_layers)} tensors, total elements: {total_elements}")
            if len(trainable_named_params) > 0:
                example_name, example_param = trainable_named_params[0]
                logging.info(f"Example trainable param: {example_name}, shape={tuple(example_param.shape)}")
                logging.info(f"Example dtype: {example_param.dtype}")

        # Use optimizer parameters from configuration
        optimizer_config = self.config.optimizer.init_args
        class_path = self.config.optimizer.class_path
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        logging.info(f"Using optimizer: {cls}, {class_path}")
        self.optimizer = cls(
            lora_layers,
            **optimizer_config,
        )
        # Cache trainable params for gradient clipping: avoids re-traversing the
        # full (mostly frozen) DiT parameter set every step.
        self._trainable_params = lora_layers

    def configure_lr_scheduler(self):
        """Configure LR scheduler after the exact steps/epoch is known."""
        from diffusers.optimization import get_scheduler

        scheduler_type = self.config.lr_scheduler.scheduler_type
        scheduler_kwargs = {
            "optimizer": self.optimizer,
            "num_warmup_steps": self.config.lr_scheduler.warmup_steps,
            "num_training_steps": self.effective_max_train_steps,
        }
        if scheduler_type in {"cosine", "cosine_with_restarts"}:
            scheduler_kwargs["num_cycles"] = self.config.lr_scheduler.num_cycles
        if scheduler_type == "polynomial":
            scheduler_kwargs["power"] = self.config.lr_scheduler.power

        self.lr_scheduler = get_scheduler(
            scheduler_type,
            **scheduler_kwargs,
        )

    def restore_training_state(self):
        """Restore LoRA-only optimizer, scheduler, and RNG state."""
        self.used_accelerate_resume = False
        if self.config.resume is None:
            return

        optimizer_path = os.path.join(self.config.resume, OPTIMIZER_FILE_NAME)
        scheduler_path = os.path.join(self.config.resume, SCHEDULER_FILE_NAME)

        try:
            optimizer_state = self._torch_load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)
            if self.lr_scheduler is not None:
                scheduler_state = self._torch_load(scheduler_path, map_location="cpu")
                self.lr_scheduler.load_state_dict(scheduler_state)
            self._load_random_state(self.config.resume)
            logging.info(f"Loaded LoRA-only training state from {self.config.resume}")
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA-only training state from {self.config.resume}: {e}") from e

    @staticmethod
    def _resolve_resume_lora_weight(resume_dir: str) -> str:
        """Resolve the LoRA weights file inside a checkpoint directory deterministically.

        LoRA-only checkpoints save PEFT LoRA weights for exact training resume
        and a Diffusers LoRA export for inference.
        """
        preferred_files = [
            os.path.join(resume_dir, PEFT_LORA_FILE_NAME),
            os.path.join(resume_dir, LORA_FILE_BASE_NAME),
        ]
        for path in preferred_files:
            if os.path.exists(path):
                return path

        model_files = [
            path
            for path in sorted(glob.glob(os.path.join(resume_dir, "*.safetensors")))
            if os.path.basename(path) != "query_prompt.safetensors"
        ]
        if len(model_files) == 1:
            return model_files[0]
        if len(model_files) > 1:
            raise ValueError(
                f"Multiple safetensors files found in {resume_dir}; expected one of {preferred_files}, got {model_files}"
            )
        raise FileNotFoundError(f"No LoRA safetensors file found under resume directory: {resume_dir}")

    @classmethod
    def quantize_model(cls, model, device):
        model = quantize_model_to_fp8(
            model,
            engine="bnb",
            verbose=True,
            device=device,
        )
        model = model.to(device)
        return model

    @classmethod
    def add_lora_adapter(cls, transformer: torch.nn.Module, config, adapter_name: str):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            init_lora_weights=config.model.lora.init_lora_weights,
            target_modules=config.model.lora.target_modules,
        )
        logging.info(f"add_lora_adapter: {lora_config}, {adapter_name}")
        transformer.add_adapter(lora_config, adapter_name=adapter_name)
        transformer.set_adapter(adapter_name)

    @classmethod
    def load_pretrain_lora_model(
        cls,
        transformer: torch.nn.Module,
        config,
        adapter_name: str,
        stage="fit",
    ):
        """
        load the pretrained lora model from a local filepath.
        Examples of pretrained_weight:
            - <local_path>/pytorch_lora_weights.safetensors
            - <local_path>/<filename>.safetensors
        >>>
        """
        pretrained_weight = getattr(config.model.lora, "pretrained_weight", None)
        if pretrained_weight:
            if not os.path.exists(pretrained_weight):
                raise FileNotFoundError(f"LoRA weight not found: {pretrained_weight}")

            sha256 = calculate_sha256_file(pretrained_weight)
            logging.info(f"sha256 for pretrained_weight: {sha256}")
            lora_type = classify_lora_weight(pretrained_weight)
            # DIFFUSERS can be loaded directly, otherwise, need to add lora first
            if lora_type != "PEFT":
                transformer.load_lora_adapter(pretrained_weight, adapter_name=adapter_name)
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
            else:
                # add lora first
                # Configure model
                cls.add_lora_adapter(transformer, config, adapter_name)

                state_dict = safetensors.torch.load_file(pretrained_weight)
                loaded_lora_keys = filter_lora_state_keys(state_dict.keys())
                if len(loaded_lora_keys) == 0:
                    raise ValueError(f"PEFT LoRA weight has no LoRA state keys: {pretrained_weight}")

                missing, unexpected = transformer.load_state_dict(
                    state_dict,
                    strict=False,
                )
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
                missing_lora_keys = filter_lora_state_keys(missing)
                if len(missing_lora_keys) > 0:
                    raise ValueError(
                        f"Missing LoRA keys while loading PEFT weights for stage={stage}: "
                        f"{len(missing_lora_keys)}, first_missing={missing_lora_keys[0]}"
                    )
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
                if len(missing) > 0:
                    logging.info(
                        f"non-LoRA missing keys ignored: {len(missing)}, first_missing={missing[0]}"
                    )
                else:
                    logging.info("missing keys: 0")
                # self.load_lora(self.config.model.lora.pretrained_weight)
            logging.info(f"set_lora: Loaded lora from {pretrained_weight}")

        elif stage == "fit":
            cls.add_lora_adapter(transformer, config, adapter_name)

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image from [0,1] to [-1,1]"""
        image = image.to(self.weight_dtype)
        return image * 2.0 - 1.0

    def prepare_predict_timesteps(self, num_inference_steps: int, image_seq_len: int) -> Tuple[torch.Tensor, int]:
        """prepare timesteps for prediction"""
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        device = next(self.dit.parameters()).device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and initialize model components."""
        pass

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings. Qwen-Edit pass image and prompt, Flux-Kontext pass prompt"""
        pass

    @abstractmethod
    def encode_image_latents(self, *args, **kwargs):
        """Encode images or controls with the VAE and pack them for the transformer."""
        pass

    @abstractmethod
    def prepare_initial_latents(self, *args, **kwargs):
        """Prepare random or user-provided initial latents for prediction."""
        pass

    @abstractmethod
    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> Dict[str, torch.Tensor]:
        """Prepare embeddings for prediction. Call vae encoder and prompt encoder
        to get the embeddings. Used in fit & predict
        Update the embeddings keys in batch dict
        """
        return batch

    @abstractmethod
    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Decode VAE latent vectors to RGB images. In range [0,1]"""
        pass

    @abstractmethod
    def sampling_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sampling from embeddings. Only handle the latent diffusion steps. Output the final latents. Need
        to decode the latents to images.
        """
        pass

    @abstractmethod
    def setup_model_device_train_mode(self, stage="fit"):
        pass
