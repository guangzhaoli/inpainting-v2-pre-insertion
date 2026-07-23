import diffusers
import transformers
from src.utils.options import parse_args
from src.utils.logger import get_logger
from src.utils.opencv_runtime import configure_opencv_runtime
from src.utils.seed import seed_everything
from src.data.config import Config

logger = get_logger(__name__, log_level="INFO")


def import_trainer(config: Config):
    trainer_type = config.trainer_type
    if trainer_type == 'QwenImageEdit':
        from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer as Trainer
    elif trainer_type == 'QwenImageEditPlus':
        from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer as Trainer
    elif trainer_type == 'QwenImageEditInsertAnything':
        from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer as Trainer
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")
    return Trainer


def main():
    """使用模块化的 Trainer 类进行训练"""
    # Apply this before trainer construction and dataset loading.  The helper
    # is process-local and idempotent, so every distributed rank is limited.
    configure_opencv_runtime()

    # Import the dataset after OpenCV is configured but before seeding.  This
    # keeps any future import-time side effects from perturbing the seeded
    # training RNG streams.
    from src.data.dataset import loader

    # 解析配置
    config = parse_args()

    # 设置日志级别
    if hasattr(config, 'local_rank') and config.local_rank == 0:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    seed_everything(42)

    # 创建训练器
    Trainer = import_trainer(config)
    trainer = Trainer(config)
    # 加载数据
    batch_size = config.data.batch_size
    shuffle = config.data.shuffle
    teacher_routes = None
    fallback_teacher = None
    distillation = getattr(config, "distillation", None)
    if distillation is not None and distillation.enabled and distillation.balance_tasks:
        teacher_routes = {
            teacher.name: list(teacher.route_categories)
            for teacher in distillation.teachers
        }
        fallback_teacher = distillation.fallback_teacher
    train_dataloader = loader(
        config.data.class_path,
        config.data.init_args,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        shuffle=shuffle,
        drop_last=config.data.drop_last,
        seed=42,
        teacher_routes=teacher_routes,
        fallback_teacher=fallback_teacher,
    )

    # 开始训练
    trainer.fit(train_dataloader)


if __name__ == "__main__":
    main()
