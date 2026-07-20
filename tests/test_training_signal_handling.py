import signal
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.trainer.base_trainer import BaseTrainer


class TrainingSignalHandlingTest(unittest.TestCase):
    def test_sigint_is_graceful_but_sigterm_keeps_default_behavior(self):
        trainer = SimpleNamespace(training_interrupted=False)
        installed = {}

        with patch(
            "src.trainer.base_trainer.signal.signal",
            side_effect=lambda signum, handler: installed.__setitem__(signum, handler),
        ):
            BaseTrainer.setup_signal_handlers(trainer)

        self.assertIn(signal.SIGINT, installed)
        self.assertIs(installed[signal.SIGTERM], signal.SIG_DFL)
        installed[signal.SIGINT](signal.SIGINT, None)
        self.assertTrue(trainer.training_interrupted)

    def test_interrupt_flag_is_propagated_across_ranks(self):
        trainer = SimpleNamespace(
            training_interrupted=False,
            accelerator=SimpleNamespace(num_processes=2, device=torch.device("cpu")),
        )

        def emulate_peer_interrupt(flag, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            flag.fill_(1)

        with (
            patch("src.trainer.base_trainer.torch.distributed.is_available", return_value=True),
            patch("src.trainer.base_trainer.torch.distributed.is_initialized", return_value=True),
            patch(
                "src.trainer.base_trainer.torch.distributed.all_reduce",
                side_effect=emulate_peer_interrupt,
            ) as all_reduce,
        ):
            interrupted = BaseTrainer._sync_training_interrupted(trainer)

        self.assertTrue(interrupted)
        self.assertTrue(trainer.training_interrupted)
        all_reduce.assert_called_once()

    def test_sigint_arriving_during_collective_is_not_overwritten(self):
        trainer = SimpleNamespace(
            training_interrupted=False,
            accelerator=SimpleNamespace(num_processes=2, device=torch.device("cpu")),
        )

        def signal_during_collective(_flag, op=None):
            trainer.training_interrupted = True

        with (
            patch("src.trainer.base_trainer.torch.distributed.is_available", return_value=True),
            patch("src.trainer.base_trainer.torch.distributed.is_initialized", return_value=True),
            patch(
                "src.trainer.base_trainer.torch.distributed.all_reduce",
                side_effect=signal_during_collective,
            ),
        ):
            interrupted = BaseTrainer._sync_training_interrupted(trainer)

        self.assertTrue(interrupted)
        self.assertTrue(trainer.training_interrupted)

    def test_train_epoch_does_not_checkpoint_inside_interrupt_branch(self):
        trainer = SimpleNamespace(
            start_epoch=0,
            resume_batch_in_epoch=0,
            batch_in_epoch=0,
            global_step=0,
            effective_max_train_steps=10,
            training_interrupted=True,
            _set_dataloader_epoch=MagicMock(),
            _sync_training_interrupted=MagicMock(return_value=True),
            save_checkpoint=MagicMock(),
        )

        stopped = BaseTrainer.train_epoch(trainer, epoch=0, train_dataloader=[{}])

        self.assertTrue(stopped)
        trainer.save_checkpoint.assert_not_called()


class _FitHarness:
    load_pretrain_lora_model = classmethod(lambda cls, dit, config, adapter_name: None)

    def __init__(self):
        self.config = SimpleNamespace(
            resume=None,
            model=SimpleNamespace(quantize=False),
            logging=SimpleNamespace(output_dir="unused"),
        )
        self.adapter_name = "default"
        self.dit = object()
        self.start_epoch = 0
        self.num_epochs = 1
        self.global_step = 7
        self.training_interrupted = True
        self.accelerator = SimpleNamespace(
            wait_for_everyone=MagicMock(),
            end_training=MagicMock(),
        )
        self.fps_logger = SimpleNamespace(start=MagicMock(), last_fps=MagicMock(return_value=0.0))
        self.save_checkpoint = MagicMock()
        self.train_epoch = MagicMock(return_value=True)

    setup_signal_handlers = MagicMock()
    setup_accelerator = MagicMock()
    load_model = MagicMock()
    load_additional_modules = MagicMock()
    setup_model_device_train_mode = MagicMock()
    load_resume_progress = MagicMock()
    configure_optimizers = MagicMock()
    setup_criterion = MagicMock()
    setup_validation = MagicMock()
    accelerator_prepare = MagicMock(side_effect=lambda dataloader: dataloader)
    compute_training_schedule = MagicMock()
    configure_lr_scheduler = MagicMock()
    restore_training_state = MagicMock()
    save_train_config = MagicMock()
    setup_progressbar = MagicMock()


class TrainingFinalCheckpointTest(unittest.TestCase):
    def test_fit_saves_last_checkpoint_once_after_interruption(self):
        trainer = _FitHarness()

        with patch("src.trainer.base_trainer.print_model_summary_table"):
            BaseTrainer.fit(trainer, train_dataloader=[])

        trainer.save_checkpoint.assert_called_once_with(0, 7, is_last=True)


if __name__ == "__main__":
    unittest.main()
