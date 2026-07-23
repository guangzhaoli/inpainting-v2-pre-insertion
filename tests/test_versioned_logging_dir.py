import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.trainer.base_trainer import BaseTrainer


class _LoggingDirHarness:
    _logging_project_dir = BaseTrainer._logging_project_dir
    setup_versioned_logging_dir = BaseTrainer.setup_versioned_logging_dir
    _synchronize_versioned_logging_dir = BaseTrainer._synchronize_versioned_logging_dir

    def __init__(self, output_dir: str, *, is_main_process: bool = True):
        self.config = SimpleNamespace(
            logging=SimpleNamespace(
                output_dir=output_dir,
                tracker_project_name="test-project",
            ),
            trainer="unused",
        )
        self.accelerator = SimpleNamespace(
            is_main_process=is_main_process,
            project_configuration=SimpleNamespace(
                project_dir="initial-project-dir",
                logging_dir="initial-logging-dir",
            ),
        )


class VersionedLoggingDirTest(unittest.TestCase):
    def test_concurrent_jobs_atomically_reserve_distinct_versions(self):
        worker_count = 8
        with TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test-project"
            project_dir.mkdir()
            trainers = [_LoggingDirHarness(tmpdir) for _ in range(worker_count)]

            # Force every simulated job to select v0 before any mkdir succeeds.
            # The retry loop must still allocate one distinct directory per job.
            first_attempt = threading.Barrier(worker_count)
            real_mkdir = os.mkdir

            def racing_mkdir(path, mode=0o777, *, dir_fd=None):
                if Path(path).name == "v0":
                    first_attempt.wait(timeout=5)
                if dir_fd is None:
                    return real_mkdir(path, mode)
                return real_mkdir(path, mode, dir_fd=dir_fd)

            with (
                patch("src.trainer.base_trainer.os.mkdir", side_effect=racing_mkdir),
                ThreadPoolExecutor(max_workers=worker_count) as executor,
            ):
                allocated = list(
                    executor.map(
                        lambda trainer: trainer.setup_versioned_logging_dir(),
                        trainers,
                    )
                )

            self.assertEqual(len(set(allocated)), worker_count)
            self.assertEqual(
                {Path(path).name for path in allocated},
                {f"v{index}" for index in range(worker_count)},
            )
            self.assertTrue(all(Path(path).is_dir() for path in allocated))

    def test_main_process_broadcasts_one_reserved_directory_to_every_worker(self):
        with TemporaryDirectory() as tmpdir:
            main = _LoggingDirHarness(tmpdir, is_main_process=True)
            worker = _LoggingDirHarness(tmpdir, is_main_process=False)
            worker.setup_versioned_logging_dir = Mock(
                side_effect=AssertionError("non-main worker must not reserve a run directory")
            )
            broadcast_value = []

            def emulate_broadcast(values, from_process=0):
                self.assertEqual(from_process, 0)
                if values[0] is not None:
                    broadcast_value[:] = values
                else:
                    values[:] = broadcast_value
                return values

            with patch(
                "src.trainer.base_trainer.broadcast_object_list",
                side_effect=emulate_broadcast,
            ) as broadcast:
                main_dir = main._synchronize_versioned_logging_dir()
                worker_dir = worker._synchronize_versioned_logging_dir()

            self.assertEqual(main_dir, worker_dir)
            self.assertEqual(main.config.logging.output_dir, main_dir)
            self.assertEqual(worker.config.logging.output_dir, main_dir)
            self.assertEqual(main.accelerator.project_configuration.project_dir, main_dir)
            self.assertEqual(worker.accelerator.project_configuration.project_dir, main_dir)
            self.assertEqual(main.accelerator.project_configuration.logging_dir, main_dir)
            self.assertEqual(worker.accelerator.project_configuration.logging_dir, main_dir)
            self.assertEqual(broadcast.call_count, 2)
            worker.setup_versioned_logging_dir.assert_not_called()
            self.assertEqual(
                [entry.name for entry in (Path(tmpdir) / "test-project").iterdir()],
                ["v0"],
            )


if __name__ == "__main__":
    unittest.main()
