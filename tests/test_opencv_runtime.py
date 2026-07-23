import os
import random
import unittest
from unittest.mock import call, patch

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import loader
from src.utils import opencv_runtime


class _TinyDataset(Dataset):
    def __init__(self, _config):
        self.values = [0, 1]

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class OpenCVRuntimeTest(unittest.TestCase):
    def setUp(self):
        opencv_runtime._configured_pid = None
        opencv_runtime._configured_threads = None

    def tearDown(self):
        opencv_runtime._configured_pid = None
        opencv_runtime._configured_threads = None

    def test_default_is_one_thread_and_configuration_is_process_local(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(opencv_runtime.os, "getpid", side_effect=[101, 101, 202]),
            patch.object(opencv_runtime.cv2, "setNumThreads") as set_threads,
            patch.object(opencv_runtime.cv2.ocl, "setUseOpenCL") as set_opencl,
        ):
            self.assertEqual(opencv_runtime.configure_opencv_runtime(), 1)
            self.assertEqual(opencv_runtime.configure_opencv_runtime(), 1)
            self.assertEqual(opencv_runtime.configure_opencv_runtime(), 1)

        self.assertEqual(set_threads.call_args_list, [call(1)] * 2)
        self.assertEqual(set_opencl.call_args_list, [call(False)] * 2)

    def test_explicit_positive_thread_override(self):
        with (
            patch.dict(
                os.environ,
                {opencv_runtime.OPENCV_THREADS_ENV: "3"},
                clear=True,
            ),
            patch.object(opencv_runtime.cv2, "setNumThreads") as set_threads,
            patch.object(opencv_runtime.cv2.ocl, "setUseOpenCL") as set_opencl,
        ):
            self.assertEqual(opencv_runtime.configure_opencv_runtime(), 3)

        set_threads.assert_called_once_with(3)
        set_opencl.assert_called_once_with(False)

    def test_invalid_thread_overrides_fail_before_touching_opencv(self):
        invalid_values = ("0", "-2", "1.5", "many", str(2**31))
        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with (
                    patch.dict(
                        os.environ,
                        {opencv_runtime.OPENCV_THREADS_ENV: invalid_value},
                        clear=True,
                    ),
                    patch.object(opencv_runtime.cv2, "setNumThreads") as set_threads,
                ):
                    with self.assertRaisesRegex(ValueError, "positive integer|between"):
                        opencv_runtime.configure_opencv_runtime()
                    set_threads.assert_not_called()

    def test_environment_cannot_change_after_process_configuration(self):
        with (
            patch.dict(
                os.environ,
                {opencv_runtime.OPENCV_THREADS_ENV: "1"},
                clear=True,
            ),
            patch.object(opencv_runtime.os, "getpid", return_value=101),
            patch.object(opencv_runtime.cv2, "setNumThreads"),
            patch.object(opencv_runtime.cv2.ocl, "setUseOpenCL"),
        ):
            opencv_runtime.configure_opencv_runtime()
            os.environ[opencv_runtime.OPENCV_THREADS_ENV] = "2"
            with self.assertRaisesRegex(RuntimeError, "changed from 1 to 2"):
                opencv_runtime.configure_opencv_runtime()

    def test_worker_initializer_does_not_change_rng_states(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(opencv_runtime.cv2, "setNumThreads"),
            patch.object(opencv_runtime.cv2.ocl, "setUseOpenCL"),
        ):
            opencv_runtime.configure_opencv_dataloader_worker(7)

        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state)

    @patch("src.data.dataset.configure_opencv_runtime")
    def test_loader_configures_parent_and_registers_worker_initializer(
        self, configure_parent
    ):
        dataloader = loader(
            class_path="tests.test_opencv_runtime._TinyDataset",
            init_args={},
            batch_size=1,
            num_workers=1,
            persistent_workers=False,
            prefetch_factor=None,
            shuffle=False,
            drop_last=False,
        )

        configure_parent.assert_called_once_with()
        self.assertIs(
            dataloader.worker_init_fn,
            opencv_runtime.configure_opencv_dataloader_worker,
        )


if __name__ == "__main__":
    unittest.main()
