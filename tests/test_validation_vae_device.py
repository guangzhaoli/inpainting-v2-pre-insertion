import unittest
from contextlib import contextmanager
import weakref

import torch

from src.trainer.base_trainer import BaseTrainer
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer


class _LogicalTensor:
    def __init__(self, device):
        self.device = torch.device(device)


class _TrackingDecoder:
    def __init__(self, device="cpu", failures=None):
        self.tensor = _LogicalTensor(device)
        self.moves = []
        self.failures = {torch.device(key): value for key, value in (failures or {}).items()}

    def parameters(self):
        return iter((self.tensor,))

    def buffers(self):
        return iter(())

    def to(self, device):
        device = torch.device(device)
        self.moves.append(device)
        self.tensor.device = device
        failure = self.failures.pop(device, None)
        if failure is not None:
            raise failure
        return self


class _FakeVae:
    def __init__(self, decoder, device):
        self.decoder = decoder
        self.post_quant_conv = _TrackingDecoder(device)


class _TrackedEmbeddings(dict):
    pass


class _LoadedPredictionHarness:
    _tensor_batch_to_pil_images = staticmethod(BaseTrainer._tensor_batch_to_pil_images)

    def __init__(self):
        self.dit = torch.nn.Linear(1, 1)
        self.dit.train()
        self.vae = object()
        self.text_encoder = object()
        self.events = []

    def prepare_predict_batch_data(self, **kwargs):
        self.events.append("prepare")
        return kwargs

    def prepare_embeddings(self, batch, stage):
        self.events.append(f"embeddings:{stage}")
        embeddings = _TrackedEmbeddings(height=2, width=2)
        self.embeddings_ref = weakref.ref(embeddings)
        return embeddings

    def sampling_from_embeddings(self, embeddings):
        self.events.append("sample")
        return torch.zeros(1)

    @contextmanager
    def _loaded_model_decode_context(self):
        self.events.append(f"embeddings_alive={self.embeddings_ref() is not None}")
        self.events.append(f"decode_context_enter:inference={torch.is_inference_mode_enabled()}")
        try:
            yield
        finally:
            self.events.append(f"decode_context_exit:inference={torch.is_inference_mode_enabled()}")

    def decode_vae_latent(self, latents, height, width):
        self.events.append(f"decode:inference={torch.is_inference_mode_enabled()}")
        return torch.zeros((1, 3, height, width), dtype=torch.float32)


class _AliasedLoadedPredictionHarness(_LoadedPredictionHarness):
    def prepare_predict_batch_data(self, **kwargs):
        self.events.append("prepare")
        return _TrackedEmbeddings(kwargs)

    def prepare_embeddings(self, batch, stage):
        self.events.append(f"embeddings:{stage}")
        batch.update(height=2, width=2)
        self.embeddings_ref = weakref.ref(batch)
        return batch


class _FailingSamplingHarness(_LoadedPredictionHarness):
    def sampling_from_embeddings(self, embeddings):
        self.events.append("sample")
        raise RuntimeError("sampling failed")


class _FailingDecodeHarness(_LoadedPredictionHarness):
    def decode_vae_latent(self, latents, height, width):
        self.events.append(f"decode:inference={torch.is_inference_mode_enabled()}")
        raise RuntimeError("decode failed")


class ValidationVaeDeviceTest(unittest.TestCase):
    def _make_trainer(self, decoder, target_device):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.vae = _FakeVae(decoder=decoder, device=target_device)
        return trainer

    def test_loaded_prediction_limits_device_context_to_decode(self):
        trainer = _LoadedPredictionHarness()

        images = BaseTrainer.predict_loaded_model(trainer, marker="sample")

        self.assertEqual(len(images), 1)
        self.assertEqual(
            trainer.events,
            [
                "prepare",
                "embeddings:predict",
                "sample",
                "embeddings_alive=False",
                "decode_context_enter:inference=False",
                "decode:inference=True",
                "decode_context_exit:inference=False",
            ],
        )
        self.assertTrue(trainer.dit.training)

    def test_loaded_prediction_releases_aliased_batch_before_decode(self):
        trainer = _AliasedLoadedPredictionHarness()

        images = BaseTrainer.predict_loaded_model(trainer, marker="sample")

        self.assertEqual(len(images), 1)
        self.assertIn("embeddings_alive=False", trainer.events)
        self.assertTrue(trainer.dit.training)

    def test_loaded_prediction_restores_dit_state_after_sampling_exception(self):
        trainer = _FailingSamplingHarness()

        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            BaseTrainer.predict_loaded_model(trainer, marker="sample")

        self.assertTrue(trainer.dit.training)

    def test_loaded_prediction_restores_dit_state_after_decode_exception(self):
        trainer = _FailingDecodeHarness()

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            BaseTrainer.predict_loaded_model(trainer, marker="sample")

        self.assertTrue(trainer.dit.training)

    def test_decoder_is_restored_after_success(self):
        decoder = _TrackingDecoder("cpu")
        trainer = self._make_trainer(decoder, "cuda:7")

        with trainer._loaded_model_decode_context():
            self.assertEqual(decoder.tensor.device, torch.device("cuda:7"))

        self.assertEqual(decoder.tensor.device, torch.device("cpu"))
        self.assertEqual(decoder.moves, [torch.device("cuda:7"), torch.device("cpu")])

    def test_decoder_is_restored_after_decode_exception(self):
        decoder = _TrackingDecoder("cpu")
        trainer = self._make_trainer(decoder, "cuda:3")

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            with trainer._loaded_model_decode_context():
                self.assertEqual(decoder.tensor.device, torch.device("cuda:3"))
                raise RuntimeError("decode failed")

        self.assertEqual(decoder.tensor.device, torch.device("cpu"))
        self.assertEqual(decoder.moves, [torch.device("cuda:3"), torch.device("cpu")])

    def test_decoder_is_restored_after_target_migration_exception(self):
        decoder = _TrackingDecoder(
            "cpu",
            failures={"cuda:4": RuntimeError("target migration failed")},
        )
        trainer = self._make_trainer(decoder, "cuda:4")

        with self.assertRaisesRegex(RuntimeError, "target migration failed"):
            with trainer._loaded_model_decode_context():
                self.fail("decode context must not run after target migration fails")

        self.assertEqual(decoder.tensor.device, torch.device("cpu"))
        self.assertEqual(decoder.moves, [torch.device("cuda:4"), torch.device("cpu")])

    def test_operation_and_restore_exceptions_are_both_preserved(self):
        decoder = _TrackingDecoder(
            "cpu",
            failures={"cpu": RuntimeError("restore failed")},
        )
        trainer = self._make_trainer(decoder, "cuda:5")

        with self.assertRaises(BaseExceptionGroup) as caught:
            with trainer._loaded_model_decode_context():
                raise ValueError("decode failed")

        self.assertEqual(
            [str(error) for error in caught.exception.exceptions],
            ["decode failed", "restore failed"],
        )
        self.assertIsInstance(caught.exception.exceptions[0], ValueError)
        self.assertIsInstance(caught.exception.exceptions[1], RuntimeError)
        self.assertEqual(decoder.moves, [torch.device("cuda:5"), torch.device("cpu")])

    def test_decoder_already_on_target_device_is_not_moved(self):
        decoder = _TrackingDecoder("cpu")
        trainer = self._make_trainer(decoder, "cpu")

        with trainer._loaded_model_decode_context():
            self.assertEqual(decoder.tensor.device, torch.device("cpu"))

        self.assertEqual(decoder.moves, [])

    def test_mixed_post_quant_conv_devices_are_rejected(self):
        decoder = _TrackingDecoder("cpu")
        trainer = self._make_trainer(decoder, "cpu")
        trainer.vae.post_quant_conv.buffers = lambda: iter((_LogicalTensor("cuda:1"),))

        with self.assertRaisesRegex(RuntimeError, "vae.post_quant_conv spans multiple devices"):
            with trainer._loaded_model_decode_context():
                self.fail("decode context must not run with an ambiguous VAE decode device")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for a real module migration")
    def test_real_decoder_module_moves_to_cuda_and_back(self):
        decoder = torch.nn.Linear(2, 2).cpu()
        target_device = torch.device("cuda", torch.cuda.current_device())
        trainer = self._make_trainer(decoder, target_device)

        with trainer._loaded_model_decode_context():
            self.assertEqual(next(decoder.parameters()).device, target_device)

        self.assertEqual(next(decoder.parameters()).device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
