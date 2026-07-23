import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.nn as nn

from src.loss.region_perceptual_loss import (
    RegionPerceptualLoss,
    _images_to_lpips_range,
    _masked_bbox_crop,
    region_timestep_weights,
)


class _FakeDino(nn.Module):
    """Deterministic stand-in for DINOv2: mean-pool RGB -> fixed-dim embedding."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(3, dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim, 3))

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(-2, -1))
        feats = self.proj(pooled)

        class _Out:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return _Out(feats.unsqueeze(1))


class _PositionalOnlyDino(nn.Module):
    def forward(self, images: torch.Tensor, /):
        return images.mean(dim=(-2, -1))


class _FakeLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).square().mean(dim=(1, 2, 3), keepdim=True) * self.scale


class RegionPerceptualLossTest(unittest.TestCase):
    def test_local_torch_hub_dino_adapter_preserves_pixel_gradients(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "dinov2"
            repo.mkdir()
            (repo / "hubconf.py").touch()
            checkpoint = root / "dinov2_vits14_pretrain.pth"
            checkpoint.touch()

            encoder = _PositionalOnlyDino()
            with patch("torch.hub.load", return_value=encoder) as hub_load:
                criterion = RegionPerceptualLoss(
                    fg_metric="none",
                    enable_ref_dino=True,
                    dino_backend="torch_hub",
                    dino_model_name="dinov2_vits14",
                    dino_repo_path=str(repo),
                    dino_checkpoint_path=str(checkpoint),
                )
                hub_load.assert_not_called()
                pixels = torch.rand(1, 3, 16, 16, requires_grad=True)
                embedding = criterion._dino_embed(pixels)
                hub_load.assert_called_once_with(
                    str(repo.resolve()),
                    "dinov2_vits14",
                    source="local",
                    pretrained=True,
                    weights=str(checkpoint.resolve()),
                )
                embedding.sum().backward()
                self.assertIsNotNone(pixels.grad)
                self.assertTrue(torch.isfinite(pixels.grad).all())

    def test_external_models_are_not_loaded_for_zero_weight_terms(self):
        with (
            patch("src.loss.region_perceptual_loss.build_lpips_module") as build_lpips,
            patch.object(RegionPerceptualLoss, "_load_dinov2") as load_dino,
        ):
            criterion = RegionPerceptualLoss(fg_metric="lpips", enable_ref_dino=True)
            pred = torch.rand(1, 3, 16, 16)
            out = criterion(
                pred,
                pred,
                torch.ones(1, 16, 16),
                fg_weight=0.0,
                bg_weight=0.0,
                ref_dino_weight=0.0,
                pred_range="0-1",
                target_range="0-1",
            )
        build_lpips.assert_not_called()
        load_dino.assert_not_called()
        self.assertEqual(float(out["total"]), 0.0)

    def test_lpips_downsample_alias_maps_to_input_size(self):
        criterion = RegionPerceptualLoss(fg_metric="none", lpips_downsample=128)
        self.assertEqual(criterion.lpips_input_size, 128)

    def test_region_timestep_weights_favor_low_noise(self):
        sigmas = torch.tensor([0.1, 0.5, 0.9]).view(3, 1, 1)
        weights = region_timestep_weights(sigmas, power=2.0, max_sigma=1.0)
        self.assertEqual(tuple(weights.shape), (3,))
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[2]))
        # max_sigma gate
        weights_gated = region_timestep_weights(sigmas, power=1.0, max_sigma=0.4)
        self.assertGreater(float(weights_gated[0]), 0.0)
        self.assertEqual(float(weights_gated[1]), 0.0)
        self.assertEqual(float(weights_gated[2]), 0.0)

    def test_range_conversion_before_masking_for_minus_one_one(self):
        img = torch.full((1, 3, 8, 8), -0.2)
        img[:, :, 2:6, 2:6] = 0.4
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        masked = img * mask
        wrong = _images_to_lpips_range(masked, input_range="auto")
        correct = _images_to_lpips_range(img, input_range="-1-1") * mask
        self.assertAlmostEqual(float(correct[:, :, 3, 3].mean()), 0.4, places=5)
        self.assertNotAlmostEqual(float(wrong[:, :, 3, 3].mean()), 0.4, places=5)

    def test_fg_lpips_and_bg_l1_are_finite(self):
        # pred: VAE-like [-1,1]; target: pixel tar_image [0,1]
        pred = (torch.rand(2, 3, 48, 48) * 2 - 1).requires_grad_(True)
        target = torch.rand(2, 3, 48, 48)
        mask = torch.zeros(2, 48, 48)
        mask[:, 8:40, 8:40] = 1.0
        weights = torch.tensor([1.0, 0.5])
        with patch(
            "src.loss.region_perceptual_loss.build_lpips_module",
            return_value=_FakeLPIPS(),
        ) as build_lpips:
            criterion = RegionPerceptualLoss(
                fg_metric="lpips",
                lpips_net="alex",
                lpips_input_size=64,
            )
            self.assertIsNone(criterion.lpips)
            out = criterion(
                pred,
                target,
                mask,
                fg_weight=1.0,
                bg_weight=1.0,
                pred_range="-1-1",
                target_range="0-1",
                sample_weights=weights,
            )
        build_lpips.assert_called_once()
        self.assertTrue(torch.isfinite(out["total"]))
        self.assertTrue(torch.isfinite(out["fg_lpips"]))
        self.assertTrue(torch.isfinite(out["bg_l1"]))
        self.assertEqual(float(out["ref_dino"].item()), 0.0)
        out["total"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_background_l1_is_zero_when_bg_matches(self):
        criterion = RegionPerceptualLoss(fg_metric="none")
        image = torch.rand(1, 3, 32, 32)
        mask = torch.zeros(1, 32, 32)
        mask[:, 8:24, 8:24] = 1.0
        out = criterion(
            image,
            image.clone(),
            mask,
            fg_weight=0.0,
            bg_weight=1.0,
            pred_range="0-1",
            target_range="0-1",
        )
        self.assertLess(float(out["bg_l1"].item()), 1e-6)
        self.assertLess(float(out["total"].item()), 1e-6)

    def test_background_l1_penalizes_bg_change_only(self):
        criterion = RegionPerceptualLoss(fg_metric="none")
        target = torch.zeros(1, 3, 16, 16)
        pred = target.clone()
        mask = torch.zeros(1, 16, 16)
        mask[:, 4:12, 4:12] = 1.0
        pred[:, :, 0:2, :] = 1.0
        out = criterion(
            pred,
            target,
            mask,
            fg_weight=0.0,
            bg_weight=1.0,
            pred_range="0-1",
            target_range="0-1",
        )
        self.assertGreater(float(out["bg_l1"].item()), 0.05)

        pred2 = target.clone()
        pred2[:, :, 4:12, 4:12] = 1.0
        out2 = criterion(
            pred2,
            target,
            mask,
            fg_weight=0.0,
            bg_weight=1.0,
            pred_range="0-1",
            target_range="0-1",
        )
        self.assertLess(float(out2["bg_l1"].item()), 1e-6)

    def test_empty_mask_sample_does_not_zero_whole_batch(self):
        criterion = RegionPerceptualLoss(fg_metric="none")
        target = torch.zeros(2, 3, 16, 16)
        pred = target.clone()
        mask = torch.zeros(2, 16, 16)
        mask[1, 4:12, 4:12] = 1.0
        pred[1, :, 0:2, :] = 1.0
        out = criterion(
            pred,
            target,
            mask,
            fg_weight=0.0,
            bg_weight=1.0,
            pred_range="0-1",
            target_range="0-1",
        )
        self.assertGreater(float(out["bg_l1"].item()), 0.01)

    def test_zero_sample_weights_zero_region_terms(self):
        criterion = RegionPerceptualLoss(fg_metric="none")
        pred = torch.rand(2, 3, 16, 16, requires_grad=True)
        target = torch.zeros(2, 3, 16, 16)
        mask = torch.zeros(2, 16, 16)
        mask[:, 4:12, 4:12] = 1.0
        out = criterion(
            pred,
            target,
            mask,
            fg_weight=0.0,
            bg_weight=1.0,
            pred_range="0-1",
            target_range="0-1",
            sample_weights=torch.zeros(2),
        )
        self.assertEqual(float(out["total"].item()), 0.0)
        out["total"].backward()
        # Graph-connected zero: grads may be zero but must exist/finite.
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_sample_weights_scale_absolutely_for_batch_size_one(self):
        """w(t) must not cancel under sum(wL)/sum(w) when B=1 (default training)."""
        criterion = RegionPerceptualLoss(fg_metric="none")
        pred = torch.ones(1, 3, 16, 16)
        target = torch.zeros(1, 3, 16, 16)
        mask = torch.zeros(1, 16, 16)
        mask[:, 4:12, 4:12] = 1.0
        pred[:, :, 0:2, :] = 1.0
        out_full = criterion(
            pred,
            target,
            mask,
            fg_weight=0.0,
            bg_weight=1.0,
            pred_range="0-1",
            target_range="0-1",
            sample_weights=torch.tensor([1.0]),
        )
        out_small = criterion(
            pred,
            target,
            mask,
            fg_weight=0.0,
            bg_weight=1.0,
            pred_range="0-1",
            target_range="0-1",
            sample_weights=torch.tensor([0.25]),
        )
        self.assertAlmostEqual(
            float(out_small["bg_l1"].item()),
            0.25 * float(out_full["bg_l1"].item()),
            places=5,
        )

    def test_ref_dino_requires_enable_flag(self):
        criterion = RegionPerceptualLoss(fg_metric="none", enable_ref_dino=False)
        pred = torch.rand(1, 3, 32, 32, requires_grad=True)
        target = torch.rand(1, 3, 32, 32)
        mask = torch.ones(1, 32, 32)
        ref = torch.rand(1, 3, 32, 32)
        with self.assertRaises(RuntimeError):
            criterion(
                pred,
                target,
                mask,
                fg_weight=0.0,
                bg_weight=0.0,
                ref_dino_weight=1.0,
                ref_images=ref,
                ref_mask=mask,
            )

    def test_ref_dino_skips_empty_ref_mask(self):
        fake = _FakeDino(dim=8)
        criterion = RegionPerceptualLoss(
            fg_metric="none",
            enable_ref_dino=True,
            dino_encoder=fake,
            dino_input_size=32,
        )
        pred = torch.full((1, 3, 64, 64), -1.0)
        pred[:, 0, 16:48, 16:48] = 1.0
        tar_mask = torch.zeros(1, 64, 64)
        tar_mask[:, 16:48, 16:48] = 1.0
        ref = torch.zeros(1, 3, 64, 64)
        ref[:, 0, 16:48, 16:48] = 1.0
        empty_ref_mask = torch.zeros(1, 64, 64)
        out = criterion(
            pred,
            pred.clone(),
            tar_mask,
            fg_weight=0.0,
            bg_weight=0.0,
            ref_dino_weight=1.0,
            ref_images=ref,
            ref_mask=empty_ref_mask,
            pred_range="-1-1",
            target_range="-1-1",
            ref_range="0-1",
        )
        self.assertEqual(float(out["ref_dino"].item()), 0.0)

    def test_ref_dino_identity_is_low_for_same_object(self):
        fake = _FakeDino(dim=8)
        criterion = RegionPerceptualLoss(
            fg_metric="none",
            enable_ref_dino=True,
            dino_encoder=fake,
            dino_input_size=32,
        )
        pred = torch.full((1, 3, 64, 64), -1.0)
        pred[:, 0, 16:48, 16:48] = 1.0
        ref = torch.zeros(1, 3, 64, 64)
        ref[:, 0, 16:48, 16:48] = 1.0
        mask = torch.zeros(1, 64, 64)
        mask[:, 16:48, 16:48] = 1.0
        out = criterion(
            pred,
            pred.clone(),
            mask,
            fg_weight=0.0,
            bg_weight=0.0,
            ref_dino_weight=1.0,
            ref_images=ref,
            ref_mask=mask,
            pred_range="-1-1",
            target_range="-1-1",
            ref_range="0-1",
        )
        self.assertLess(float(out["ref_dino"].item()), 1e-4)

    def test_ref_dino_identity_is_higher_for_different_object(self):
        fake = _FakeDino(dim=8)
        criterion = RegionPerceptualLoss(
            fg_metric="none",
            enable_ref_dino=True,
            dino_encoder=fake,
            dino_input_size=32,
        )
        pred = torch.full((1, 3, 64, 64), -1.0)
        pred[:, 0, 16:48, 16:48] = 1.0
        pred = pred.clone().requires_grad_(True)

        ref_same = torch.zeros(1, 3, 64, 64)
        ref_same[:, 0, 16:48, 16:48] = 1.0
        ref_diff = torch.zeros(1, 3, 64, 64)
        ref_diff[:, 2, 16:48, 16:48] = 1.0
        mask = torch.zeros(1, 64, 64)
        mask[:, 16:48, 16:48] = 1.0

        same = criterion(
            pred.detach(),
            pred.detach(),
            mask,
            fg_weight=0.0,
            bg_weight=0.0,
            ref_dino_weight=1.0,
            ref_images=ref_same,
            ref_mask=mask,
            pred_range="-1-1",
            target_range="-1-1",
            ref_range="0-1",
        )
        diff = criterion(
            pred,
            pred.detach(),
            mask,
            fg_weight=0.0,
            bg_weight=0.0,
            ref_dino_weight=1.0,
            ref_images=ref_diff,
            ref_mask=mask,
            pred_range="-1-1",
            target_range="-1-1",
            ref_range="0-1",
        )
        self.assertGreater(float(diff["ref_dino"].item()), float(same["ref_dino"].item()) + 0.05)
        diff["total"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_masked_bbox_crop_shape(self):
        image = torch.rand(2, 3, 40, 50)
        mask = torch.zeros(2, 40, 50)
        mask[0, 5:20, 10:30] = 1.0
        mask[1, 0:10, 0:10] = 1.0
        crop, valid = _masked_bbox_crop(image, mask, out_size=16, pad_ratio=0.0, input_range="0-1")
        self.assertEqual(tuple(crop.shape), (2, 3, 16, 16))
        self.assertTrue(bool(valid.all().item()))


if __name__ == "__main__":
    unittest.main()
