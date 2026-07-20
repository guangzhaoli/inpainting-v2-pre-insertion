import unittest

import torch

from src.loss.edit_mask_loss import MaskEditLoss


class MaskEditLossTest(unittest.TestCase):
    def _mse(self, pred, target):
        return ((pred.float() - target.float()) ** 2)

    def test_region_balanced_ignores_mask_area(self):
        """Small and large masks with same per-region errors should match."""
        B, S, C = 1, 100, 4
        pred = torch.zeros(B, S, C)
        target = torch.zeros(B, S, C)

        # errors: fg=4, bg=1
        # encode by setting pred so that (pred-target)^2 = error
        # use pred = sqrt(error) on all channels (mean over channels still = error)
        pred[:, :, :] = 0.0
        target_small = target.clone()
        target_large = target.clone()

        small_mask = torch.zeros(B, S)
        small_mask[:, :5] = 1.0  # 5%
        large_mask = torch.zeros(B, S)
        large_mask[:, :50] = 1.0  # 50%

        # fg tokens: target = -2 so mse = 4; bg tokens: target = -1 so mse = 1
        for m, t in ((small_mask, target_small), (large_mask, target_large)):
            t[:, m[0] > 0.5] = -2.0
            t[:, m[0] <= 0.5] = -1.0

        crit = MaskEditLoss(forground_weight=2.0, background_weight=1.0, mode="region_balanced")
        loss_small = crit(small_mask, pred, target_small)
        loss_large = crit(large_mask, pred, target_large)

        # (2 * 4 + 1 * 1) / 3 = 3
        expected = torch.tensor(3.0)
        self.assertTrue(torch.allclose(loss_small, expected, atol=1e-5), loss_small.item())
        self.assertTrue(torch.allclose(loss_large, expected, atol=1e-5), loss_large.item())
        self.assertTrue(torch.allclose(loss_small, loss_large, atol=1e-5))

    def test_token_weighted_depends_on_mask_area(self):
        B, S, C = 1, 100, 1
        pred = torch.zeros(B, S, C)
        target = torch.zeros(B, S, C)

        small_mask = torch.zeros(B, S)
        small_mask[:, :5] = 1.0
        large_mask = torch.zeros(B, S)
        large_mask[:, :50] = 1.0

        # fg mse=4, bg mse=1
        target_small = torch.zeros(B, S, C)
        target_large = torch.zeros(B, S, C)
        target_small[:, small_mask[0] > 0.5] = -2.0
        target_small[:, small_mask[0] <= 0.5] = -1.0
        target_large[:, large_mask[0] > 0.5] = -2.0
        target_large[:, large_mask[0] <= 0.5] = -1.0

        crit = MaskEditLoss(forground_weight=2.0, background_weight=1.0, mode="token_weighted")
        loss_small = crit(small_mask, pred, target_small)
        loss_large = crit(large_mask, pred, target_large)

        # small: (5*2*4 + 95*1*1) / (10+95) = (40+95)/105 = 135/105
        # large: (50*2*4 + 50*1*1) / (100+50) = (400+50)/150 = 450/150 = 3
        self.assertTrue(torch.allclose(loss_small, torch.tensor(135.0 / 105.0), atol=1e-5))
        self.assertTrue(torch.allclose(loss_large, torch.tensor(3.0), atol=1e-5))
        self.assertGreater(abs(loss_small.item() - loss_large.item()), 0.5)

    def test_region_balanced_fallback_all_foreground(self):
        B, S, C = 2, 8, 2
        mask = torch.ones(B, S)
        pred = torch.zeros(B, S, C)
        target = torch.full((B, S, C), -3.0)  # mse = 9
        crit = MaskEditLoss(forground_weight=2.0, background_weight=1.0, mode="region_balanced")
        loss = crit(mask, pred, target)
        self.assertTrue(torch.allclose(loss, torch.tensor(9.0), atol=1e-5))

    def test_region_balanced_fallback_all_background(self):
        B, S, C = 2, 8, 2
        mask = torch.zeros(B, S)
        pred = torch.zeros(B, S, C)
        target = torch.full((B, S, C), -2.0)  # mse = 4
        crit = MaskEditLoss(forground_weight=2.0, background_weight=1.0, mode="region_balanced")
        loss = crit(mask, pred, target)
        self.assertTrue(torch.allclose(loss, torch.tensor(4.0), atol=1e-5))

    def test_weighting_is_applied(self):
        B, S, C = 1, 4, 1
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        pred = torch.zeros(B, S, C)
        target = torch.tensor([[[-2.0], [-2.0], [-1.0], [-1.0]]])  # mse 4,4,1,1
        weighting = torch.tensor([[[2.0], [2.0], [2.0], [2.0]]])  # double all
        crit = MaskEditLoss(forground_weight=1.0, background_weight=1.0, mode="region_balanced")
        loss = crit(mask, pred, target, weighting=weighting)
        # means become 8 and 2; mix (1*8+1*2)/2 = 5
        self.assertTrue(torch.allclose(loss, torch.tensor(5.0), atol=1e-5))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            MaskEditLoss(mode="not_a_mode")


if __name__ == "__main__":
    unittest.main()
