"""Differentiable, aligned local detail supervision and interaction masks."""

import torch
import torch.nn.functional as F


def interaction_masks(mask, radius):
    """Return a boundary band and distant background on the original pixel grid."""
    if radius <= 0:
        return torch.zeros_like(mask), 1.0 - mask
    kernel = 2 * int(radius) + 1
    # Rectangular morphology is exactly separable, including edge padding.
    def pool(value):
        vertical = F.max_pool2d(value, (kernel, 1), stride=1, padding=(radius, 0))
        return F.max_pool2d(vertical, (1, kernel), stride=1, padding=(0, radius))

    dilated = pool(mask)
    eroded = -pool(-mask)
    return (dilated - eroded).clamp(0, 1), 1.0 - dilated


def foreground_detail_crops(pred, target, mask, patch_size, count, output_size):
    """Sample aligned patches at foreground pixels without resizing the whole object.

    A detached mixture of uniform foreground occupancy and target gradients
    favors texture/logo details while retaining flat/color regions. The source
    patch geometry is identical for prediction and GT. Indexing keeps the
    prediction graph intact; reference images are never used as pixel GT.
    """
    batch, _, height, width = pred.shape
    patch_h, patch_w = min(patch_size, height), min(patch_size, width)
    crops_pred, crops_target, owners = [], [], []
    luminance = target.detach().float().mean(dim=1)
    dx = F.pad((luminance[:, :, 1:] - luminance[:, :, :-1]).abs(), (0, 1))
    dy = F.pad((luminance[:, 1:, :] - luminance[:, :-1, :]).abs(), (0, 0, 0, 1))
    for b in range(batch):
        fg = mask[b, 0].float()
        if not bool(fg.any()):
            continue
        detail = (dx[b] + dy[b]) * fg
        weights = fg / fg.sum().clamp_min(1) + detail / detail.sum().clamp_min(1e-6)
        centers = torch.multinomial(weights.flatten(), count, replacement=True)
        for center in centers.tolist():
            cy, cx = divmod(center, width)
            y = min(max(0, cy - patch_h // 2), height - patch_h)
            x = min(max(0, cx - patch_w // 2), width - patch_w)
            foreground = mask[b:b+1, :, y:y+patch_h, x:x+patch_w]
            for source, dest in [(pred, crops_pred), (target, crops_target)]:
                patch = source[b:b+1, :, y:y+patch_h, x:x+patch_w] * foreground
                dest.append(F.interpolate(patch, (output_size, output_size), mode="bilinear", align_corners=False))
            owners.append(b)
    if not owners:
        return None, None, []
    return torch.cat(crops_pred), torch.cat(crops_target), owners
