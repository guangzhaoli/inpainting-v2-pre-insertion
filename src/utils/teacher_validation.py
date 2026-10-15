"""Validation diagnostics that do not assume the source is a paired target GT."""

import numpy as np
import torch
from PIL import Image, ImageFilter


def _subject_tensor(image, mask, size=224):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask = mask.point(lambda x: 255 if x > 127 else 0)
    box = mask.getbbox()
    if box is None:
        return None
    image = Image.composite(image, Image.new("RGB", image.size), mask).crop(box)
    array = np.array(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def teacher_validation_metrics(sample, raw_output, criterion=None, device="cpu", exclusion_radius=0):
    # Always measure raw output: compositing must not make background metrics trivial.
    source = sample.tar_image.convert("RGB")
    scale = min(1.0, 512 / max(source.size))
    size = tuple(max(1, round(s * scale)) for s in source.size)
    pred = np.asarray(raw_output.convert("RGB").resize(size), dtype=np.float32) / 255
    target = np.asarray(source.resize(size), dtype=np.float32) / 255
    mask = sample.tar_mask.convert("L").resize(size, Image.Resampling.NEAREST)
    radius = int(round(exclusion_radius * scale))
    if radius:
        mask = mask.filter(ImageFilter.MaxFilter(2 * radius + 1))
    bg = np.asarray(mask) <= 127
    metrics = {}
    if bg.any():
        metrics["raw_background_mae"] = float(np.abs(pred - target)[bg].mean())
    if criterion is not None and getattr(criterion, "enable_ref_dino", False):
        a = _subject_tensor(raw_output, sample.tar_mask, criterion.dino_input_size)
        b = _subject_tensor(sample.ref_image, sample.ref_mask, criterion.dino_input_size)
        if a is not None and b is not None:
            features = criterion._dino_embed(torch.cat([a, b]).to(device))
            metrics["reference_dino_cosine"] = float(torch.nn.functional.cosine_similarity(features[:1], features[1:]).item())
    return metrics
