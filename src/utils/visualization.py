from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def ensure_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def ensure_mask(mask: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(mask).convert("L")


def save_mask(mask: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_mask(mask).save(path)


def build_strict_mask_composite(
    source: Image.Image,
    mask: Image.Image,
    output: Image.Image,
    mask_threshold: float,
) -> Image.Image:
    if not 0 <= mask_threshold < 1:
        raise ValueError("mask_threshold must be in [0, 1)")
    source = ensure_rgb(source)
    mask = ensure_mask(mask)
    output = ensure_rgb(output)
    if mask.size != source.size:
        raise ValueError(
            "strict-mask composite requires source and mask to have identical geometry, "
            f"got source={source.size}, mask={mask.size}"
        )
    if output.size != source.size:
        output = output.resize(source.size, Image.Resampling.LANCZOS)
    binary_mask = binarize_mask(mask, mask_threshold)
    return Image.composite(output, source, binary_mask)


def binarize_mask(mask: Image.Image, mask_threshold: float) -> Image.Image:
    if not 0 <= mask_threshold < 1:
        raise ValueError("mask_threshold must be in [0, 1)")
    mask_array = np.asarray(ensure_mask(mask), dtype=np.uint8)
    threshold_scale = 1.0 if int(mask_array.max()) <= 1 else 255.0
    binary = mask_array > threshold_scale * float(mask_threshold)
    return Image.fromarray(binary.astype(np.uint8) * 255, mode="L")


def build_expanded_soft_mask(
    mask: Image.Image,
    mask_threshold: float,
    dilation_px: int,
    transition_width_px: int,
) -> Image.Image:
    """Build an expanded edit mask with a cosine-decayed outer transition.

    The thresholded mask is first dilated by ``dilation_px`` with an elliptical
    structuring element. Pixels in the expanded region have weight one. Outside
    that region, weights decay from one to zero over ``transition_width_px``.
    """

    if isinstance(dilation_px, bool) or not isinstance(dilation_px, (int, np.integer)):
        raise TypeError("dilation_px must be an integer")
    if dilation_px < 0:
        raise ValueError("dilation_px must be >= 0")
    if isinstance(transition_width_px, bool) or not isinstance(
        transition_width_px,
        (int, np.integer),
    ):
        raise TypeError("transition_width_px must be an integer")
    if transition_width_px < 2:
        raise ValueError("transition_width_px must be >= 2")

    binary = np.asarray(binarize_mask(mask, mask_threshold), dtype=np.uint8) // 255
    if dilation_px > 0:
        kernel_size = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        expanded = cv2.dilate(binary, kernel, iterations=1).astype(np.uint8)
    else:
        expanded = binary.copy()

    weights = expanded.astype(np.float32)
    if expanded.any() and not expanded.all():
        distance = cv2.distanceTransform(
            (1 - expanded).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        transition = (distance > 0) & (distance < float(transition_width_px))
        normalized = distance[transition] / float(transition_width_px)
        weights[transition] = 0.5 * (1.0 + np.cos(np.pi * normalized))

    alpha = np.rint(weights.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(alpha, mode="L")


def build_expanded_soft_mask_composite(
    source: Image.Image,
    mask: Image.Image,
    output: Image.Image,
    mask_threshold: float,
    dilation_px: int,
    transition_width_px: int,
) -> tuple[Image.Image, Image.Image]:
    """Return ``(composite, soft_mask)`` for expanded-soft RGB recomposition."""

    source = ensure_rgb(source)
    mask = ensure_mask(mask)
    output = ensure_rgb(output)
    if mask.size != source.size:
        raise ValueError(
            "expanded-soft composite requires source and mask to have identical geometry, "
            f"got source={source.size}, mask={mask.size}"
        )
    if output.size != source.size:
        output = output.resize(source.size, Image.Resampling.LANCZOS)

    soft_mask = build_expanded_soft_mask(
        mask,
        mask_threshold=mask_threshold,
        dilation_px=dilation_px,
        transition_width_px=transition_width_px,
    )
    return Image.composite(output, source, soft_mask), soft_mask


def make_mask_overlay(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> Image.Image:
    base = ensure_rgb(image)
    mask_l = ensure_mask(mask)
    if mask_l.size != base.size:
        mask_l = mask_l.resize(base.size, Image.Resampling.NEAREST)

    overlay = Image.new("RGB", base.size, color)
    mask_np = (np.array(mask_l, dtype=np.float32) / 255.0 * int(alpha * 255)).astype(np.uint8)
    alpha_mask = Image.fromarray(mask_np, mode="L")
    return Image.composite(overlay, base, alpha_mask)


def draw_centered_label(draw: ImageDraw.ImageDraw, text: str, panel_width: int, y: int):
    bbox = draw.textbbox((0, 0), text)
    x = max(0, (panel_width - (bbox[2] - bbox[0])) // 2)
    draw.text((x, y), text, fill=(20, 20, 20))


def make_panel(image: Image.Image, label: str, panel_size: int) -> Image.Image:
    label_h = 28
    panel = Image.new("RGB", (panel_size, panel_size + label_h), (245, 245, 245))
    image = ensure_rgb(image)
    image.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
    x = (panel_size - image.width) // 2
    y = label_h + (panel_size - image.height) // 2
    panel.paste(image, (x, y))

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size - 1, label_h - 1), fill=(235, 235, 235))
    draw_centered_label(draw, label, panel_size, 7)
    draw.rectangle((0, 0, panel_size - 1, panel_size + label_h - 1), outline=(210, 210, 210))
    return panel


def make_row(panels: Sequence[tuple[str, Image.Image]], panel_size: int) -> Image.Image:
    if not panels:
        raise ValueError("make_row requires at least one panel")
    gap = 8
    rendered = [make_panel(image, label, panel_size) for label, image in panels]
    width = sum(panel.width for panel in rendered) + gap * (len(rendered) - 1)
    height = max(panel.height for panel in rendered)
    row = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in rendered:
        row.paste(panel, (x, 0))
        x += panel.width + gap
    return row


def stack_rows(row_paths: Sequence[Path], output_path: Path, gap: int = 12) -> Image.Image:
    if not row_paths:
        raise ValueError("stack_rows requires at least one row")
    rows = [Image.open(path).convert("RGB") for path in row_paths]
    try:
        width = max(row.width for row in rows)
        height = sum(row.height for row in rows) + gap * (len(rows) - 1)
        sheet = Image.new("RGB", (width, height), (255, 255, 255))
        y = 0
        for row in rows:
            sheet.paste(row, ((width - row.width) // 2, y))
            y += row.height + gap
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
        return sheet
    finally:
        for row in rows:
            row.close()


def make_infer_row(
    ref_image: Image.Image,
    ref_mask: Image.Image,
    source: Image.Image,
    source_mask: Image.Image,
    output: Image.Image,
    composite: Image.Image,
    panel_size: int,
    output_mask: Image.Image | None = None,
) -> Image.Image:
    if output_mask is None:
        output_mask = source_mask
    output_mask = ensure_mask(output_mask).resize(ensure_rgb(output).size, Image.Resampling.NEAREST)
    return make_row(
        [
            ("ref_image", ref_image),
            ("ref_mask_overlay", make_mask_overlay(ref_image, ref_mask)),
            ("source", source),
            ("source_mask_overlay", make_mask_overlay(source, source_mask)),
            ("output", output),
            ("output_mask_overlay", make_mask_overlay(output, output_mask)),
            ("composite", composite),
        ],
        panel_size,
    )
