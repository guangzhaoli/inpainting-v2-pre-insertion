#!/usr/bin/env python3
"""Build a paper-ready qualitative comparison grid from review_keep_export cases.

Default layout (rows = cases, columns = Reference | Source | methods | Ours):
  - letterbox/cover-fit every image into a fixed cell for alignment
  - Source uses a cyan semi-transparent mask fill + white outline
  - column headers bottom-aligned (Ours bold only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = REPO_ROOT / "output" / "review_keep_export"
DEFAULT_OUT = REPO_ROOT / "output" / "paper_figures" / "qualitative_keep5"

# Paper-facing method order. Keys match filenames under results/*.png (without .png).
METHOD_SPECS: tuple[tuple[str, str], ...] = (
    ("anydoor", "AnyDoor"),
    ("insert_anything", "Insert Anything"),
    ("qwen_edit_2511", "Qwen-Edit"),
    ("qwen_image_2_0", "Qwen-Image-2.0"),
    ("flux_kontext", "FLUX Kontext"),
    ("a2edit_moe", "A2Edit-MoE"),
    ("ours", "Ours"),
)

DEFAULT_CASES: tuple[str, ...] = (
    "anyinsertion/ai_gar_garment_006",
    "anyinsertion/ai_obj_object_000",
    "anyinsertion/ai_obj_object_038",
    "paperinsert_900/per_work_object_handling_034",
    "testset_20260120/accessory__6",
)

ROW_LABELS = {
    "anyinsertion/ai_gar_garment_006": "Garment",
    "anyinsertion/ai_obj_object_000": "Object",
    "anyinsertion/ai_obj_object_038": "Object",
    "paperinsert_900/per_work_object_handling_034": "Object",
    "testset_20260120/accessory__6": "Accessory",
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _find_first(case_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        p = case_dir / name
        if p.is_file():
            return p
    return None


def _binarize_mask(mask: Image.Image, threshold: int = 128) -> Image.Image:
    gray = mask.convert("L")
    return gray.point(lambda v: 255 if v >= threshold else 0, mode="L")


def _mask_overlay(
    image: Image.Image,
    mask: Image.Image,
    *,
    color: tuple[int, int, int] = (0, 220, 255),
    alpha: float = 0.55,
    edge_color: tuple[int, int, int] = (255, 255, 255),
    edge_width: int = 4,
) -> Image.Image:
    """Paper-style mask cue: vivid semi-transparent fill + crisp outline.

    Cyan fill is high-contrast on both dark and bright scenes; white edge
    keeps the insertion region readable after focus-crop + downscale.
    """
    from PIL import ImageChops, ImageFilter

    base = image.convert("RGB")
    m = _binarize_mask(mask)
    if m.size != base.size:
        m = m.resize(base.size, Image.Resampling.NEAREST)

    a = max(0.0, min(1.0, float(alpha)))
    base_rgba = base.convert("RGBA")
    fill = Image.new("RGBA", base.size, (*color, 0))
    fill_alpha = Image.new("L", base.size, 0)
    fill_alpha.paste(int(round(255 * a)), mask=m)
    fill.putalpha(fill_alpha)
    out = Image.alpha_composite(base_rgba, fill)

    # Outer contour for extra visibility when the fill color is close to the scene.
    # MaxFilter/MinFilter require odd kernel sizes.
    if edge_width > 0:
        dil_k = edge_width * 2 + 1
        if dil_k % 2 == 0:
            dil_k += 1
        ero_k = max(3, edge_width if edge_width % 2 == 1 else edge_width + 1)
        dil = m.filter(ImageFilter.MaxFilter(size=dil_k))
        ero = m.filter(ImageFilter.MinFilter(size=ero_k))
        edge = ImageChops.subtract(dil, ero)
        edge_rgb = Image.new("RGB", base.size, edge_color)
        out_rgb = out.convert("RGB")
        out_rgb.paste(edge_rgb, mask=edge)
        return out_rgb
    return out.convert("RGB")


def _mask_bbox(mask: Image.Image, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    m = _binarize_mask(mask)
    if m.size != image_size:
        m = m.resize(image_size, Image.Resampling.NEAREST)
    return m.getbbox()


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a bbox from src image coordinates onto a (possibly different) result size."""
    sx = dst_size[0] / max(1, src_size[0])
    sy = dst_size[1] / max(1, src_size[1])
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    pad_frac: float = 0.12,
    min_pad: int = 16,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    w, h = image_size
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad_x = max(min_pad, int(bw * pad_frac))
    pad_y = max(min_pad, int(bh * pad_frac))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


def _ref_cutout(ref: Image.Image, ref_mask: Image.Image | None) -> Image.Image:
    """Reference object on white, masked if mask available; tight crop for larger subject."""
    rgb = ref.convert("RGB")
    if ref_mask is None:
        return rgb
    m = _binarize_mask(ref_mask)
    if m.size != rgb.size:
        m = m.resize(rgb.size, Image.Resampling.NEAREST)
    bbox = m.getbbox()
    if bbox is None:
        return rgb
    x0, y0, x1, y1 = _expand_bbox(bbox, rgb.size, pad_frac=0.08, min_pad=12)
    crop = rgb.crop((x0, y0, x1, y1))
    mc = m.crop((x0, y0, x1, y1))
    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, mask=mc)
    return white


def _focus_crop(
    image: Image.Image,
    focus_bbox: tuple[int, int, int, int] | None,
    *,
    pad_frac: float = 0.18,
) -> Image.Image:
    """Crop around mask focus so tall scenes fill the panel more aggressively."""
    rgb = image.convert("RGB")
    if focus_bbox is None:
        return rgb
    box = _expand_bbox(focus_bbox, rgb.size, pad_frac=pad_frac, min_pad=24)
    return rgb.crop(box)


def fit_panel(
    image: Image.Image,
    size: tuple[int, int],
    bg: tuple[int, int, int] = (255, 255, 255),
    mode: str = "cover",
) -> Image.Image:
    """Fit image into a fixed cell.

    - cover: scale-to-fill then center-crop (subjects look larger; paper default)
    - contain: letterbox (no crop; can leave large empty margins on tall/wide images)
    """
    cell_w, cell_h = size
    cell = Image.new("RGB", size, bg)
    if image.size[0] <= 0 or image.size[1] <= 0:
        return cell
    rgb = image.convert("RGB")
    if mode == "contain":
        fitted = ImageOps.contain(rgb, size, method=Image.Resampling.LANCZOS)
        ox = (cell_w - fitted.width) // 2
        oy = (cell_h - fitted.height) // 2
        cell.paste(fitted, (ox, oy))
        return cell
    # cover
    fitted = ImageOps.fit(rgb, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.45))
    return fitted


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill=(20, 20, 20),
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - tw) // 2
    y = y0 + (y1 - y0 - th) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def draw_bottom_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill=(20, 20, 20),
    *,
    pad_bottom: int = 4,
) -> None:
    """Horizontally center text and align its baseline/bottom tightly to the box bottom."""
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = x0 + (x1 - x0 - tw) // 2 - bbox[0]
    # Place so the glyph bottom sits pad_bottom above the box bottom.
    y = y1 - pad_bottom - bbox[3]
    # Keep a tiny top clamp so descenders never go above the header box top.
    if y + bbox[1] < y0:
        y = y0 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def load_case_panels(
    case_dir: Path,
    methods: Sequence[tuple[str, str]],
    *,
    focus_on_mask: bool = True,
) -> dict[str, Image.Image | None]:
    ref_path = _find_first(case_dir, ("ref_image.png", "ref_image.jpg", "ref_image.jpeg", "ref_image.webp"))
    tar_path = _find_first(case_dir, ("tar_image.png", "tar_image.jpg", "tar_image.jpeg", "tar_image.webp"))
    ref_mask_path = _find_first(case_dir, ("ref_mask.png", "ref_mask.jpg"))
    tar_mask_path = _find_first(case_dir, ("tar_mask.png", "tar_mask.jpg"))
    if ref_path is None or tar_path is None:
        raise FileNotFoundError(f"Missing ref/tar image under {case_dir}")

    ref = _open_rgb(ref_path)
    tar = _open_rgb(tar_path)
    ref_mask = _open_rgb(ref_mask_path) if ref_mask_path else None
    tar_mask = _open_rgb(tar_mask_path) if tar_mask_path else None

    focus = _mask_bbox(tar_mask, tar.size) if (focus_on_mask and tar_mask is not None) else None
    # Source column: vivid cyan fill + white edge (clear after paper downscale).
    source_vis = (
        _mask_overlay(tar, tar_mask, color=(0, 220, 255), alpha=0.55, edge_color=(255, 255, 255), edge_width=4)
        if tar_mask is not None
        else tar
    )
    if focus_on_mask:
        source_vis = _focus_crop(source_vis, focus, pad_frac=0.22)

    panels: dict[str, Image.Image | None] = {
        "reference": _ref_cutout(ref, ref_mask),
        "source": source_vis,
    }
    for key, _label in methods:
        p = case_dir / "results" / f"{key}.png"
        if not p.is_file():
            panels[key] = None
            continue
        img = _open_rgb(p)
        # Always apply the same relative mask crop, even when result resolution
        # differs from tar (e.g. ours restored-raw 1024x1536 vs tar 1801x2600).
        if focus_on_mask and focus is not None:
            if img.size != tar.size:
                focus_i = _scale_bbox(focus, tar.size, img.size)
            else:
                focus_i = focus
            img = _focus_crop(img, focus_i, pad_frac=0.22)
        panels[key] = img
    return panels


def _layout_metrics(
    methods: Sequence[tuple[str, str]],
    n_rows: int,
    *,
    cell_w: int,
    cell_h: int,
    gap: int,
    row_label_w: int = 0,
    title: str | None = None,
) -> dict:
    """Shared layout numbers for raster and vector backends."""
    col_specs = [("reference", "Reference"), ("source", "Source"), *list(methods)]
    n_cols = len(col_specs)
    header_font_size = max(36, min(52, int(cell_w * 0.095)))
    font_h_regular = _load_font(header_font_size, bold=False)
    font_h_bold = _load_font(header_font_size, bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    max_text_h = 0
    for key, label in col_specs:
        font = font_h_bold if key == "ours" else font_h_regular
        bb = probe.textbbox((0, 0), label, font=font)
        max_text_h = max(max_text_h, bb[3] - bb[1])
    # Leave a bit more top room so ascenders (Q, A, etc.) are not clipped
    # when rendered as vector text in PDF (matplotlib metrics differ from PIL).
    header_pad_top = 10
    header_pad_bottom = 6
    header_h = max(header_pad_top + max_text_h + header_pad_bottom, 44)
    title_h = 60 if title else 0
    width = row_label_w + n_cols * cell_w + (n_cols + 1) * gap
    height = title_h + header_h + n_rows * cell_h + (n_rows + 1) * gap + gap
    return {
        "col_specs": col_specs,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "header_font_size": header_font_size,
        "header_h": header_h,
        "header_pad_top": header_pad_top,
        "header_pad_bottom": header_pad_bottom,
        "title_h": title_h,
        "width": width,
        "height": height,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "gap": gap,
        "row_label_w": row_label_w,
    }


def build_grid(
    case_dirs: Sequence[Path],
    case_keys: Sequence[str],
    methods: Sequence[tuple[str, str]],
    *,
    cell_w: int = 360,
    cell_h: int = 480,
    gap: int = 8,
    header_h: int = 56,
    row_label_w: int = 0,
    highlight_ours: bool = True,
    title: str | None = None,
    fit_mode: str = "cover",
    focus_on_mask: bool = True,
) -> Image.Image:
    layout = _layout_metrics(
        methods,
        len(case_dirs),
        cell_w=cell_w,
        cell_h=cell_h,
        gap=gap,
        row_label_w=row_label_w,
        title=title,
    )
    col_specs = layout["col_specs"]
    header_font_size = layout["header_font_size"]
    header_h = layout["header_h"]
    header_pad_bottom = layout["header_pad_bottom"]
    title_h = layout["title_h"]
    width = layout["width"]
    height = layout["height"]

    font_h_regular = _load_font(header_font_size, bold=False)
    font_h_bold = _load_font(header_font_size, bold=True)
    font_title = _load_font(32, bold=True)
    font_na = _load_font(32, bold=True)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw_centered_text(draw, (0, 10, width, title_h - 4), title, font_title)

    # column headers: bottom-aligned; only Ours is bold
    y_header = title_h
    for ci, (key, label) in enumerate(col_specs):
        x0 = row_label_w + gap + ci * (cell_w + gap)
        font = font_h_bold if key == "ours" else font_h_regular
        draw_bottom_centered_text(
            draw,
            (x0, y_header, x0 + cell_w, y_header + header_h),
            label,
            font,
            pad_bottom=header_pad_bottom,
        )

    for ri, (case_dir, case_key) in enumerate(zip(case_dirs, case_keys)):
        panels = load_case_panels(case_dir, methods, focus_on_mask=focus_on_mask)
        y0 = title_h + header_h + gap + ri * (cell_h + gap)

        for ci, (key, _label) in enumerate(col_specs):
            x0 = row_label_w + gap + ci * (cell_w + gap)
            img = panels.get(key)
            # Reference cutouts stay letterboxed so small objects aren't over-cropped.
            panel_mode = "contain" if key == "reference" else fit_mode
            if img is None:
                panel = Image.new("RGB", (cell_w, cell_h), (235, 235, 235))
                pd = ImageDraw.Draw(panel)
                draw_centered_text(pd, (0, 0, cell_w, cell_h), "N/A", font_na, fill=(120, 120, 120))
            else:
                panel = fit_panel(img, (cell_w, cell_h), mode=panel_mode)
            canvas.paste(panel, (x0, y0))
            draw.rectangle(
                (x0, y0, x0 + cell_w - 1, y0 + cell_h - 1),
                outline=(210, 210, 210),
                width=1,
            )
            if highlight_ours and key == "ours":
                draw.rectangle(
                    (x0 - 3, y0 - 3, x0 + cell_w + 2, y0 + cell_h + 2),
                    outline=(220, 40, 40),
                    width=4,
                )

    return canvas


def build_grid_pdf(
    case_dirs: Sequence[Path],
    case_keys: Sequence[str],
    methods: Sequence[tuple[str, str]],
    out_pdf: Path,
    *,
    cell_w: int = 360,
    cell_h: int = 480,
    gap: int = 8,
    row_label_w: int = 0,
    highlight_ours: bool = False,
    title: str | None = None,
    fit_mode: str = "cover",
    focus_on_mask: bool = True,
    dpi: int = 100,
) -> Path:
    """Write a hybrid PDF: embedded raster panels + vector selectable text headers.

    Uses matplotlib PDF backend with pdf.fonttype=42 (TrueType) so labels stay
    sharp and can be selected/copied in PDF viewers / LaTeX includegraphics.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    # TrueType embedding → selectable text (not Type 3 bitmaps).
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    layout = _layout_metrics(
        methods,
        len(case_dirs),
        cell_w=cell_w,
        cell_h=cell_h,
        gap=gap,
        row_label_w=row_label_w,
        title=title,
    )
    col_specs = layout["col_specs"]
    header_font_size_px = layout["header_font_size"]
    header_h = layout["header_h"]
    header_pad_bottom = layout["header_pad_bottom"]
    title_h = layout["title_h"]
    width = layout["width"]
    height = layout["height"]

    # Matplotlib font size is in *points* (1/72"), while our layout is in pixels.
    # Convert so rendered glyph height matches the PIL/PNG header sizing.
    #   px = pt * dpi / 72  =>  pt = px * 72 / dpi
    def _px_to_pt(px: float) -> float:
        return float(px) * 72.0 / float(dpi)

    header_font_pt = _px_to_pt(header_font_size_px)
    title_font_pt = _px_to_pt(32)
    na_font_pt = _px_to_pt(32)

    font_regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fp_regular = font_manager.FontProperties(fname=font_regular_path, size=header_font_pt)
    fp_bold = font_manager.FontProperties(fname=font_bold_path, size=header_font_pt)
    fp_title = font_manager.FontProperties(fname=font_bold_path, size=title_font_pt)
    fp_na = font_manager.FontProperties(fname=font_bold_path, size=na_font_pt)

    # Extra figure padding (inches) so top ascenders / PDF crop never clip labels.
    pad_in = 0.08
    fig_w_in = width / float(dpi) + 2 * pad_in
    fig_h_in = height / float(dpi) + 2 * pad_in
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor("white")
    # Axes inset by pad so text near top/bottom stays inside the page.
    ax_l = pad_in / fig_w_in
    ax_b = pad_in / fig_h_in
    ax_w = (width / float(dpi)) / fig_w_in
    ax_h = (height / float(dpi)) / fig_h_in
    ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # top-left origin, matches PIL
    ax.set_axis_off()
    ax.set_facecolor("white")
    # Prevent any residual clipping of vector text at axes edges.
    ax.set_clip_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.text(
            width / 2,
            title_h / 2,
            title,
            ha="center",
            va="center",
            fontproperties=fp_title,
            color=(0.08, 0.08, 0.08),
            clip_on=False,
        )

    # Vector headers: bottom-aligned in the header band.
    # With inverted y, va='baseline' + small pad keeps glyphs inside header_h.
    y_header_bottom = title_h + header_h - header_pad_bottom
    for ci, (key, label) in enumerate(col_specs):
        x0 = row_label_w + gap + ci * (cell_w + gap)
        cx = x0 + cell_w / 2
        fp = fp_bold if key == "ours" else fp_regular
        ax.text(
            cx,
            y_header_bottom,
            label,
            ha="center",
            va="baseline",
            fontproperties=fp,
            color=(0.08, 0.08, 0.08),
            clip_on=False,
            zorder=10,
        )

    for ri, case_dir in enumerate(case_dirs):
        panels = load_case_panels(case_dir, methods, focus_on_mask=focus_on_mask)
        y0 = title_h + header_h + gap + ri * (cell_h + gap)
        for ci, (key, _label) in enumerate(col_specs):
            x0 = row_label_w + gap + ci * (cell_w + gap)
            img = panels.get(key)
            panel_mode = "contain" if key == "reference" else fit_mode
            if img is None:
                panel = Image.new("RGB", (cell_w, cell_h), (235, 235, 235))
            else:
                panel = fit_panel(img, (cell_w, cell_h), mode=panel_mode)

            # extent: [left, right, bottom, top] in data coords; y grows downward.
            ax.imshow(
                panel,
                extent=(x0, x0 + cell_w, y0 + cell_h, y0),
                interpolation="bilinear",
                aspect="auto",
                zorder=1,
            )
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    cell_w,
                    cell_h,
                    fill=False,
                    edgecolor=(210 / 255, 210 / 255, 210 / 255),
                    linewidth=0.6,
                    zorder=2,
                )
            )
            if img is None:
                ax.text(
                    x0 + cell_w / 2,
                    y0 + cell_h / 2,
                    "N/A",
                    ha="center",
                    va="center",
                    fontproperties=fp_na,
                    color=(0.47, 0.47, 0.47),
                    zorder=3,
                )
            if highlight_ours and key == "ours":
                ax.add_patch(
                    Rectangle(
                        (x0 - 3, y0 - 3),
                        cell_w + 6,
                        cell_h + 6,
                        fill=False,
                        edgecolor=(220 / 255, 40 / 255, 40 / 255),
                        linewidth=2.0,
                        zorder=4,
                    )
                )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Keep exact layout size (no tight crop) so panels align with raster PNG.
    fig.savefig(
        out_pdf,
        format="pdf",
        dpi=dpi,
        facecolor="white",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    return out_pdf


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT,
        help="review_keep_export root",
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="suite/case_id relative to export-root",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--cell",
        type=int,
        default=None,
        help="Square cell size (legacy). Prefer --cell-w/--cell-h.",
    )
    p.add_argument("--cell-w", type=int, default=400, help="Panel width in pixels")
    p.add_argument("--cell-h", type=int, default=520, help="Panel height in pixels (taller = better for portrait)")
    p.add_argument("--gap", type=int, default=8)
    p.add_argument(
        "--fit",
        choices=("cover", "contain"),
        default="cover",
        help="cover fills cell (larger subjects); contain letterboxes",
    )
    p.add_argument(
        "--no-focus-crop",
        action="store_true",
        help="Do not zoom to tar_mask region before fitting",
    )
    p.add_argument(
        "--highlight-ours",
        action="store_true",
        help="Draw a red border around the Ours column (off by default for paper figures)",
    )
    p.add_argument("--title", type=str, default="")
    p.add_argument(
        "--methods",
        nargs="+",
        default=[m[0] for m in METHOD_SPECS],
        help="Method keys under results/",
    )
    p.add_argument(
        "--also-compact",
        action="store_true",
        help="Also emit a compact figure without Qwen-Image-2.0 / A2Edit-MoE",
    )
    p.add_argument(
        "--pdf",
        action="store_true",
        default=True,
        help="Also write vector-text PDF (default: on)",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF export",
    )
    p.add_argument(
        "--pdf-dpi",
        type=int,
        default=100,
        help="PDF figure dpi (affects page size = pixel_layout/dpi inches)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_root = args.export_root.resolve()
    case_dirs: list[Path] = []
    case_keys: list[str] = []
    for key in args.cases:
        key = key.strip().strip("/")
        d = export_root / key
        if not d.is_dir():
            raise FileNotFoundError(f"Case dir not found: {d}")
        case_dirs.append(d)
        case_keys.append(key)

    label_map = {k: v for k, v in METHOD_SPECS}
    methods = [(k, label_map.get(k, k)) for k in args.methods]

    if args.cell is not None:
        cell_w = cell_h = int(args.cell)
    else:
        cell_w, cell_h = int(args.cell_w), int(args.cell_h)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or None
    focus_on_mask = not args.no_focus_crop
    common_kw = dict(
        cell_w=cell_w,
        cell_h=cell_h,
        gap=args.gap,
        highlight_ours=bool(args.highlight_ours),
        title=title,
        fit_mode=args.fit,
        focus_on_mask=focus_on_mask,
    )

    grid = build_grid(case_dirs, case_keys, methods, **common_kw)
    full_png = args.out_dir / "qualitative_full.png"
    # optimize=False is much faster on large canvases
    grid.save(full_png, format="PNG", compress_level=3)
    full_jpg = args.out_dir / "qualitative_full.jpg"
    grid.convert("RGB").save(full_jpg, format="JPEG", quality=95, subsampling=0)

    write_pdf = bool(args.pdf) and not bool(args.no_pdf)
    full_pdf = args.out_dir / "qualitative_full.pdf"
    compact_pdf = args.out_dir / "qualitative_compact.pdf"
    outputs = {
        "full_png": str(full_png),
        "full_jpg": str(full_jpg),
    }

    if write_pdf:
        build_grid_pdf(
            case_dirs,
            case_keys,
            methods,
            full_pdf,
            **common_kw,
            dpi=int(args.pdf_dpi),
        )
        outputs["full_pdf"] = str(full_pdf)
        print(f"Wrote {full_pdf}  (vector text, fonttype=42)")

    if args.also_compact:
        compact_methods = [
            ("anydoor", "AnyDoor"),
            ("insert_anything", "Insert Anything"),
            ("qwen_edit_2511", "Qwen-Edit"),
            ("flux_kontext", "FLUX Kontext"),
            ("ours", "Ours"),
        ]
        compact = build_grid(case_dirs, case_keys, compact_methods, **common_kw)
        compact_png = args.out_dir / "qualitative_compact.png"
        compact.save(compact_png, format="PNG", compress_level=3)
        compact.convert("RGB").save(
            args.out_dir / "qualitative_compact.jpg",
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        outputs["compact_png"] = str(compact_png)
        if write_pdf:
            build_grid_pdf(
                case_dirs,
                case_keys,
                compact_methods,
                compact_pdf,
                **common_kw,
                dpi=int(args.pdf_dpi),
            )
            outputs["compact_pdf"] = str(compact_pdf)
            print(f"Wrote {compact_pdf}  (vector text, fonttype=42)")

    meta = {
        "cases": case_keys,
        "methods": [m[0] for m in methods],
        "cell_w": cell_w,
        "cell_h": cell_h,
        "fit": args.fit,
        "focus_on_mask": focus_on_mask,
        "outputs": outputs,
        "grid_size": list(grid.size),
        "pdf_vector_text": write_pdf,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {full_png}  size={grid.size[0]}x{grid.size[1]}  cell={cell_w}x{cell_h} fit={args.fit}")
    print(f"Wrote {full_jpg}")
    if args.also_compact:
        print(f"Wrote {args.out_dir / 'qualitative_compact.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
