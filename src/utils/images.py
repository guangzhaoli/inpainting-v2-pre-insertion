import torch.nn.functional as F
import torch
from typing import Tuple, Union
import PIL
from PIL import Image
import numpy as np
import cv2
import math


def resize_bhw(x: torch.Tensor, h: int, w: int, mode="bilinear"):
    x = x.unsqueeze(1)
    # [B, 1, H, W]
    x = F.interpolate(
        x, size=(h, w), mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
        antialias=True if mode in {"bilinear", "bicubic"} and (h < x.shape[-2] or w < x.shape[-1]) else False,
    )
    return x.squeeze(1)


def resize_mask_bhw(x: torch.Tensor, h: int, w: int):
    return resize_bhw(x, h, w, mode="nearest")


def make_image_shape_devisible(width, height, vae_scale_factor: int) -> Tuple[int, int]:
    """make width and height devisible by vae_scale_factor * 2"""
    multiple_of = vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    return width, height


def make_image_devisible(image: Union[torch.Tensor, PIL.Image, np.ndarray], vae_scale_factor: int) -> torch.Tensor:
    """make image devisible by vae_scale_factor * 2, suppose image is B,C,H,W"""

    if isinstance(image, torch.Tensor):
        height, width = image.shape[2:]
        width, height = make_image_shape_devisible(width, height, vae_scale_factor)
        image = F.interpolate(image, size=(height, width), mode='bilinear')
    elif isinstance(image, PIL.Image.Image):
        width, height = image.size
        width, height = make_image_shape_devisible(width, height, vae_scale_factor)
        image = image.resize((width, height))
    elif isinstance(image, np.ndarray):  # H,W,C
        height, width = image.shape[:2]
        width, height = make_image_shape_devisible(width, height, vae_scale_factor)
        image = cv2.resize(image, (width, height))
    return image


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height, None


def calculate_best_resolution(width: int, height: int, best_resolution: int = 1024 * 1024):
    calculated_width, calculated_height, _ = calculate_dimensions(best_resolution, width / height)
    return calculated_width, calculated_height


def image_adjust_best_resolution(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray]):
    """Preprocess images for caching, suppose image is B,C,H,W
    tensor: B,C,H,W
    PIL.Image.Image: H,W,C
    np.ndarray: H,W,C
    """
    if isinstance(image, torch.Tensor):
        height, width = image.shape[2:]
    elif isinstance(image, PIL.Image.Image):
        width, height = image.size
    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    calculated_width, calculated_height = calculate_best_resolution(width, height)
    if isinstance(image, torch.Tensor):
        new_image = F.interpolate(image, size=(calculated_height, calculated_width), mode="bilinear")
    elif isinstance(image, PIL.Image.Image):
        new_image = image.resize((calculated_width, calculated_height), PIL.Image.Resampling.BICUBIC)
    elif isinstance(image, np.ndarray):
        new_image = cv2.resize(image, (calculated_width, calculated_height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return new_image


def crop_to_mask_with_padding(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding_ratio: float = 0.1,
    min_size: int = 64,
    align_to: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Crop image and mask to the bounding box of mask region with padding.
    Maintains aspect ratio and ensures dimensions are aligned.

    Args:
        image: [B, C, H, W] or [C, H, W] tensor in range [0, 1]
        mask: [B, H, W] or [H, W] tensor in range [0, 1]
        padding_ratio: Ratio of padding relative to bbox size (default: 0.1 = 10%)
        min_size: Minimum dimension for cropped region
        align_to: Align dimensions to multiples of this value (for VAE compatibility)

    Returns:
        cropped_image: [B, C, H', W'] cropped image
        cropped_mask: [B, H', W'] cropped mask
        crop_info: dict with crop parameters for potential restoration
            - bbox: (y1, x1, y2, x2) original bounding box
            - padded_bbox: (y1, x1, y2, x2) after padding
            - original_shape: (H, W)
    """
    # Handle batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, C, H, W = image.shape

    # Find bounding box for each sample in batch
    cropped_images = []
    cropped_masks = []
    crop_infos = []

    for b in range(B):
        mask_b = mask[b]  # [H, W]

        # Find non-zero regions
        nonzero_indices = torch.nonzero(mask_b > 0.5, as_tuple=False)

        if len(nonzero_indices) == 0:
            # No mask region, return original
            cropped_images.append(image[b])
            cropped_masks.append(mask_b)
            crop_infos.append({
                'bbox': (0, 0, H, W),
                'padded_bbox': (0, 0, H, W),
                'original_shape': (H, W)
            })
            continue

        # Get bounding box
        y_coords = nonzero_indices[:, 0]
        x_coords = nonzero_indices[:, 1]
        y1, y2 = y_coords.min().item(), y_coords.max().item()
        x1, x2 = x_coords.min().item(), x_coords.max().item()

        # Calculate padding
        bbox_h = y2 - y1 + 1
        bbox_w = x2 - x1 + 1
        pad_h = int(bbox_h * padding_ratio)
        pad_w = int(bbox_w * padding_ratio)

        # Apply padding with boundary checks
        y1_padded = max(0, y1 - pad_h)
        x1_padded = max(0, x1 - pad_w)
        y2_padded = min(H, y2 + pad_h + 1)
        x2_padded = min(W, x2 + pad_w + 1)

        # Ensure minimum size
        crop_h = y2_padded - y1_padded
        crop_w = x2_padded - x1_padded

        if crop_h < min_size:
            deficit = min_size - crop_h
            y1_padded = max(0, y1_padded - deficit // 2)
            y2_padded = min(H, y2_padded + (deficit - deficit // 2))

        if crop_w < min_size:
            deficit = min_size - crop_w
            x1_padded = max(0, x1_padded - deficit // 2)
            x2_padded = min(W, x2_padded + (deficit - deficit // 2))

        # Align to multiples (for VAE)
        crop_h = y2_padded - y1_padded
        crop_w = x2_padded - x1_padded

        aligned_h = (crop_h // align_to) * align_to
        aligned_w = (crop_w // align_to) * align_to

        # Ensure at least align_to size
        if aligned_h == 0:
            aligned_h = align_to
        if aligned_w == 0:
            aligned_w = align_to

        # Adjust crop region to meet aligned dimensions
        h_adjust = crop_h - aligned_h
        w_adjust = crop_w - aligned_w

        y1_aligned = y1_padded + h_adjust // 2
        y2_aligned = y1_aligned + aligned_h
        x1_aligned = x1_padded + w_adjust // 2
        x2_aligned = x1_aligned + aligned_w

        # Final boundary check
        if y2_aligned > H:
            y1_aligned = H - aligned_h
            y2_aligned = H
        if x2_aligned > W:
            x1_aligned = W - aligned_w
            x2_aligned = W

        y1_aligned = max(0, y1_aligned)
        x1_aligned = max(0, x1_aligned)

        # Crop
        cropped_img = image[b, :, y1_aligned:y2_aligned, x1_aligned:x2_aligned]
        cropped_msk = mask_b[y1_aligned:y2_aligned, x1_aligned:x2_aligned]

        cropped_images.append(cropped_img)
        cropped_masks.append(cropped_msk)
        crop_infos.append({
            'bbox': (y1, x1, y2 + 1, x2 + 1),
            'padded_bbox': (y1_aligned, x1_aligned, y2_aligned, x2_aligned),
            'original_shape': (H, W)
        })

    # Handle variable sizes in batch by padding to max dimensions
    if len(cropped_images) > 1:
        # Find maximum dimensions across batch
        max_h = max(img.shape[1] for img in cropped_images)
        max_w = max(img.shape[2] for img in cropped_images)

        # Align max dimensions to multiples (for VAE compatibility)
        max_h = ((max_h + align_to - 1) // align_to) * align_to
        max_w = ((max_w + align_to - 1) // align_to) * align_to

        # Pad each image/mask to max dimensions
        padded_images = []
        padded_masks = []

        for i in range(len(cropped_images)):
            img = cropped_images[i]  # [C, H, W]
            msk = cropped_masks[i]   # [H, W]

            h_diff = max_h - img.shape[1]
            w_diff = max_w - img.shape[2]

            if h_diff > 0 or w_diff > 0:
                # Calculate padding (center-align)
                pad_top = h_diff // 2
                pad_bottom = h_diff - pad_top
                pad_left = w_diff // 2
                pad_right = w_diff - pad_left

                # Pad image with white (1.0) to match masked background
                # Pad format: (left, right, top, bottom)
                img_padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom),
                                  mode='constant', value=1.0)
                # Pad mask with 0 (background)
                msk_padded = F.pad(msk, (pad_left, pad_right, pad_top, pad_bottom),
                                  mode='constant', value=0.0)

                padded_images.append(img_padded)
                padded_masks.append(msk_padded)

                # Record padding info for debugging
                crop_infos[i]['batch_padding'] = (pad_top, pad_bottom, pad_left, pad_right)
                crop_infos[i]['final_shape'] = (max_h, max_w)
            else:
                padded_images.append(img)
                padded_masks.append(msk)
                crop_infos[i]['batch_padding'] = None
                crop_infos[i]['final_shape'] = (img.shape[1], img.shape[2])

        cropped_images = padded_images
        cropped_masks = padded_masks

    # Stack batch (now all have same size)
    cropped_images = torch.stack(cropped_images, dim=0)
    cropped_masks = torch.stack(cropped_masks, dim=0)

    if squeeze_output:
        cropped_images = cropped_images.squeeze(0)
        cropped_masks = cropped_masks.squeeze(0)
        crop_infos = crop_infos[0]

    return cropped_images, cropped_masks, crop_infos


def extract_foreground_from_image_and_mask(
    image: Union[Image.Image, np.ndarray],
    mask: Union[Image.Image, np.ndarray],
    padding_ratio: float = 0.15,
    min_size: int = 64,
    align_to: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align mask to image, crop the masked foreground region, and fill background with white."""
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.array(image)
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[:, :, None], 3, axis=2)
        elif image_np.ndim == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

    if isinstance(mask, Image.Image):
        mask_np = np.array(mask.convert("L"))
    else:
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

    if mask_np.shape[:2] != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_np = (mask_np > 127).astype(np.uint8) * 255
    if mask_np.max() == 0:
        return image_np, mask_np

    height, width = mask_np.shape[:2]
    y_coords, x_coords = np.where(mask_np > 127)
    y1, y2 = int(y_coords.min()), int(y_coords.max()) + 1
    x1, x2 = int(x_coords.min()), int(x_coords.max()) + 1

    bbox_h = y2 - y1
    bbox_w = x2 - x1
    pad_h = int(bbox_h * padding_ratio)
    pad_w = int(bbox_w * padding_ratio)

    y1_padded = max(0, y1 - pad_h)
    x1_padded = max(0, x1 - pad_w)
    y2_padded = min(height, y2 + pad_h)
    x2_padded = min(width, x2 + pad_w)

    crop_h = y2_padded - y1_padded
    crop_w = x2_padded - x1_padded
    if crop_h < min_size:
        deficit = min_size - crop_h
        y1_padded = max(0, y1_padded - deficit // 2)
        y2_padded = min(height, y2_padded + (deficit - deficit // 2))
    if crop_w < min_size:
        deficit = min_size - crop_w
        x1_padded = max(0, x1_padded - deficit // 2)
        x2_padded = min(width, x2_padded + (deficit - deficit // 2))

    crop_h = y2_padded - y1_padded
    crop_w = x2_padded - x1_padded
    aligned_h = max(align_to, (crop_h // align_to) * align_to)
    aligned_w = max(align_to, (crop_w // align_to) * align_to)

    h_adjust = crop_h - aligned_h
    w_adjust = crop_w - aligned_w
    y1_aligned = y1_padded + h_adjust // 2
    x1_aligned = x1_padded + w_adjust // 2
    y2_aligned = y1_aligned + aligned_h
    x2_aligned = x1_aligned + aligned_w

    if y2_aligned > height:
        y1_aligned = height - aligned_h
        y2_aligned = height
    if x2_aligned > width:
        x1_aligned = width - aligned_w
        x2_aligned = width

    y1_aligned = max(0, y1_aligned)
    x1_aligned = max(0, x1_aligned)

    cropped_image_np = image_np[y1_aligned:y2_aligned, x1_aligned:x2_aligned]
    cropped_mask_np = mask_np[y1_aligned:y2_aligned, x1_aligned:x2_aligned]
    masked_image_np = np.where(cropped_mask_np[:, :, None] > 127, cropped_image_np, 255).astype(np.uint8)
    return np.ascontiguousarray(masked_image_np), np.ascontiguousarray(cropped_mask_np)
