"""Lighting and camera augmentation for reference-based insertion.

Teacher pairs share one exposure (adjacent video frames or same-scene edits).
Between reference and target photos the light colour differs by 0.2–1.0°
(median) for pets, people and objects, against 4–6° on the eval set, and the
teachers copy 0.8–0.97 of a reference lighting change. Each sample takes one
branch:

* ``reference``: re-light the reference object (white balance along the
  Planckian locus, exposure, tone, optional low-frequency shading). The white
  background stays white, so it carries no cue about the cast. Teaches the
  model to ignore the reference's light, at some colour-identity cost.
* ``scene``: one global grade on the whole target, GT included. The reference
  is untouched and the GT stays a self-consistent photo. Teaches the model to
  follow the scene, at no identity cost.
* ``none``: the original pair, which keeps colour fidelity.

Edge spill, camera degradation and reference geometry are drawn independently.
Images are float32 HWC sRGB in [0, 1]; grading happens in linear RGB.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
REFERENCE_CCT = 6504.0
_MIN_MIRED = 1e6 / 25000.0
_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1 / 2.4) - 0.055)


def planckian_rgb(cct: float) -> np.ndarray:
    """Linear sRGB colour of a black body, G = 1 (Kim et al. 2002 cubic fit, 1667–25000 K)."""
    t = float(np.clip(cct, 1667.0, 25000.0))
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    rgb = _XYZ_TO_LINEAR_SRGB @ np.array([x / y, 1.0, (1.0 - x - y) / y])
    return rgb / rgb[1]


def white_balance_gains(delta_mired: float, tint: float = 0.0) -> np.ndarray:
    """Von Kries gains that move the light from 6504 K by ``delta_mired``.

    Positive values lower the colour temperature (warmer light); ±100 mired is
    about a 21° change of light colour. ``tint`` scales green (+) against
    magenta (−). Gains have unit luminance, so exposure stays a separate knob.
    """
    mired = max(1e6 / REFERENCE_CCT + float(delta_mired), _MIN_MIRED)
    gains = planckian_rgb(1e6 / mired) / planckian_rgb(REFERENCE_CCT)
    gains = gains * np.array([1.0, 1.0 + float(tint), 1.0])
    return (gains / float(LUMA @ gains)).astype(np.float32)


def soft_shoulder(linear: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Compress values above ``knee`` into [knee, 1) so brightened highlights keep detail."""
    over = linear > knee
    out = linear.copy()
    out[over] = knee + (1.0 - knee) * np.tanh((linear[over] - knee) / (1.0 - knee))
    return out


@dataclass
class Grade:
    """One global grade: light colour and exposure in linear RGB, then a tone tweak."""

    delta_mired: float
    tint: float
    exposure_ev: float
    gamma: float
    saturation: float

    @property
    def gains(self) -> np.ndarray:
        return white_balance_gains(self.delta_mired, self.tint) * np.float32(2.0**self.exposure_ev)


def sample_grade(cfg, rng: np.random.Generator) -> Grade:
    """Laplace light-colour shifts and Gaussian exposure, both truncated (``LightingRangeConfig``)."""
    delta = rng.laplace(0.0, cfg.wb_mired_scale) if cfg.wb_mired_scale > 0 else 0.0
    exposure = rng.normal(0.0, cfg.exposure_ev_std) if cfg.exposure_ev_std > 0 else 0.0
    log_gamma = math.log1p(cfg.gamma_max)
    return Grade(
        delta_mired=float(np.clip(delta, -cfg.wb_mired_max, cfg.wb_mired_max)),
        tint=float(rng.uniform(-cfg.tint_max, cfg.tint_max)),
        exposure_ev=float(np.clip(exposure, -cfg.exposure_ev_max, cfg.exposure_ev_max)),
        gamma=float(math.exp(rng.uniform(-log_gamma, log_gamma))),
        saturation=float(rng.uniform(1.0 - cfg.saturation_max, 1.0 + cfg.saturation_max)),
    )


def apply_grade(image: np.ndarray, grade: Grade, shading: Optional[np.ndarray] = None) -> np.ndarray:
    """Grade an sRGB image; ``shading`` is an optional HxW gain field multiplied in linear light."""
    gains = grade.gains
    linear = srgb_to_linear(image) * gains
    peak = float(gains.max())
    if shading is not None:
        linear = linear * shading[..., None]
        peak *= float(shading.max())
    if peak > 1.0:
        linear = soft_shoulder(linear)
    srgb = linear_to_srgb(linear) ** grade.gamma
    luma = (srgb @ LUMA)[..., None]
    return np.clip(luma + grade.saturation * (srgb - luma), 0.0, 1.0).astype(np.float32)


def _object_box(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Centre and half extents (cx, cy, hx, hy) of the mask's bounding box."""
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) == 0:
        return None
    return (
        (xs.min() + xs.max()) / 2.0,
        (ys.min() + ys.max()) / 2.0,
        max((xs.max() - xs.min()) / 2.0, 1.0),
        max((ys.max() - ys.min()) / 2.0, 1.0),
    )


def low_frequency_shading(mask: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """exp(log-gain field) over the object's box: a ramp from a random side plus 1–2 broad blobs.

    The ramp reaches ±``strength`` at the box edge along its direction, so the
    reference looks lit from another side than the one the GT shows.
    """
    height, width = mask.shape
    box = _object_box(mask)
    if box is None:
        return np.ones((height, width), dtype=np.float32)
    cx, cy, hx, hy = box
    v, u = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = (u - cx) / hx, (v - cy) / hy
    angle = rng.uniform(0.0, 2.0 * math.pi)
    field = rng.uniform(-strength, strength) * (math.cos(angle) * u + math.sin(angle) * v)
    for _ in range(int(rng.integers(1, 3))):
        mx, my = rng.uniform(-0.6, 0.6, size=2)
        spread = rng.uniform(0.35, 0.7)
        field = field + rng.uniform(-strength, strength) * np.exp(-((u - mx) ** 2 + (v - my) ** 2) / (2 * spread**2))
    return np.exp(field).astype(np.float32)


def _long_side(binary: np.ndarray) -> int:
    ys, xs = np.nonzero(binary)
    return int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)


def _depth(binary: np.ndarray) -> np.ndarray:
    """Euclidean distance from each mask pixel to the nearest background pixel; 0 outside."""
    return cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _within(binary: np.ndarray, radius: float) -> np.ndarray:
    """Pixels within ``radius`` of ``binary``: dilation by a disk, in time independent of the radius."""
    if not binary.any():
        return binary.copy()
    return _depth(~binary) <= radius


def _thick(binary: np.ndarray, depth: np.ndarray, radius: float) -> np.ndarray:
    """Opening by a disk: mask pixels that an inscribed disk of ``radius`` covers.

    The extra pixel absorbs grid effects, so smooth outlines are fully covered.
    """
    return binary & _within(depth > radius, radius + 1.0)


def jitter_mask_edge(
    mask: np.ndarray, rng: np.random.Generator, radius_frac: Tuple[float, float] = (0.002, 0.008)
) -> np.ndarray:
    """Erode or dilate a 0/255 reference mask by a radius relative to the object's long side.

    Dilation pulls a ring of the reference photo's own background in, like a
    loose user mask; erosion trims the outline, like a tight one. Parts
    narrower than about eight radii (chains, straps, strands) and where they
    attach are not eroded: a tight mask still covers them, and the GT keeps them.
    """
    binary = mask > 127
    if not binary.any():
        return mask
    radius = max(1, int(round(rng.uniform(*radius_frac) * _long_side(binary))))
    if rng.random() < 0.5:
        depth = _depth(binary)
        thin = binary & ~_thick(binary, depth, 4 * radius)
        out = (depth > radius) | (binary & _within(thin, radius))
    else:
        out = _within(binary, radius)
    return mask if not out.any() else (out * 255).astype(np.uint8)


def add_edge_spill(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    width_frac: Tuple[float, float] = (0.004, 0.012),
    strength: Tuple[float, float] = (0.1, 0.4),
) -> np.ndarray:
    """Blend the object's rim toward a random colour, like background spill in hair or fur.

    Parts narrower than about four band widths keep their colour: the band
    would cover a chain or strap entirely and recolour it.
    """
    binary = mask > 0.5
    if not binary.any():
        return image
    width = max(1.0, rng.uniform(*width_frac) * _long_side(binary))
    depth = _depth(binary)
    weight = np.clip(1.0 - (depth - 1.0) / width, 0.0, 1.0) * _thick(binary, depth, 2 * width) * rng.uniform(*strength)
    color = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
    weight = weight[..., None].astype(np.float32)
    return (image * (1.0 - weight) + color * weight).astype(np.float32)


def degrade_camera(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resampling, blur, sensor noise and JPEG, each with its own chance: a different camera."""
    height, width = image.shape[:2]
    out = image
    if rng.random() < 0.3:
        scale = rng.uniform(0.45, 0.85)
        small = cv2.resize(out, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.4:
        sigma = rng.uniform(0.004, 0.02)
        out = out + rng.normal(0.0, sigma, (height, width, 1)) + rng.normal(0.0, 0.5 * sigma, (height, width, 3))
    if rng.random() < 0.5:
        quality = int(rng.integers(45, 93))
        bgr = (np.clip(out, 0.0, 1.0)[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
        _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def warp_reference(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    rotation_deg: float,
    scale: float,
    perspective: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Small rotation, scale and perspective about the object centre; the canvas stays white.

    Upscaling stops at 1 + scale/2: the reference crop pads the object by 15%.
    """
    box = _object_box(mask)
    if box is None:
        return image, mask
    cx, cy, hx, hy = box
    height, width = mask.shape
    rotation = np.eye(3)
    rotation[:2] = cv2.getRotationMatrix2D(
        (cx, cy), rng.uniform(-rotation_deg, rotation_deg), rng.uniform(1.0 - scale, 1.0 + 0.5 * scale)
    )
    corners = np.float32([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]])
    jitter = rng.uniform(-perspective, perspective, size=(4, 2)) * np.float32([2 * hx, 2 * hy])
    homography = cv2.getPerspectiveTransform(corners, (corners + jitter).astype(np.float32)) @ rotation
    warped = cv2.warpPerspective(
        image, homography, (width, height), flags=cv2.INTER_LINEAR, borderValue=(1.0, 1.0, 1.0)
    )
    warped_mask = cv2.warpPerspective(
        mask.astype(np.float32), homography, (width, height), flags=cv2.INTER_NEAREST, borderValue=0.0
    )
    if not (warped_mask > 0.5).any():
        return image, mask
    return warped, warped_mask


@dataclass
class LightingPlan:
    """Everything drawn for one sample, so the dataset applies it in two places."""

    branch: str
    grade: Optional[Grade]
    shading: bool
    edge: bool
    camera_reference: bool
    camera_scene: bool
    geometry: bool

    @property
    def changes_reference(self) -> bool:
        return self.branch == "reference" or self.edge or self.camera_reference or self.geometry

    @property
    def changes_scene(self) -> bool:
        return self.branch == "scene" or self.camera_scene

    def metadata(self) -> dict:
        grade = self.grade
        return {
            "branch": self.branch,
            "delta_mired": 0.0 if grade is None else grade.delta_mired,
            "exposure_ev": 0.0 if grade is None else grade.exposure_ev,
            "shading": self.shading,
            "edge": self.edge,
            "camera_reference": self.camera_reference,
            "camera_scene": self.camera_scene,
            "geometry": self.geometry,
        }


def sample_lighting_plan(cfg, rng: np.random.Generator) -> LightingPlan:
    draw = rng.random()
    if draw < cfg.reference_prob:
        branch = "reference"
    elif draw < cfg.reference_prob + cfg.scene_prob:
        branch = "scene"
    else:
        branch = "none"
    grade = None if branch == "none" else sample_grade(cfg.reference if branch == "reference" else cfg.scene, rng)
    return LightingPlan(
        branch=branch,
        grade=grade,
        shading=branch == "reference" and rng.random() < cfg.shading_prob,
        edge=rng.random() < cfg.edge_prob,
        camera_reference=rng.random() < cfg.camera_reference_prob,
        camera_scene=rng.random() < cfg.camera_scene_prob,
        geometry=rng.random() < cfg.geometry_prob,
    )


def augment_reference(
    image: np.ndarray, mask: np.ndarray, plan: LightingPlan, cfg, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp, re-light, degrade and spill the reference object; outside its mask stays white."""
    if plan.geometry:
        image, mask = warp_reference(
            image, mask, rng, cfg.geometry_rotation_deg, cfg.geometry_scale, cfg.geometry_perspective
        )
    out = image
    if plan.branch == "reference":
        shading = low_frequency_shading(mask, cfg.shading_strength, rng) if plan.shading else None
        out = apply_grade(out, plan.grade, shading)
    if plan.camera_reference:
        out = degrade_camera(out, rng)
    if plan.edge:
        out = add_edge_spill(out, mask, rng)
    return np.where((mask > 0.5)[..., None], out, 1.0).astype(np.float32), mask.astype(np.float32)


def augment_scene(image: np.ndarray, plan: LightingPlan, rng: np.random.Generator) -> np.ndarray:
    """One global grade and/or camera degradation over the whole target, GT included."""
    out = image
    if plan.branch == "scene":
        out = apply_grade(out, plan.grade)
    if plan.camera_scene:
        out = degrade_camera(out, rng)
    return out
