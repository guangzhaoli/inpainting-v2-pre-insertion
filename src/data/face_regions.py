"""Optional CPU face regions for local identity supervision, without downloads."""

from functools import lru_cache

import cv2
import numpy as np
import torch


@lru_cache(maxsize=1)
def _detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV's packaged frontal face detector is unavailable")
    return detector


def face_region_mask(image, edit_mask):
    """Select the largest detected face overlapping the editable foreground.

    Zero means no confident face. Never substitute a whole-body identity crop
    for a failed face detection. Hair-only samples usually yield a zero mask.
    """
    rgb = (image.detach().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))))
    faces = _detector().detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    best = None
    for x, y, w, h in faces:
        x1, y1 = max(0, int(x / scale)), max(0, int(y / scale))
        x2, y2 = min(width, int((x+w) / scale)), min(height, int((y+h) / scale))
        overlap = edit_mask[y1:y2, x1:x2]
        if overlap.numel() and float((overlap > .5).float().mean()) >= .8:
            if best is None or (x2-x1) * (y2-y1) > best[0]:
                best = ((x2-x1) * (y2-y1), x1, y1, x2, y2)
    result = torch.zeros_like(edit_mask)
    if best is not None:
        _, x1, y1, x2, y2 = best
        result[y1:y2, x1:x2] = 1
    return result
