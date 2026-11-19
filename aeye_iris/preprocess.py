"""Image preprocessing variants benchmarked in the combination grid."""
from __future__ import annotations

import cv2
import numpy as np

CHANNELS = ("gray", "red", "v")


def to_channel(bgr: np.ndarray, channel: str = "red") -> np.ndarray:
    """Single-channel view used by the classical segmenters.

    red: in visible light the red channel gives the best iris/sclera contrast and the
    most uniform iris for dark eyes (melanin absorbs less red); gray is the default
    luminance; v is HSV value (max of channels).
    """
    if bgr.ndim == 2:
        return bgr
    if channel == "gray":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if channel == "red":
        return bgr[:, :, 2].copy()
    if channel == "v":
        return bgr.max(axis=2)
    raise ValueError(channel)


def clahe(gray: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def specular_mask(bgr: np.ndarray, rel_thresh: float = 0.92, dilate: int = 2) -> np.ndarray:
    """Corneal reflections: very bright, low-saturation, locally peaked pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    k = max(3, int(round(min(bgr.shape[:2]) / 25)) | 1)
    tophat = cv2.morphologyEx(v.astype(np.uint8), cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # reflections are LOCAL peaks (the sclera is bright and unsaturated too, but not
    # peaked at this scale) or fully saturated pixels
    thr = max(200.0, rel_thresh * float(np.percentile(v, 99.9)))
    m = ((v >= 250) & (s < 60)) | ((tophat > 50) & (v >= min(thr, 230)) & (s < 110)) | ((tophat > 80) & (v > 160))
    m = m.astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2))
    return m


def inpaint_specular(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = specular_mask(bgr)
    if mask.sum() == 0:
        return bgr
    return cv2.inpaint(bgr, mask * 255, 3, cv2.INPAINT_TELEA)


def prepare(bgr: np.ndarray, channel: str = "red", use_clahe: bool = False,
            inpaint: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (single-channel image, cleaned BGR, specular mask)."""
    spec = specular_mask(bgr)
    clean = inpaint_specular(bgr, spec) if inpaint else bgr
    g = to_channel(clean, channel)
    if use_clahe:
        g = clahe(g)
    return g, clean, spec


def resize_max(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return bgr, 1.0
    return cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA), s
