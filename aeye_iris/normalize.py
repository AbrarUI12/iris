"""Daugman rubber-sheet normalisation and back-projection.

Polar convention (documented for the zone map):
* row 0 = pupil boundary, last row = limbus (radius is the *fraction* of the
  pupil->limbus distance, so pupil dilation does not move zone boundaries).
* column 0 = 12 o'clock; columns increase CLOCKWISE AS SEEN IN THE PHOTO.
  With ``mirror_left=True`` the left eye runs counter-clockwise so that a column means
  the same nasal/temporal position in both eyes. Default False (photo clock face), which
  is how iris charts are normally read; the zone map stores labels per eye anyway.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import Ellipse
from .types import NormalizedIris, SegResult


def polar_maps(pupil: Ellipse, iris: Ellipse, n_r: int = 64, n_t: int = 512,
               eye_side: str = "right", mirror_left: bool = False):
    theta = np.arange(n_t) / n_t * 2 * np.pi  # 0 at 12 o'clock
    if eye_side == "left" and mirror_left:
        phi = -theta - np.pi / 2
    else:
        phi = theta - np.pi / 2  # image angle (y down): -pi/2 = 12 o'clock, cw positive
    px, py = pupil.point_at(phi)
    ix, iy = iris.point_at(phi)
    rho = (np.arange(n_r) + 0.5) / n_r
    map_x = (1 - rho)[:, None] * px[None, :] + rho[:, None] * ix[None, :]
    map_y = (1 - rho)[:, None] * py[None, :] + rho[:, None] * iy[None, :]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def normalize(bgr: np.ndarray, seg: SegResult, n_r: int = 64, n_t: int = 512,
              eye_side: str = "right", mirror_left: bool = False) -> NormalizedIris:
    assert seg.ok, "segmentation failed"
    mx, my = polar_maps(seg.pupil, seg.iris, n_r, n_t, eye_side, mirror_left)
    img = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    h, w = bgr.shape[:2]
    inside = (mx >= 0) & (my >= 0) & (mx <= w - 1) & (my <= h - 1)
    if seg.iris_mask is not None:
        vm = cv2.remap(seg.iris_mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT)
        valid = inside & (vm > 0)
    else:
        valid = inside
    return NormalizedIris(img, valid, mx, my, eye_side)


def polar_mask_to_image(mask: np.ndarray, norm: NormalizedIris, shape) -> np.ndarray:
    """Back-project a polar mask to image space (for overlays)."""
    out = np.zeros(shape[:2], np.uint8)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return out
    px = np.round(norm.map_x[ys, xs]).astype(int)
    py = np.round(norm.map_y[ys, xs]).astype(int)
    ok = (px >= 0) & (py >= 0) & (px < shape[1]) & (py < shape[0])
    out[py[ok], px[ok]] = 1
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
