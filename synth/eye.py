"""Procedural visible-light eye renderer with exact ground truth.

Used for unit tests and as a host for synthetic marking injection. It is deliberately
simple; it is NOT a substitute for real data and is never used to report segmentation
accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np

from aeye_iris.geometry import Ellipse, annulus_mask

IRIS_BGR = {
    "blue": (175, 135, 95),
    "brown": (40, 65, 105),
    "green_mixed": (85, 130, 110),
}


def iris_texture(nr: int, nt: int, base_bgr, rng: np.random.Generator,
                 fibre: str = "radial", collarette_frac: float = 0.35) -> np.ndarray:
    """Polar texture (nr x nt x 3): radial fibres + collarette ring + darker limbus."""
    base = np.array(base_bgr, np.float32)
    if fibre == "radial":
        # streaks: smooth along r (axis 0), sharp along theta
        n = rng.normal(0, 1, (max(4, nr // 8), nt)).astype(np.float32)
        n = cv2.resize(n, (nt, nr), interpolation=cv2.INTER_CUBIC)
    else:  # transverse: smooth along theta, sharp along r
        n = rng.normal(0, 1, (nr, max(4, nt // 16))).astype(np.float32)
        n = cv2.resize(n, (nt, nr), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), 0.7)
    r = np.linspace(0, 1, nr, dtype=np.float32)[:, None]
    shade = 1.0 + 0.12 * n
    shade *= 1.0 - 0.25 * np.clip((r - 0.85) / 0.15, 0, 1)  # limbal darkening
    shade *= 1.0 + 0.10 * np.exp(-((r - collarette_frac) / 0.04) ** 2)  # collarette
    tex = base[None, None, :] * shade[..., None]
    return np.clip(tex, 0, 255).astype(np.uint8)


def polar_to_image(tex: np.ndarray, shape, pupil: Ellipse, iris: Ellipse) -> tuple[np.ndarray, np.ndarray]:
    """Warp a polar texture into the annulus. Returns (image, mask). theta=0 at 12 o'clock,
    increasing clockwise in the image."""
    h, w = shape
    nr, nt = tex.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phi = np.arctan2(yy - iris.cy, xx - iris.cx)  # image angle, 0 = 3 o'clock, cw positive
    rp = pupil.radius_at(phi) + 0.0
    # pupil center may be offset: approximate radial coordinate between boundaries
    px, py = pupil.point_at(np.arctan2(yy - pupil.cy, xx - pupil.cx))
    d_in = np.hypot(px - iris.cx, py - iris.cy)
    d_out = iris.radius_at(phi)
    d = np.hypot(xx - iris.cx, yy - iris.cy)
    rho = (d - d_in) / np.maximum(d_out - d_in, 1e-3)
    theta = (phi + np.pi / 2) % (2 * np.pi)  # 0 at 12 o'clock, clockwise
    mx = (theta / (2 * np.pi) * nt).astype(np.float32)
    my = (rho * (nr - 1)).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    mask = annulus_mask((h, w), pupil, iris)
    del rp
    return img, mask


def render_eye(h: int = 300, w: int = 400, color: str = "brown", seed: int = 0,
               lid_cover: float = 0.15, glare: bool = True, fibre: str = "radial",
               texture: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.25, 0.33) * min(h, w)
    cx = w / 2 + rng.uniform(-0.08, 0.08) * w
    cy = h / 2 + rng.uniform(-0.05, 0.05) * h
    iris = Ellipse(cx, cy, R * rng.uniform(0.97, 1.03), R, rng.uniform(-10, 10))
    pr = R * rng.uniform(0.25, 0.45)
    pupil = Ellipse.circle(cx + rng.uniform(-0.04, 0.04) * R, cy + rng.uniform(-0.04, 0.04) * R, pr)

    skin = np.array([120, 150, 205], np.float32) * rng.uniform(0.6, 1.1)
    img = np.empty((h, w, 3), np.float32)
    img[:] = skin
    # eye opening (almond): two parabolas
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    half_w = 0.48 * w
    top_y = cy - R * (1 - lid_cover) + R * 1.1 * ((xx - cx) / half_w) ** 2
    bot_y = cy + R * 0.95 - R * 0.9 * ((xx - cx) / half_w) ** 2
    opening = (yy > top_y) & (yy < bot_y) & (np.abs(xx - cx) < half_w)
    sclera = np.array([215, 220, 235], np.float32)
    img[opening] = sclera
    tex = texture if texture is not None else iris_texture(64, 512, IRIS_BGR[color], rng, fibre)
    iris_img, ann = polar_to_image(tex, (h, w), pupil, iris)
    sel = ann.astype(bool) & opening
    img[sel] = iris_img[sel]
    pm = pupil.to_mask((h, w)).astype(bool) & opening
    img[pm] = (15, 15, 18)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    spec = np.zeros((h, w), np.uint8)
    if glare:
        gx, gy = int(cx + 0.3 * pr), int(cy - 0.4 * pr)
        cv2.circle(spec, (gx, gy), max(2, int(0.12 * pr)), 1, -1)
        img[spec.astype(bool)] = 255
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    iris_mask = (sel & ~spec.astype(bool)).astype(np.uint8)
    return img, {"pupil": pupil, "iris": iris, "iris_mask": iris_mask,
                 "pupil_mask": pm.astype(np.uint8), "color": color}
