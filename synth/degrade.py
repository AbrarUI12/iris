"""Controlled capture degradations with known labels, for the quality-gate benchmark."""
from __future__ import annotations

import cv2
import numpy as np

KINDS = ("defocus", "motion", "dark", "overexposed", "glare", "jpeg", "lowres")


def degrade(bgr: np.ndarray, kind: str, rng: np.random.Generator, severity: float = 1.0) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = severity
    if kind == "defocus":
        r = int(round(2 + 4 * s)) * max(1, w // 400)
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1, -1)
        return cv2.filter2D(bgr, -1, k / k.sum())
    if kind == "motion":
        L = int(round(6 + 10 * s)) * max(1, w // 400)
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1)
        k = cv2.warpAffine(k, M, (L, L))
        return cv2.filter2D(bgr, -1, k / max(k.sum(), 1e-6))
    if kind == "dark":
        return np.clip(bgr.astype(np.float32) * (0.25 - 0.12 * s), 0, 255).astype(np.uint8)
    if kind == "overexposed":
        return np.clip(bgr.astype(np.float32) * (2.2 + 1.0 * s) + 30, 0, 255).astype(np.uint8)
    if kind == "glare":
        out = bgr.astype(np.float32)
        for _ in range(int(2 + 3 * s)):
            cx, cy = rng.uniform(0.35, 0.65) * w, rng.uniform(0.35, 0.65) * h
            r = rng.uniform(0.04, 0.09) * min(h, w) * (0.7 + s)
            yy, xx = np.mgrid[0:h, 0:w]
            a = np.clip(1.3 - np.hypot(xx - cx, yy - cy) / r, 0, 1)[..., None]
            out = out * (1 - a) + 255 * a
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        q = int(max(3, 12 - 8 * s))
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if kind == "lowres":
        f = 0.25 - 0.1 * s
        small = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(kind)
