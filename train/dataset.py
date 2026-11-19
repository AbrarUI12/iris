"""Manifest-driven dataset for iris/pupil segmentation training.

Manifest CSV columns: id, dataset, image, iris_mask, pupil_mask, subject, split
Label encoding: 0 background, 1 visible iris, 2 pupil.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path: str | Path, split: str | None = None, datasets: list[str] | None = None) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    if datasets:
        rows = [r for r in rows if r["dataset"] in datasets]
    return rows


def load_label(row: dict, root: Path) -> np.ndarray:
    im = cv2.imread(str(root / row["iris_mask"]), cv2.IMREAD_GRAYSCALE) > 127
    lab = im.astype(np.uint8)
    if row.get("pupil_mask"):
        pm = cv2.imread(str(root / row["pupil_mask"]), cv2.IMREAD_GRAYSCALE) > 127
        lab[pm & ~im] = 2
    return lab


class SegDataset(Dataset):
    def __init__(self, rows: list[dict], root: str | Path, size: int = 224, augment: bool = False,
                 cache: bool = True, zoom_p: float = 0.55):
        self.rows, self.root, self.size, self.augment = rows, Path(root), size, augment
        self.zoom_p = zoom_p
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.rows)

    def _load(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        r = self.rows[i]
        img = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        lab = load_label(r, self.root)
        if lab.shape != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # cache at 3x training size: iris-centred zoom-in needs the extra resolution
        s = (3 * self.size) / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self._cache is not None:
            self._cache[i] = (img, lab)
        return img, lab

    def _aug(self, img, lab, rng):
        h, w = img.shape[:2]
        ang = rng.uniform(-15, 15)
        union = (lab > 0).astype(np.uint8)
        if rng.random() < self.zoom_p and union.sum() > 50:
            # iris-centred zoom: covers macro close-ups where the iris FILLS the frame
            # (A-EYE capture, clinical iris cameras), not only periocular framing
            (cx, cy), R = cv2.minEnclosingCircle(cv2.findNonZero(union))
            target = rng.uniform(0.25, 0.9) * min(h, w) / 2
            sc = max(1.0, target / max(R, 1.0))  # zoom IN only (zoom-out tiles mirrored eyes)
            M = cv2.getRotationMatrix2D((cx, cy), ang, sc)
            M[:, 2] += (w / 2 - cx, h / 2 - cy)
            M[:, 2] += rng.uniform(-0.12, 0.12, 2) * (w, h)
        else:
            sc = rng.uniform(0.8, 1.25)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            M[:, 2] += rng.uniform(-0.1, 0.1, 2) * (w, h)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lab = cv2.warpAffine(lab, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        if rng.random() < 0.2:
            # circular aperture vignette (macro lens / clinical iris camera look)
            yy, xx = np.mgrid[0:h, 0:w]
            rr = rng.uniform(0.45, 0.75) * max(h, w)
            cxv, cyv = w / 2 + rng.uniform(-0.1, 0.1) * w, h / 2 + rng.uniform(-0.1, 0.1) * h
            out = np.hypot(xx - cxv, yy - cyv) > rr
            img = img.copy()
            img[out] = 0
            lab = lab.copy()
            lab[out] = 0
        # photometric: brightness/contrast/gamma/hue/white balance, blur, noise, jpeg
        f = img.astype(np.float32)
        f = f * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
        f = f * rng.uniform(0.85, 1.15, 3)  # white balance
        f = np.clip(f, 0, 255)
        f = 255 * (f / 255) ** rng.uniform(0.7, 1.4)
        if rng.random() < 0.3:
            f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 2.0))
        if rng.random() < 0.3:
            f = f + rng.normal(0, rng.uniform(2, 8), f.shape)
        img = np.clip(f, 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(20, 70))])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return np.ascontiguousarray(img), np.ascontiguousarray(lab)

    def __getitem__(self, i):
        img, lab = self._load(i)
        if self.augment:
            rng = np.random.default_rng()
            img, lab = self._aug(img, lab, rng)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(lab.astype(np.int64))
        u = (lab > 0).astype(np.uint8)
        edge = (cv2.dilate(u, np.ones((3, 3), np.uint8)) - cv2.erode(u, np.ones((3, 3), np.uint8))) > 0
        p = (lab == 2).astype(np.uint8)
        edge |= (cv2.dilate(p, np.ones((3, 3), np.uint8)) - cv2.erode(p, np.ones((3, 3), np.uint8))) > 0
        return x, y, torch.from_numpy(edge.astype(np.float32))[None]
