"""Evaluation metrics for segmentation, classification and detection."""
from __future__ import annotations

import cv2
import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def nice1_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """NICE.I E1: fraction of disagreeing pixels."""
    return float(((pred > 0) ^ (gt > 0)).mean())


def boundary_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile symmetric boundary distance (pixels)."""
    def edge(m):
        m = (m > 0).astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    ep, eg = edge(pred), edge(gt)
    if ep.sum() == 0 or eg.sum() == 0:
        return float("nan")
    dt_g = cv2.distanceTransform((~eg).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~ep).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.concatenate([dt_g[ep], dt_p[eg]])
    return float(np.percentile(d, 95))


def center_error(pred_e, gt_e, norm: float) -> float:
    if pred_e is None or gt_e is None:
        return float("nan")
    return float(np.hypot(pred_e.cx - gt_e.cx, pred_e.cy - gt_e.cy) / max(norm, 1e-6))


# ----------------------------------------------------------------------------- detection
def match_regions(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], thr: float = 0.1):
    """Greedy IoU matching. Returns (tp, fp, fn, matched_ious). A low threshold is used
    because small markings in normalised polar space have imprecise extents."""
    used = set()
    tp, ious = 0, []
    for p in pred_masks:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_masks):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj)
            tp += 1
            ious.append(best)
    return tp, len(pred_masks) - tp, len(gt_masks) - len(used), ious


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
