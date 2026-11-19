"""Learned / pretrained segmenters.

* onnx      – our LightIrisNet-style MobileNetV3 U-Net exported to ONNX (fp32 or int8).
              Mask -> pupil ellipse (fit) + limbus ellipse (fit to lateral boundary
              points only, so eyelid-truncated masks still give the full circle).
* hybrid    – onnx coarse geometry, then classical band-limited ray-cast refinement on
              the image gradient (sub-pixel boundary at model resolution limits).
* mediapipe – MediaPipe iris_landmark.tflite (Apache-2.0) limbus + eyelid contour,
              classical integro-differential pupil search inside the limbus.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..geometry import Ellipse, ellipse_from_mask, ransac_ellipse
from ..preprocess import prepare, resize_max, specular_mask
from ..types import SegResult
from .base import finalize, ray_profiles, register, smooth1d

MODELS = Path(__file__).resolve().parents[2] / "models"


def _limbus_from_mask(union: np.ndarray, center: tuple[float, float] | None = None) -> Ellipse | None:
    """Fit the limbus to the outer boundary of (iris U pupil), using only boundary points
    in the lateral sectors (eyelids truncate the top/bottom of the visible mask)."""
    m = (union > 0).astype(np.uint8)
    if m.sum() < 30:
        return None
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        m = (lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if center is None:
        mo = cv2.moments(m)
        center = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
    ang = np.arctan2(c[:, 1] - center[1], c[:, 0] - center[0])
    lat = np.abs(np.sin(ang)) < 0.7
    pts = c[lat] if lat.sum() > 20 else c
    e = ransac_ellipse(pts[:, 0], pts[:, 1], tol=max(1.5, 0.02 * np.sqrt(m.sum() / np.pi)), max_ecc=1.3)
    if e is None:
        return ellipse_from_mask(m)
    return e


class _OnnxBase:
    def __init__(self, model: str = "seg_small_224.onnx", size: int | None = None,
                 threads: int = 1, occl_k: float = 3.0):
        import onnxruntime as ort
        path = Path(model)
        if not path.exists():
            path = MODELS / model
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        shp = self.sess.get_inputs()[0].shape
        self.size = size or int(shp[-1])
        self.model_name = path.stem
        self.occl_k = occl_k

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        x = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = x[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        logits = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0]
        # upsample logits (not labels) for smoother boundaries
        lg = np.stack([cv2.resize(l, (w, h), interpolation=cv2.INTER_LINEAR) for l in logits])
        return lg.argmax(0).astype(np.uint8)

    def geometry(self, lab: np.ndarray):
        pupil_m = (lab == 2).astype(np.uint8)
        union = (lab > 0).astype(np.uint8)
        pupil = ellipse_from_mask(pupil_m) if pupil_m.sum() > 20 else None
        limbus = _limbus_from_mask(union, pupil.center if pupil else None)
        return pupil, limbus


@register("onnx")
class OnnxSeg(_OnnxBase):
    def __call__(self, bgr: np.ndarray) -> SegResult:
        lab = self.predict(bgr)
        pupil, limbus = self.geometry(lab)
        if limbus is not None and pupil is None:  # pupil not segmented: typical ratio
            pupil = Ellipse.circle(limbus.cx, limbus.cy, 0.3 * limbus.r)
        if pupil is None or limbus is None:
            return SegResult(None, None, method=f"onnx[{self.model_name}]")
        # visible-iris mask straight from the network (it learned lids/lashes/glints)
        iris_mask = ((lab == 1) & (limbus.to_mask(lab.shape) > 0)).astype(np.uint8)
        return SegResult(pupil, limbus, iris_mask, (lab == 2).astype(np.uint8),
                         f"onnx[{self.model_name}]")


@register("hybrid")
class HybridSeg(_OnnxBase):
    """Model geometry + classical band-limited edge refinement."""

    def __init__(self, band: float = 0.08, n_rays: int = 64, **kw):
        super().__init__(**kw)
        self.band, self.n_rays = band, n_rays

    def _refine(self, g, e: Ellipse, angles, sign=+1):
        out_x, out_y = [], []
        for phi in angles:
            r0 = float(e.radius_at(np.array([phi]))[0])
            radii = np.linspace(r0 * (1 - self.band), r0 * (1 + self.band), 24, dtype=np.float32)
            p = smooth1d(ray_profiles(g, e.center, radii, np.array([phi]))[0], 1.0)
            d = np.diff(p) * sign
            j = int(np.argmax(d))
            if d[j] <= 0:
                continue
            r = (radii[j] + radii[j + 1]) / 2
            out_x.append(e.cx + r * np.cos(phi))
            out_y.append(e.cy + r * np.sin(phi))
        if len(out_x) < 8:
            return e
        f = ransac_ellipse(np.array(out_x), np.array(out_y), tol=max(1.0, 0.02 * e.r), max_ecc=1.3)
        if f is None or abs(f.r - e.r) > 2 * self.band * e.r:
            return e
        return f

    def __call__(self, bgr: np.ndarray) -> SegResult:
        lab = self.predict(bgr)
        pupil, limbus = self.geometry(lab)
        tag = f"hybrid[{self.model_name}]"
        if limbus is None:
            return SegResult(None, None, method=tag)
        g = cv2.GaussianBlur(prepare(bgr, "red", False, True)[0], (0, 0), 1.0).astype(np.float32)
        lat = np.concatenate([np.linspace(-35, 45, self.n_rays // 2),
                              np.linspace(135, 215, self.n_rays // 2)]) * np.pi / 180
        limbus = self._refine(g, limbus, lat)
        if pupil is None:
            pupil = Ellipse.circle(limbus.cx, limbus.cy, 0.3 * limbus.r)
        else:
            pupil = self._refine(g, pupil, np.linspace(-50, 230, self.n_rays) * np.pi / 180)
        iris_mask = ((lab == 1) & (limbus.to_mask(lab.shape) > 0) & (pupil.to_mask(lab.shape) == 0))
        return SegResult(pupil, limbus, iris_mask.astype(np.uint8),
                         pupil.to_mask(lab.shape), tag)


@register("mediapipe")
class MediaPipeIris:
    """MediaPipe iris landmark model on a square crop around the eye.

    The model was trained on face-pipeline eye crops where the iris occupies a fixed
    fraction of the crop. We therefore run it at several crop scales and iterate:
    crop side = `k` x current iris-radius estimate (k calibrated in the benchmark).
    Pupil: classical integro-differential search inside the limbus.
    """

    def __init__(self, k: float = 4.0, iters: int = 3, occl: bool = True, occl_k: float = 3.0,
                 lid_mask: bool = True):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # pragma: no cover
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        p = MODELS / "pretrained" / "iris_landmark.tflite"
        self.it = Interpreter(model_path=str(p), num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.outs = {o["name"]: o["index"] for o in self.it.get_output_details()}
        self.k, self.iters, self.occl, self.occl_k, self.lid_mask = k, iters, occl, occl_k, lid_mask

    def _run(self, rgb, cx, cy, side):
        M = np.float32([[64 / side, 0, 32 - cx * 64 / side], [0, 64 / side, 32 - cy * 64 / side]])
        x = cv2.warpAffine(rgb, M, (64, 64), flags=cv2.INTER_AREA,
                           borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
        self.it.set_tensor(self.inp, x[None])
        self.it.invoke()
        iris = self.it.get_tensor(self.outs["output_iris"]).reshape(5, 3)[:, :2]
        eye = self.it.get_tensor(self.outs["output_eyes_contours_and_brows"]).reshape(71, 3)[:, :2]
        to_img = lambda p: (p - 32) * side / 64 + np.array([cx, cy])  # noqa: E731
        return to_img(iris), to_img(eye)

    def __call__(self, bgr: np.ndarray) -> SegResult:
        from .classical import PUPIL_ANGLES, _grid, _ido_search
        tag = f"mediapipe[k={self.k}]"
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cx, cy, side = w / 2, h / 2, float(max(h, w))
        iris = eye = None
        for i in range(self.iters):
            iris, eye = self._run(rgb, cx, cy, side)
            c = iris[0]
            r = float(np.mean(np.hypot(*(iris[1:] - c).T)))
            cx, cy = float(c[0]), float(c[1])
            if i < self.iters - 1:
                side = max(16.0, self.k * r)
        limbus = Ellipse.circle(cx, cy, r)
        # pupil: classical search inside the limbus
        g, clean, spec = prepare(bgr, "red", False, True)
        gs, s = resize_max(g, 240)
        gs = cv2.GaussianBlur(gs, (0, 0), 1.0).astype(np.float32)
        R = r * s
        c = _grid(cx * s - 0.2 * R, cx * s + 0.2 * R, cy * s - 0.2 * R, cy * s + 0.2 * R, max(1.0, R / 20))
        radii = np.arange(max(2.0, 0.12 * R), 0.75 * R, 1.0, dtype=np.float32)
        if len(radii) < 4:
            return SegResult(None, None, method=tag)
        (px, py), pr, _ = _ido_search(gs, c, radii, PUPIL_ANGLES, 1.0)
        pupil = Ellipse.circle(px / s, py / s, pr / s)
        res = finalize(clean, pupil, limbus, spec, tag, self.occl, self.occl_k)
        if self.lid_mask and res.ok and eye is not None:
            # first 16 contour points = eye opening (lower 9 + upper 7) in MediaPipe order
            poly = np.round(eye[:16]).astype(np.int32)
            lid = np.zeros((h, w), np.uint8)
            cv2.fillPoly(lid, [cv2.convexHull(poly)], 1)
            if lid.sum() > 0.3 * res.iris_mask.sum():
                res.iris_mask = (res.iris_mask & lid).astype(np.uint8)
        return res
