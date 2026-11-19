"""Config-driven end-to-end iris pipeline.

python -m aeye_iris.pipeline --left L.jpg --right R.jpg --out out_dir [--config pipeline.yaml]

Stages: precheck -> segmentation -> quality assess -> normalise -> colour -> markings
-> ScanObservation JSON (+ overlay PNGs). Output is descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import __version__
from .color import LearnedColor, classify_thresholds, iris_pixels_lab
from .markings.detectors import DETECTORS, collarette, fibre_orientation
from .normalize import normalize, polar_mask_to_image
from .quality import QualityThresholds, assess, precheck
from .segment import build
from .types import (ColorOut, EllipseOut, EyeObservation, MarkingOut, ScanObservation,
                    SegmentationOut, Status)
from .zones import ZoneMap

DEFAULT_CONFIG = Path(__file__).with_name("pipeline.yaml")


class Pipeline:
    def __init__(self, config: dict | str | Path | None = None):
        if config is None:
            config = DEFAULT_CONFIG
        if not isinstance(config, dict):
            config = yaml.safe_load(Path(config).read_text())
        self.cfg = config
        s = config["segmentation"]
        self.seg = build(s["name"], **s.get("params", {}))
        qcfg = dict(config.get("quality", {}))
        qm_path = qcfg.pop("model_path", None)
        self.qt = QualityThresholds(**qcfg)
        self.qmodel = None
        if qm_path:
            mp = Path(qm_path)
            if not mp.exists():
                mp = Path(__file__).resolve().parents[1] / qm_path
            if mp.exists():
                from .quality import QualityModel
                self.qmodel = QualityModel.load(mp)
        n = config.get("normalization", {})
        self.n_r, self.n_t = n.get("n_r", 64), n.get("n_t", 512)
        self.mirror_left = n.get("mirror_left", False)
        c = config.get("color", {})
        self.color_model = None
        if c.get("model_path"):
            mp = Path(c["model_path"])
            if not mp.exists():
                mp = Path(__file__).resolve().parents[1] / c["model_path"]
            if mp.exists():
                self.color_model = LearnedColor.load(mp)
        self.color_margin = c.get("uncertain_margin", 0.15)
        self.zones = ZoneMap.load(config["zone_map"]) if config.get("zone_map") else ZoneMap()
        self.mk = config.get("markings", {})

    # ------------------------------------------------------------------ one eye
    def analyze_eye(self, bgr: np.ndarray, eye_side: str, image_id: str, overlay_dir: Path | None = None) -> EyeObservation:
        t = {}
        t0 = time.perf_counter()
        pre = precheck(bgr, self.qt)
        t["precheck"] = (time.perf_counter() - t0) * 1000
        if pre.status == "retake":
            return EyeObservation(image_id=image_id, eye_side=eye_side, quality=pre, timings_ms=t)
        t0 = time.perf_counter()
        seg = self.seg(bgr)
        t["segmentation"] = (time.perf_counter() - t0) * 1000
        q = assess(bgr, seg, self.qt, self.qmodel)
        t["quality"] = (time.perf_counter() - t0) * 1000 - t["segmentation"]
        obs = EyeObservation(image_id=image_id, eye_side=eye_side, quality=q, timings_ms=t)
        if not seg.ok:
            return obs
        ann_frac = q.metrics.get("visible_frac", 0.0)
        obs.segmentation = SegmentationOut(
            method_version=seg.method, pupil=EllipseOut(**seg.pupil.to_dict()),
            iris=EllipseOut(**seg.iris.to_dict()), visible_iris_fraction=round(ann_frac, 4))
        # colour
        t0 = time.perf_counter()
        lab = iris_pixels_lab(bgr, seg.iris_mask, seg.pupil, seg.iris)
        if q.color_assessable and len(lab) >= 200:
            if self.color_model is not None:
                cat, conf = self.color_model.predict(lab, self.color_margin)
                ver = f"color_{self.color_model.p['model']}_{self.color_model.kind}"
            else:
                cat, conf = classify_thresholds(lab, self.color_margin)
                ver = "color_thresholds_v1"
            obs.color = ColorOut(category=cat, confidence=round(conf, 3),
                                 lab_median=[round(float(x), 2) for x in np.median(lab, 0)], method_version=ver)
        t["color"] = (time.perf_counter() - t0) * 1000
        # markings
        t0 = time.perf_counter()
        norm = normalize(bgr, seg, self.n_r, self.n_t, eye_side, self.mirror_left)
        all_cands = []
        for kind, (fn, _methods) in DETECTORS.items():
            mcfg = self.mk.get(kind, {})
            method = mcfg.get("method", _methods[0])
            k = mcfg.get("k", 3.0)
            ver = f"{kind}_{method}_k{k}"
            if not q.fine_markings_assessable:
                obs.markings.append(MarkingOut(marking_type=kind, status=Status.unassessable, method_version=ver))
                continue
            cands = fn(norm, method, k=k)
            all_cands += cands
            if not cands:
                obs.markings.append(MarkingOut(marking_type=kind, status=Status.not_detected, method_version=ver))
            for i, c in enumerate(sorted(cands, key=lambda c: -c.score)[: mcfg.get("max_per_eye", 10)]):
                r, col = c.centroid
                sector, ring = self.zones.zone_of_polar(r, col, self.n_r, self.n_t)
                obs.markings.append(MarkingOut(
                    marking_type=kind, status=Status.present_candidate, region_id=f"{kind}_{i+1}",
                    sector=sector, ring=ring, polar_bbox=c.bbox, score=round(c.score, 3),
                    attributes=c.attrs, method_version=ver))
        # collarette + fibre: descriptive measurements
        if q.fine_markings_assessable:
            cm = self.mk.get("collarette", {}).get("method", "gradient")
            col = collarette(norm, cm)
            if col:
                attrs = {k: v for k, v in col.items() if k != "rho_curve"}
                obs.markings.append(MarkingOut(marking_type="collarette", status=Status.present_candidate,
                                               region_id="collarette", attributes=attrs,
                                               method_version=f"collarette_{cm}"))
            fm = self.mk.get("fibre_type", {}).get("method", "structure_tensor")
            fo = fibre_orientation(norm, fm)
            if fo:
                thr = self.mk.get("fibre_type", {}).get("threshold", 0.2)
                label = "radial" if fo["radial_index"] > thr else ("transverse" if fo["radial_index"] < -thr else "mixed")
                fo["label"] = label
                obs.markings.append(MarkingOut(marking_type="fibre_type", status=Status.present_candidate,
                                               region_id="fibre", attributes=fo, method_version=f"fibre_{fm}"))
        else:
            for kind in ("collarette", "fibre_type"):
                obs.markings.append(MarkingOut(marking_type=kind, status=Status.unassessable))
        t["markings"] = (time.perf_counter() - t0) * 1000
        obs.timings_ms = {k: round(v, 2) for k, v in t.items()}
        if overlay_dir is not None:
            v = self.zones.draw(bgr, seg.pupil, seg.iris)
            colors = {"dark_pigmentation": (0, 0, 255), "white_marking": (255, 255, 255),
                      "lacuna": (255, 0, 255), "contraction_furrow": (255, 128, 0)}
            for c in all_cands:
                m = polar_mask_to_image(c.mask, norm, bgr.shape)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(v, cnts, -1, colors.get(c.kind, (0, 255, 0)), 1)
            cv2.imwrite(str(overlay_dir / f"{image_id}_overlay.png"), v)
        return obs

    # ------------------------------------------------------------------ scan
    def analyze(self, images: dict[str, np.ndarray], scan_id: str | None = None,
                overlay_dir: str | Path | None = None) -> ScanObservation:
        od = Path(overlay_dir) if overlay_dir else None
        if od:
            od.mkdir(parents=True, exist_ok=True)
        eyes = []
        for side in ("left", "right"):
            if side in images:
                eyes.append(self.analyze_eye(images[side], side, f"{side}_iris", od))
        sid = scan_id or hashlib.sha1(b"".join(images[s].tobytes()[:4096] for s in sorted(images))).hexdigest()[:12]
        return ScanObservation(scan_id=sid, pipeline_version=__version__,
                               zone_map_version=self.zones.version, config=self.cfg, eyes=eyes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left")
    ap.add_argument("--right")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default="runs/scan")
    a = ap.parse_args()
    imgs = {}
    for side in ("left", "right"):
        p = getattr(a, side)
        if p:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise SystemExit(f"cannot read {p}")
            imgs[side] = im
    if not imgs:
        raise SystemExit("give --left and/or --right")
    pipe = Pipeline(a.config)
    t0 = time.perf_counter()
    obs = pipe.analyze(imgs, overlay_dir=a.out)
    out = Path(a.out)
    (out / "observation.json").write_text(obs.model_dump_json(indent=1))
    print(f"wrote {out/'observation.json'} in {(time.perf_counter()-t0)*1000:.0f} ms")


if __name__ == "__main__":
    main()
