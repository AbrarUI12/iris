"""Config-driven end-to-end iris pipeline.

python -m aeye_iris.pipeline --left L.jpg --right R.jpg --out out_dir [--config pipeline.yaml]
python -m aeye_iris.pipeline --pairs runs/photos --out runs/results

Stages: precheck -> segmentation -> quality assess -> normalise -> colour -> markings
-> ScanObservation JSON (+ overlay PNGs). Output is descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
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
EYE_SIDES = ("left", "right")
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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
            ver = f"{kind}_{method}_k{k}{'_strict' if mcfg.get('strict') else ''}"
            if not q.fine_markings_assessable:
                obs.markings.append(MarkingOut(marking_type=kind, status=Status.unassessable, method_version=ver))
                continue
            cands = fn(norm, method, k=k, strict=mcfg.get("strict", False))
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
            _, encoded = cv2.imencode(".png", v)
            (overlay_dir / f"{image_id}_overlay.png").write_bytes(encoded.tobytes())
        return obs

    # ------------------------------------------------------------------ scan
    def analyze(self, images: dict[str, np.ndarray], scan_id: str | None = None,
                overlay_dir: str | Path | None = None) -> ScanObservation:
        od = Path(overlay_dir) if overlay_dir else None
        if od:
            od.mkdir(parents=True, exist_ok=True)
        eyes = []
        for side in EYE_SIDES:
            if side in images:
                eyes.append(self.analyze_eye(images[side], side, f"{side}_iris", od))
        sid = scan_id or hashlib.sha1(b"".join(images[s].tobytes()[:4096] for s in sorted(images))).hexdigest()[:12]
        return ScanObservation(scan_id=sid, pipeline_version=__version__,
                               zone_map_version=self.zones.version, config=self.cfg, eyes=eyes)


def read_photos(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    images = {}
    for side, path in paths.items():
        try:
            data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except (OSError, cv2.error) as exc:
            raise ValueError(f"cannot read {path}") from exc
        if image is None:
            raise ValueError(f"cannot read {path}")
        images[side] = image
    return images


def pair_paths(folder: Path) -> dict[str, Path]:
    photos = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in PHOTO_EXTENSIONS]
    paths = {}
    for side in EYE_SIDES:
        matches = [path for path in photos if path.stem.lower() == side]
        if len(matches) != 1:
            raise ValueError(f"expected one {side} photo, found {len(matches)}")
        paths[side] = matches[0]
    return paths


def write_scan(pipe: Pipeline, images: dict[str, np.ndarray], out: Path) -> None:
    t0 = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        staging = Path(temp)
        obs = pipe.analyze(images, overlay_dir=staging)
        (staging / "observation.json").write_text(obs.model_dump_json(indent=1), encoding="utf-8")
        out.mkdir(exist_ok=True)
        for side in EYE_SIDES:
            name = f"{side}_iris_overlay.png"
            overlay = staging / name
            if overlay.is_file():
                overlay.replace(out / name)
            else:
                (out / name).unlink(missing_ok=True)
        (staging / "observation.json").replace(out / "observation.json")
    print(f"wrote {out/'observation.json'} in {(time.perf_counter()-t0)*1000:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path)
    ap.add_argument("--right", type=Path)
    ap.add_argument("--pairs", type=Path, help="folder containing one subfolder per left/right photo pair")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", type=Path, default=Path("runs/scan"))
    a = ap.parse_args()
    paths = {side: getattr(a, side) for side in EYE_SIDES if getattr(a, side) is not None}
    if a.pairs is None:
        if not paths:
            ap.error("give --left and/or --right, or use --pairs")
        try:
            images = read_photos(paths)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        write_scan(Pipeline(a.config), images, a.out)
        return

    if paths:
        ap.error("--pairs cannot be combined with --left or --right")
    if not a.pairs.is_dir():
        ap.error(f"pair folder does not exist: {a.pairs}")
    folders = sorted(folder for folder in a.pairs.iterdir() if folder.is_dir())
    if not folders:
        ap.error(f"no pair subfolders in {a.pairs}")
    pipe = Pipeline(a.config)
    failed = False
    for folder in folders:
        try:
            images = read_photos(pair_paths(folder))
            write_scan(pipe, images, a.out / folder.name)
        except (OSError, ValueError, cv2.error) as exc:
            print(f"Skipping {folder.name}: {exc}", file=sys.stderr)
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
