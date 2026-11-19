"""Build data/manifest.csv with PERSON-LEVEL train/val/test splits.

Label masks are converted to two binary PNGs per image (visible iris, pupil) under
data/cache/masks/<dataset>/ so every dataset looks the same to train/ and bench/.

Splits are assigned by a hash of the subject id (60/15/25), so both eyes and all
repeat captures of a person land in the same split.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

RAW = Path("data/raw")
CACHE = Path("data/cache/masks")
OUT = Path("data/manifest.csv")


def split_of(subject: str, salt: str) -> str:
    h = int(hashlib.sha1(f"{salt}:{subject}".encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else ("val" if h < 75 else "test")


def write_masks(ds, stem, iris, pupil):
    d = CACHE / ds
    d.mkdir(parents=True, exist_ok=True)
    ip, pp = d / f"{stem}_iris.png", d / f"{stem}_pupil.png"
    cv2.imwrite(str(ip), (iris > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(pp), (pupil > 0).astype(np.uint8) * 255)
    return str(ip), str(pp)


def nir_pupil_eye_iris():
    """itguides/pupil-eye-and-iris-segmentation: grayscale NIR 640x480,
    mask classes 0 bg, 1 sclera/eye, 2 pupil, 3 iris (verified visually). Name: <subject>_<eye>_..."""
    base = RAW / "pupil-eye-iris-seg" / "IRIS + PUPIL + EYE"
    rows = []
    for part in ("train", "val"):
        for f in sorted((base / part / "image").glob("*.png")):
            m = cv2.imread(str(base / part / "segmentation" / f.name), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            subj = f.stem.split("_")[0]
            ip, pp = write_masks("nir_eye", f.stem, m == 3, m == 2)
            rows.append(dict(id=f"nir_eye/{f.stem}", dataset="nir_eye", image=str(f), iris_mask=ip,
                             pupil_mask=pp, subject=subj, split=split_of(subj, "nir"), eye=f.stem.split("_")[1]))
    return rows


SYN_RGB = {"bg": (255, 255, 255), "sclera": (156, 204, 78), "iris": (41, 141, 132),
           "pupil": (41, 255, 64), "face": (0, 158, 204)}  # resources/segmentation_color_map.json (RGB)


def synthetic_eyes(max_n: int = 6000, seed: int = 0):
    """allexmendes/synthetic-human-eyes (CC BY 4.0): 640x360 RGB periocular renders with
    colour-coded masks. We keep renders with a reasonably open eye and crop a 4:3 window
    around the iris with RANDOM offset/scale (simulating app-guided framing without
    handing the segmenter the centre), saved at 400x300 like UBIRIS.
    Each render = its own 'subject' (render identities are not published)."""
    base = RAW / "synthetic-eyes"
    out_img = Path("data/cache/images/synth_eyes")
    out_img.mkdir(parents=True, exist_ok=True)
    names = list(SYN_RGB)
    pal = np.array([SYN_RGB[n][::-1] for n in names], np.float32)  # -> BGR
    rng = np.random.default_rng(seed)
    rows = []
    for f in sorted((base / "lit").glob("*.png"))[:max_n]:
        s = cv2.imread(str(base / "segmentation" / f.name), cv2.IMREAD_COLOR)
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if s is None or img is None:
            continue
        d = ((s[:, :, None, :].astype(np.float32) - pal[None, None]) ** 2).sum(-1)
        lab = d.argmin(-1)
        iris, pupil = lab == names.index("iris"), lab == names.index("pupil")
        union = (iris | pupil).astype(np.uint8)
        if union.sum() < 300:
            continue
        cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), R = cv2.minEnclosingCircle(c)
        if R < 12 or union.sum() < 0.45 * np.pi * R * R:  # mostly closed eye
            continue
        # QC: in some renders the mask ignores blink/gaze (verified visually). Keep only
        # renders whose mask agrees with the pixels.
        try:
            par = json.loads((base / "annotation" / f"{f.stem}.json").read_text())["Parameters"]
            if par.get("EyeBlink", 0) > 0.3 or par.get("EyeSquint", 0) > 0.5:
                continue
        except Exception:
            pass
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        scl = lab == names.index("sclera")
        if scl.sum() < 100 or pupil.sum() < 30:
            continue
        mi, ms_, mp = np.median(L[iris]), np.median(L[scl]), np.median(L[pupil])
        if ms_ - mi < 20 or mi - mp < 3:
            continue
        side_w = R * rng.uniform(4.0, 6.0)
        side_h = side_w * 0.75
        ox, oy = rng.uniform(-0.15, 0.15) * side_w, rng.uniform(-0.12, 0.12) * side_h
        x0, y0 = cx + ox - side_w / 2, cy + oy - side_h / 2
        M = np.float32([[400 / side_w, 0, -x0 * 400 / side_w], [0, 300 / side_h, -y0 * 300 / side_h]])
        crop = cv2.warpAffine(img, M, (400, 300), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        ci = cv2.warpAffine(iris.astype(np.uint8), M, (400, 300), flags=cv2.INTER_NEAREST)
        cp = cv2.warpAffine(pupil.astype(np.uint8), M, (400, 300), flags=cv2.INTER_NEAREST)
        ip_ = out_img / f.name
        cv2.imwrite(str(ip_), crop)
        ip, pp = write_masks("synth_eyes", f.stem, ci, cp)
        subj = f.stem
        rows.append(dict(id=f"synth_eyes/{f.stem}", dataset="synth_eyes", image=str(ip_), iris_mask=ip,
                         pupil_mask=pp, subject=subj, split=split_of(subj, "syn"), eye=""))
    return rows


def upol():
    """UPOL: RGB 576x768 iris close-ups, no masks. <subj><L|R>_<n>.png.
    Hand-annotated circles (data/labels/upol_circles.csv) become masks if present."""
    base = RAW / "upol"
    circ = {}
    lp = Path("data/labels/upol_circles.csv")
    if lp.exists():
        for r in csv.DictReader(open(lp)):
            circ[r["id"]] = r
    rows = []
    for f in sorted(base.rglob("*.png")):
        subj = f.stem[:3]
        rid = f"upol/{f.stem}"
        ip = pp = ""
        if rid in circ:
            c = circ[rid]
            im = cv2.imread(str(f))
            h, w = im.shape[:2]
            I = np.zeros((h, w), np.uint8)
            P = np.zeros((h, w), np.uint8)
            cv2.circle(I, (int(float(c["icx"])), int(float(c["icy"]))), int(float(c["ir"])), 1, -1)
            cv2.circle(P, (int(float(c["pcx"])), int(float(c["pcy"]))), int(float(c["pr"])), 1, -1)
            ip, pp = write_masks("upol", f.stem, I & (1 - P), P)
        rows.append(dict(id=rid, dataset="upol", image=str(f), iris_mask=ip, pupil_mask=pp, subject=subj,
                         split=split_of(subj, "upol"), eye=f.stem[3]))
    return rows


def upol_pseudo():
    """Daugman pseudo-labels on UPOL TRAIN images that passed strict QC
    (scripts/pseudo_label_upol.py). Train split only, by construction."""
    lp = Path("data/labels/upol_pseudo.csv")
    if not lp.exists():
        return []
    return [dict(id=r["id"], dataset="upol_pseudo", image=r["image"], iris_mask=r["iris_mask"],
                 pupil_mask=r["pupil_mask"], subject=r["subject"], split="train", eye="")
            for r in csv.DictReader(open(lp))]


def main():
    rows = []
    for fn in (nir_pupil_eye_iris, synthetic_eyes, upol, upol_pseudo):
        try:
            r = fn()
        except StopIteration:
            r = []
        print(fn.__name__, len(r), {s: sum(x["split"] == s for x in r) for s in ("train", "val", "test")})
        rows += r
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "dataset", "image", "iris_mask", "pupil_mask", "subject", "split", "eye"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
