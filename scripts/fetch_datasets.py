"""Download the public iris datasets used by the benchmark from Kaggle.

Uses the Kaggle REST API on www.kaggle.com (auth via ~/.kaggle/kaggle.json or the
session proxy). Honours HTTP 429 Retry-After. Files land in data/raw/<name>/ and are
git-ignored. See data/DATASETS.md for licenses / caveats.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "ubirisv2": "chinmoyslg/ubirisv2",  # UBIRIS.v2 images + IRISSEG-EP (OperatorA) masks
    "pupil-eye-iris-seg": "itguides/pupil-eye-and-iris-segmentation",  # MIT, NIR eye/iris/pupil masks
    "upol": "ouaraskhelilrafik/iris-images",  # UPOL iris DB: RGB close-ups, no masks
    "synthetic-eyes": "allexmendes/synthetic-human-eyes",  # CC BY 4.0, RGB renders + masks
}
API = "https://www.kaggle.com/api/v1/datasets/download/{ref}"


def _auth_header() -> dict:
    p = Path.home() / ".kaggle" / "kaggle.json"
    if not p.exists():
        return {}
    c = json.loads(p.read_text())
    tok = base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
    return {"Authorization": f"Basic {tok}"}


def download(ref: str, dest: Path, max_wait: int = 4 * 3600) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "archive.zip"
    if out.exists() and zipfile.is_zipfile(out):
        return out
    waited = 0
    while True:
        req = urllib.request.Request(API.format(ref=ref), headers=_auth_header())
        try:
            with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if zipfile.is_zipfile(out):
                return out
            raise RuntimeError(f"{ref}: response is not a zip: {out.read_bytes()[:200]!r}")
        except urllib.error.HTTPError as e:
            if e.code == 429 and waited < max_wait:
                wait = int(e.headers.get("Retry-After", "600")) + 30
                print(f"[{ref}] 429, sleeping {wait}s", flush=True)
                time.sleep(wait)
                waited += wait
                continue
            raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("names", nargs="*", default=list(DATASETS))
    a = ap.parse_args()
    for name in a.names:
        dest = Path(a.root) / name
        z = download(DATASETS[name], dest)
        print(f"[{name}] downloaded {z.stat().st_size/1e6:.0f} MB; extracting", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        for inner in list(dest.rglob("*.zip")):
            if inner.name != "archive.zip":
                with zipfile.ZipFile(inner) as zf:
                    if any(i.flag_bits & 1 for i in zf.infolist()):
                        # e.g. UBIRIS.v2 images: password from the authors' licence; never bypassed
                        print(f"[{name}] {inner.name} is password-protected; skipped", flush=True)
                        continue
                    zf.extractall(inner.parent / inner.stem)
                inner.unlink()
        z.unlink()
        print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
