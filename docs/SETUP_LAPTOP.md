# Running A-EYE iris on your own laptop (e.g. RTX 4060)

**Short answer: yes, it runs.** Analysis (inference) is CPU-only by design: ~0.5 s for a two-eye scan
on a single core, so any recent laptop is fine and the GPU is **not needed**. The RTX 4060 only
helps if you want to **re-train** the segmentation model, roughly 5–10x faster than CPU.

| Task | Needs GPU? | Laptop (4060, any modern CPU) |
|---|---|---|
| Analyse photos (`aeye_iris.pipeline`) | No | ~0.3–0.6 s per two-eye scan |
| Run tests | No | seconds |
| Re-run benchmarks | No | 5–30 min each (needs the datasets, ~14 GB) |
| Re-train segmenter | Optional | GPU: a few min per model; CPU: ~20–40 min |

Requirements: Python 3.10–3.13, Git, ~2 GB disk for the code and libraries, and ~15 GB more if you download the datasets.

---

## 1. Get the code

```bash
git clone https://github.com/AbrarUI12/iris.git
```

```bash
cd iris
```

```bash
git checkout claude/tender-cray-w7opo5
```

The trained models (`models/*.onnx`, `models/*.json`) are in the repo, so you can analyse photos immediately.

## 2. Create an environment and install

**Windows (PowerShell):**

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

**Linux / macOS / WSL:**

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

Then, on any OS, install the core package plus test and benchmark tools (no PyTorch needed for analysis):

```bash
pip install -e ".[bench,dev]"
```

## 3. Check it works

```bash
pytest -q
```

Analyse your own two photos (one image per eye):

```bash
python -m aeye_iris.pipeline --left path/to/left_eye.jpg --right path/to/right_eye.jpg --out runs/scan1
```

Output: `runs/scan1/observation.json`, plus `left_iris_overlay.png` / `right_iris_overlay.png` showing the
zones and marking candidates.

**Photo tips:** one eye per photo, iris centred and sharp, eye wide open, diffuse light with no direct
flash. A close-up where the iris fills most of the frame works best.

## 4. (Optional) Use the RTX 4060 for training

Install the CUDA build of PyTorch. Check the exact command for your driver at pytorch.org; for current
NVIDIA drivers:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install -e ".[train]"
```

Confirm the GPU is visible (should print `True NVIDIA GeForce RTX 4060 Laptop GPU`):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`train/train_seg.py` uses the GPU automatically (`--device auto`) and prints `device: cuda (...)` at the
start. On a GPU machine add `--workers 4` so image loading keeps up.

## 5. (Optional) Re-download data and reproduce training/benchmarks

1. Create a free Kaggle account, then go to **Settings → API → Create New Token**. Put the downloaded
   `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
2. Download the datasets and build the splits:

```bash
python scripts/fetch_datasets.py pupil-eye-iris-seg upol synthetic-eyes
```

```bash
python scripts/build_manifest.py
```

```bash
python scripts/pseudo_label_upol.py
```

```bash
python scripts/build_manifest.py
```

The Synthetic Human Eyes download is ~13 GB. Fetch one dataset at a time (`python scripts/fetch_datasets.py synthetic-eyes`) so an interrupted download only repeats that one.

3. Train the default model on the GPU:

```bash
python -m train.train_seg --backbone small --size 160 --epochs 15 --datasets synth_eyes nir_eye --workers 4 --out models/seg_small_160
```

4. Benchmarks and report: see the "Reproduce the benchmark" section of `README.md`.

## Troubleshooting

* **`onnxruntime` / `opencv` install errors on Windows:** upgrade pip first (`python -m pip install -U pip`).
* **PDF reports (report layer) on Windows:** WeasyPrint needs the GTK runtime. Install
  "GTK3-Runtime Win64" or use WSL. Image analysis does not need it.
* **`torch.cuda.is_available()` is False:** update the NVIDIA driver and install the CUDA build of torch
  (step 4), not the default CPU build.
* **Out of GPU memory** (unlikely with 8 GB): lower `--bs` to 8.
