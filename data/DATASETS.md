# Datasets used by the benchmark

Raw data is **not committed** (`data/raw/` is git-ignored). Re-download with a free Kaggle
API token (`~/.kaggle/kaggle.json`):

```bash
python scripts/fetch_datasets.py pupil-eye-iris-seg upol synthetic-eyes
python scripts/build_manifest.py          # -> data/manifest.csv (person-level 60/15/25 splits)
```

| Name in manifest | Source | Images used | Ground truth | Licence (as listed) | Role |
|---|---|---|---|---|---|
| `synth_eyes` | Kaggle `allexmendes/synthetic-human-eyes` (50k RGB renders) | 1,573 renders that pass mask QC, cropped 4:3 around the eye with random offset/scale | Pixel masks (pupil/iris/sclera) | CC BY 4.0 | **Visible-light** segmentation train/val/test with exact masks |
| `nir_eye` | Kaggle `itguides/pupil-eye-and-iris-segmentation` | 1,275 grayscale NIR 640x480 | Human-annotated masks: classes 0 bg, 1 eye, **2 pupil, 3 iris** (verified visually) | MIT | Real-photo segmentation train/val/test (NIR domain) |
| `upol` | Kaggle `ouaraskhelilrafik/iris-images` = UPOL Iris DB (Dobeš & Machala) | 384 RGB 576x768 iris close-ups (64 people x 2 eyes x 3) | None (see labels below) | Academic use, cite the authors | **Real visible-light close-ups**: visual segmentation grading, colour labels, marking-injection hosts, end-to-end runs |

**Not used:**
* **UBIRIS.v2** (Kaggle `chinmoyslg/ubirisv2`): the images ship inside a **password-protected** archive. The password comes with the authors' research licence. Only the IRISSEG-EP masks were readable, so the dataset could not be used. No attempt was made to bypass the protection. If you obtain the licence, add the images and they slot into `build_manifest.py`.
* MICHE, MOBIUS, CUVIRIS and OpenEDS are not openly downloadable (request forms).

## QC applied to the synthetic renders

The source segmentation sometimes ignores blink/gaze. Renders are kept only if blink ≤ 0.3, squint ≤ 0.5, the sclera is ≥ 20 L* brighter than the masked iris, and the pupil is darker than the iris. A visual spot-check shows about 1 in 10 remaining masks still mismatched. This lowers every method's Dice on this set equally.

## Derived labels committed to this repo

* `data/labels/color_labels.csv`: iris colour appearance (blue / brown / green_mixed / uncertain) for all 384 UPOL images.
  Each of the 128 eyes was labelled once from the contact sheets in `data/labels/sheets/`, and the label applies to all
  3 photos of that eye. One annotator, no adjudication: this is a **pilot** label set. The UPOL camera has a strong warm cast, so blue irises look grey.
* `results/visual*/grades.csv`: pass/fail visual grades of segmentation overlays on UPOL (validation split used for tuning, test split for the final numbers).

## What these datasets can and cannot tell us

* They **can** benchmark capture-quality metrics, iris/pupil localisation (exact masks on synthetic RGB and real NIR; visual grading on real RGB close-ups), normalisation and colour measurement.
* They **cannot** benchmark the six SRS marking categories against practitioner judgement. No public dataset has such labels, so detectors are evaluated by *synthetic injection* into real UPOL iris textures (`bench/markings.py`).
* None of the photos were taken by a phone held 10–20 cm from the eye. **Every threshold, the colour model and ideally the segmenter must be re-calibrated or fine-tuned on consented A-EYE photos.**
