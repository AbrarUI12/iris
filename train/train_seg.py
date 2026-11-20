"""Train the compact iris/pupil segmenter on CPU or CUDA and export ONNX.

python -m train.train_seg --backbone small --size 224 --epochs 25 --out models/seg_small_224
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import SegDataset, read_manifest
from .model import IrisSegNet, n_params


def soft_dice(logits, y, cls):
    p = logits.softmax(1)[:, cls]
    t = (y == cls).float()
    inter = (p * t).sum((1, 2))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()


@torch.no_grad()
def evaluate(model, loader, dev="cpu"):
    model.eval()
    d = {1: [], 2: []}
    for x, y, _ in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(1)
        for c in (1, 2):
            a, b = pred == c, y == c
            s = (a.sum((1, 2)) + b.sum((1, 2))).float()
            dd = torch.where(s > 0, 2 * (a & b).sum((1, 2)).float() / s.clamp(min=1), torch.ones_like(s))
            d[c] += dd.tolist()
    return float(np.mean(d[1])), float(np.mean(d[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbone", default="small")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dec", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--repeat", nargs="*", default=[], help="dataset:k oversampling, e.g. upol_pseudo:4")
    ap.add_argument("--init", default=None, help="state_dict to start from (fine-tuning)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (uses an NVIDIA GPU if available)")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (set 4+ on a GPU machine)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)
    torch.manual_seed(0)
    np.random.seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = read_manifest(a.manifest, "train", a.datasets)
    va = read_manifest(a.manifest, "val", a.datasets)
    if a.max_train:
        tr = tr[: a.max_train]
    for spec in a.repeat:
        ds, k = spec.split(":")
        tr += [r for r in tr if r["dataset"] == ds] * (int(k) - 1)
    dl_tr = DataLoader(SegDataset(tr, a.root, a.size, True), batch_size=a.bs, shuffle=True,
                       num_workers=a.workers, drop_last=True)
    dl_va = DataLoader(SegDataset(va, a.root, a.size, False), batch_size=32, num_workers=0)
    model = IrisSegNet(a.backbone, True, a.dec)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = model.to(dev)
    print(f"train {len(tr)} val {len(va)} params {n_params(model)/1e6:.2f}M", flush=True)
    enc = [p for n, p in model.named_parameters() if n.startswith("features")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("features")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * 0.3}, {"params": rest, "lr": a.lr}],
                            weight_decay=1e-4)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=steps,
                                                pct_start=0.1 if steps >= 50 else 0.3)
    cw = torch.tensor([0.5, 1.0, 1.5], device=dev)
    best, hist = -1, []
    for ep in range(a.epochs):
        model.train()
        t0, tl = time.time(), 0.0
        for x, y, e in dl_tr:
            x, y, e = x.to(dev), y.to(dev), e.to(dev)
            logits, edge = model(x)
            loss = (F.cross_entropy(logits, y, weight=cw) + soft_dice(logits, y, 1)
                    + soft_dice(logits, y, 2) + 0.5 * F.binary_cross_entropy_with_logits(
                        edge, e, pos_weight=torch.tensor(5.0, device=dev)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
        di, dp = evaluate(model, dl_va, dev)
        hist.append({"epoch": ep, "loss": tl / len(dl_tr), "val_dice_iris": di, "val_dice_pupil": dp,
                     "sec": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        score = di + 0.5 * dp
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "best.pt")
    model = model.cpu()
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    model.eval()
    onnx_path = Path(str(out) + ".onnx")  # e.g. models/seg_small_224.onnx
    torch.onnx.export(model, torch.rand(1, 3, a.size, a.size), str(onnx_path), input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "history": hist,
                                                    "params": n_params(model)}, indent=1))
    print("exported", onnx_path, flush=True)


if __name__ == "__main__":
    main()
