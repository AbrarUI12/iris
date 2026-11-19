"""Static INT8 (QDQ, per-channel) quantisation of an exported segmenter with ONNX Runtime.

python -m train.quantize_int8 models/seg_small_224.onnx --manifest data/manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
import onnx
from onnxruntime.quantization.shape_inference import quant_pre_process

from .dataset import read_manifest


class Reader(CalibrationDataReader):
    def __init__(self, rows, root, size, name):
        self.it = iter(rows)
        self.root, self.size, self.name = Path(root), size, name

    def get_next(self):
        r = next(self.it, None)
        if r is None:
            return None
        img = cv2.imread(str(self.root / r["image"]))
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = x[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return {self.name: x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    src = Path(a.model)
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = int(inp.shape[-1])
    rows = read_manifest(a.manifest, "train")
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), replace=False)]
    pre = src.with_name(src.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre))
    dst = src.with_name(src.stem + "_int8.onnx")
    # keep the input normalisation (Sub/Div) and the output head/upsampling in float:
    # they carry large-range activations that wreck MinMax/percentile scales
    g = onnx.load(str(pre)).graph
    ops = [(n.name, n.op_type) for n in g.node]
    excl = [n for n, t in ops[:4] if t in ("Sub", "Div", "Mul", "Add")]
    convs = [n for n, t in ops if t == "Conv"]
    excl += convs[-2:] + [n for n, t in ops if t in ("Resize",)][-2:]
    quantize_static(str(pre), str(dst), Reader(rows, a.root, size, inp.name),
                    quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile, nodes_to_exclude=excl,
                    extra_options={"CalibPercentile": 99.99})
    pre.unlink()
    print("wrote", dst, f"{dst.stat().st_size/1e6:.2f} MB (fp32 {src.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
