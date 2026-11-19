"""LightIrisNet-style compact segmenter: MobileNetV3 (ImageNet) encoder + small U-Net
decoder, 3 classes (0 background, 1 visible iris, 2 pupil) + optional boundary head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _taps(features: nn.Sequential, size: int = 224) -> list[int]:
    """Indices of the last block at each output stride (2,4,8,16,32)."""
    x = torch.zeros(1, 3, size, size)
    taps, prev = [], size
    with torch.no_grad():
        for i, m in enumerate(features):
            x = m(x)
            if x.shape[-1] < prev:
                if taps:
                    pass
                prev = x.shape[-1]
                taps.append(i)
            else:
                if taps:
                    taps[-1] = i
    return taps


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.Hardswish(inplace=True))


class DSConv(nn.Sequential):
    """Depthwise-separable conv block (cheap decoder)."""
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                         nn.BatchNorm2d(cin), nn.Hardswish(inplace=True),
                         nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.Hardswish(inplace=True))


class IrisSegNet(nn.Module):
    def __init__(self, backbone: str = "small", pretrained: bool = True, dec: int = 32,
                 n_classes: int = 3, edge_head: bool = True):
        super().__init__()
        if backbone == "small":
            enc = tvm.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        elif backbone == "large":
            enc = tvm.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(backbone)
        self.features = enc.features
        self.taps = _taps(self.features)
        with torch.no_grad():
            chs, x = [], torch.zeros(1, 3, 224, 224)
            for i, m in enumerate(self.features):
                x = m(x)
                if i in self.taps:
                    chs.append(x.shape[1])
        self.lat = nn.ModuleList([ConvBNAct(c, dec, 1) for c in chs])
        self.dec = nn.ModuleList([DSConv(dec, dec) for _ in chs[:-1]])
        self.head = nn.Sequential(DSConv(dec, dec), nn.Conv2d(dec, n_classes, 1))
        self.edge = nn.Conv2d(dec, 1, 1) if edge_head else None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):  # x: B,3,H,W float in [0,1], RGB
        H, W = x.shape[-2:]
        x = (x - self.mean) / self.std
        feats = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.taps:
                feats.append(x)
        y = self.lat[-1](feats[-1])
        for j in range(len(feats) - 2, -1, -1):
            y = F.interpolate(y, size=feats[j].shape[-2:], mode="bilinear", align_corners=False)
            y = self.dec[j](y + self.lat[j](feats[j]))
        logits = F.interpolate(self.head(y), size=(H, W), mode="bilinear", align_corners=False)
        if self.edge is not None and self.training:
            edge = F.interpolate(self.edge(y), size=(H, W), mode="bilinear", align_corners=False)
            return logits, edge
        return logits


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
