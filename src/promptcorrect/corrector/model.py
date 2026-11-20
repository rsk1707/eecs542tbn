"""Corrector v0: a small U-Net that predicts where the frozen segmenter's mask is wrong. Owner: P5.

Input channels (all in the S x S object-centric crop, see crops.py), built by `build_inputs`:
  0-2  RGB, ImageNet-normalized
  3    tanh(logit / 4)          soft mask from the segmenter
  4    logit > 0                the binary mask actually evaluated
  5    prompt box (0/1)
  6    valid (0 outside the image)
  7-8  positive / negative click heatmaps (Gaussian, sigma 3 crop pixels)
Output: 2-channel error-map logits, channel 0 = false positive (mask but not GT), 1 = false negative.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from promptcorrect.corrector.crops import click_heatmaps

IN_CHANNELS = 9
FP, FN = 0, 1
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_inputs(rgb: Tensor, logits: Tensor, box: Tensor, valid: Tensor, clicks: Tensor) -> Tensor:
    """Batch tensors (rgb uint8 [N,3,S,S], logits [N,S,S], box/valid bool [N,S,S], clicks [N,K,3]) -> [N,9,S,S]."""
    S = rgb.shape[-1]
    mean = torch.tensor(_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=rgb.device).view(1, 3, 1, 1)
    lg = logits.float()
    return torch.cat([
        (rgb.float() / 255 - mean) / std,
        torch.tanh(lg / 4)[:, None],
        (lg > 0).float()[:, None],
        box.float()[:, None],
        valid.float()[:, None],
        click_heatmaps(clicks.float(), S),
    ], dim=1)


def error_targets(logits: Tensor, gt: Tensor) -> Tensor:
    """[N,S,S] logits and GT -> [N,2,S,S] float targets (FP, FN)."""
    mask, gt = logits > 0, gt.bool()
    return torch.stack([mask & ~gt, gt & ~mask], dim=1).float()


class _Block(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class CorrectorNet(nn.Module):
    def __init__(self, in_ch: int = IN_CHANNELS, base: int = 32, depth: int = 4, out_ch: int = 2):
        super().__init__()
        chs = [base * 2**i for i in range(depth)]
        self.enc = nn.ModuleList([_Block(in_ch if i == 0 else chs[i - 1], c) for i, c in enumerate(chs)])
        self.dec = nn.ModuleList([_Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth - 1))])
        self.head = nn.Conv2d(base, out_ch, 1)
        self.config = {"in_ch": in_ch, "base": base, "depth": depth, "out_ch": out_ch}

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for block in self.dec:
            skip = skips.pop()
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.head(x)


def save_checkpoint(model: CorrectorNet, path, **extra) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[CorrectorNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = CorrectorNet(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
