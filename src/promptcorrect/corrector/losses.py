"""Error-map losses: per-pixel BCE + soft Dice per (sample, channel). Owner: P5."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dice_loss(logits: Tensor, target: Tensor, valid: Tensor | None = None, smooth: float = 1.0) -> Tensor:
    """Mean over samples and channels of 1 - soft Dice. An empty target with an empty prediction costs ~0."""
    p = torch.sigmoid(logits)
    if valid is not None:
        p, target = p * valid, target * valid
    inter = (p * target).flatten(2).sum(-1)
    denom = p.flatten(2).sum(-1) + target.flatten(2).sum(-1)
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def error_map_loss(logits: Tensor, target: Tensor, valid: Tensor | None = None,
                   dice_weight: float = 1.0) -> tuple[Tensor, dict]:
    """logits/target [N,2,S,S]; valid [N,S,S] bool (pixels outside the image are ignored)."""
    if valid is None:
        v, n = None, logits.numel()
    else:
        v = valid[:, None].float()
        n = float(v.sum()) * logits.shape[1]
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=v, reduction="sum") / max(n, 1.0)
    dice = dice_loss(logits, target, v)
    return bce + dice_weight * dice, {"bce": float(bce.detach()), "dice": float(dice.detach())}
