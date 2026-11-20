"""Object-centric crops: the corrector sees a fixed S x S window around the prompt box. Owner: P5.

The window is the prompt box grown by `context` times its size on every side, so GT that a shifted or
truncated box cut off is still visible. It is resized anisotropically to S x S. Coordinates are continuous
pixel edges, as everywhere else (docs/interfaces.md section 0): crop pixel (i, j) covers
[j, j+1) x [i, i+1) in crop units.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

OUTSIDE_LOGIT = -20.0  # logits outside the image: confidently background


def crop_window(box: list[float], context: float = 0.5, min_side: float = 16.0) -> list[float]:
    """Prompt box xyxy -> crop window xyxy (may extend past the image)."""
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, min_side), max(y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = w * (0.5 + context), h * (0.5 + context)
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def _grid(window: list[float], hw: tuple[int, int], size: int) -> Tensor:
    H, W = hw
    x0, y0, x1, y1 = window
    u = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    xs = (x0 + u * (x1 - x0)) * 2 / W - 1  # align_corners=False normalization
    ys = (y0 + u * (y1 - y0)) * 2 / H - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)[None]  # [1, S, S, 2]


def crop_resize(x: Tensor, window: list[float], size: int) -> tuple[Tensor, Tensor]:
    """Bilinear crop of [C, H, W] -> ([C, S, S], valid [S, S] bool: inside the image)."""
    C, H, W = x.shape
    src = torch.cat([x.float(), torch.ones(1, H, W, dtype=torch.float32, device=x.device)])[None]
    out = F.grid_sample(src, _grid(window, (H, W), size).to(x.device), mode="bilinear",
                        padding_mode="zeros", align_corners=False)[0]
    return out[:C], out[C] > 0.5


def crop_object(image: Tensor, logits: Tensor, gt: np.ndarray | None, window: list[float], size: int) -> dict:
    """One sample's crops. `image` uint8 [3, H, W], `logits` full-res [H, W] (segmenter.to_full_res),
    `gt` bool [H, W] or None. Returns uint8/fp16 numpy arrays ready to store."""
    chans = [image.float(), logits.float()[None]]
    if gt is not None:
        chans.append(torch.from_numpy(gt).float()[None])
    out, valid = crop_resize(torch.cat(chans), window, size)
    lg = torch.where(valid, out[3], torch.full_like(out[3], OUTSIDE_LOGIT)).clamp(-32, 32)
    res = {"rgb": out[:3].round().clamp(0, 255).to(torch.uint8).numpy(),
           "logits": lg.to(torch.float16).numpy(), "valid": valid.numpy()}
    if gt is not None:
        res["gt"] = (out[4] > 0.5).numpy()
    return res


def image_to_crop(x: float, y: float, window: list[float], size: int) -> tuple[float, float]:
    x0, y0, x1, y1 = window
    return (x - x0) / (x1 - x0) * size, (y - y0) / (y1 - y0) * size


def crop_to_image(u: float, v: float, window: list[float], size: int) -> tuple[float, float]:
    x0, y0, x1, y1 = window
    return x0 + u / size * (x1 - x0), y0 + v / size * (y1 - y0)


def box_mask(box: list[float], window: list[float], size: int) -> np.ndarray:
    """Prompt box rasterized in crop coordinates (pixel centres inside the box), bool [S, S]."""
    u0, v0 = image_to_crop(box[0], box[1], window, size)
    u1, v1 = image_to_crop(box[2], box[3], window, size)
    c = np.arange(size) + 0.5
    return ((c[None, :] >= u0) & (c[None, :] < u1)) & ((c[:, None] >= v0) & (c[:, None] < v1))


def clicks_to_crop(points: list[tuple[float, float, int]], window: list[float], size: int,
                   max_clicks: int) -> np.ndarray:
    """Image-space clicks -> float32 [max_clicks, 3] (u, v, label) in crop pixels; padding rows have label -1.
    Keeps the most recent `max_clicks`."""
    out = np.full((max_clicks, 3), -1.0, dtype=np.float32)
    for k, (x, y, label) in enumerate(points[-max_clicks:] if max_clicks else []):
        u, v = image_to_crop(x, y, window, size)
        out[k] = (u, v, float(label))
    return out


def click_heatmaps(clicks: Tensor, size: int, sigma: float = 3.0) -> Tensor:
    """clicks [N, K, 3] (u, v, label; label -1 = padding) -> [N, 2, S, S] (positive, negative) Gaussians, max-pooled."""
    N, K, _ = clicks.shape
    c = torch.arange(size, device=clicks.device, dtype=torch.float32) + 0.5
    du = (c[None, None, None, :] - clicks[..., 0, None, None]) ** 2   # [N, K, 1, S]
    dv = (c[None, None, :, None] - clicks[..., 1, None, None]) ** 2   # [N, K, S, 1]
    g = torch.exp(-(du + dv) / (2 * sigma**2))                        # [N, K, S, S]
    label = clicks[..., 2]
    pos = (g * (label == 1)[..., None, None]).amax(dim=1) if K else torch.zeros(N, size, size, device=clicks.device)
    neg = (g * (label == 0)[..., None, None]).amax(dim=1) if K else torch.zeros(N, size, size, device=clicks.device)
    return torch.stack([pos, neg], dim=1)
