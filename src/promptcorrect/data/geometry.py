"""Single implementation of box/RLE conversions. Convert once at load time, never again."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def xywh_to_xyxy(box_xywh: list[float]) -> list[float]:
    """COCO [x, y, w, h] -> [x0, y0, x1, y1] (continuous pixel edges)."""
    x, y, w, h = box_xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def rle_encode(mask: np.ndarray) -> dict:
    """Boolean/0-1 mask [H, W] -> COCO RLE with `counts` as str (JSON-safe)."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(s) for s in rle["size"]], "counts": rle["counts"].decode("ascii")}


def rle_decode(rle: dict) -> np.ndarray:
    """COCO RLE (counts as str or bytes) -> boolean mask [H, W]."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return mask_utils.decode({"size": list(rle["size"]), "counts": counts}).astype(bool)


def mask_to_box(mask: np.ndarray) -> list[float]:
    """Tight xyxy box around a boolean mask (continuous edges: max index + 1)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
