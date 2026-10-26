"""Single implementation of prompt severity. Owner: P4.

Synthetic perturbations and real detector boxes are both measured with `describe`, so RQ2 compares them
on the same scale. Pixel convention: a box covers the pixels whose *centre* lies inside it, so a GT box
(continuous edges, `x1 = max index + 1`) covers its whole mask.
"""

from __future__ import annotations

import math

import numpy as np

from promptcorrect.data.geometry import rle_decode
from promptcorrect.types import ObjectRecord, Prompt

NAN = float("nan")
COLUMNS = ("box_iou_in", "coverage", "contamination", "delta_cx", "delta_cy", "delta_log_sw", "delta_log_sh")


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two xyxy boxes (continuous coordinates)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_slices(box: list[float], image_hw: tuple[int, int]) -> tuple[slice, slice]:
    """(rows, cols) of the pixels whose centre lies inside `box`, clipped to the image."""
    H, W = image_hw
    x0, y0, x1, y1 = (min(max(int(math.floor(v + 0.5)), 0), lim) for v, lim in zip(box, (W, H, W, H)))
    return slice(y0, max(y0, y1)), slice(x0, max(x0, x1))


def _mask(obj: ObjectRecord, cache: dict[str, np.ndarray] | None) -> np.ndarray:
    if cache is None:
        return rle_decode(obj.gt_mask)
    if obj.obj_id not in cache:
        cache[obj.obj_id] = rle_decode(obj.gt_mask)
    return cache[obj.obj_id]


def describe(prompt: Prompt, obj: ObjectRecord, neighbors: list[ObjectRecord], *,
             mask_cache: dict[str, np.ndarray] | None = None) -> dict:
    """-> {box_iou, coverage, contamination, delta} for a prompt aimed at `obj`.

    box_iou: IoU of the prompt box with the GT box.
    coverage: fraction of the GT mask's pixels inside the prompt box (1 = nothing truncated).
    contamination: fraction of the prompt box's pixels covered by `neighbors`' masks (other objects in the
        image; `obj` itself is skipped if present, and pixels of `obj`'s own mask never count).
        Neighbours are the object records, so crowd and size-filtered annotations are not counted.
    delta: (dcx / w, dcy / h, log(w' / w), log(h' / h)), prompt box relative to the GT box of size w x h.
    Box-based fields are NaN when `prompt.box` is None (issue O3). `mask_cache` (obj_id -> bool mask) avoids
    decoding the same RLE repeatedly in sweeps.
    """
    if prompt.box is None:
        return {"box_iou": NAN, "coverage": NAN, "contamination": NAN, "delta": (NAN, NAN, NAN, NAN)}
    pb, gb = [float(v) for v in prompt.box], obj.gt_box
    gt = _mask(obj, mask_cache)
    rows, cols = box_slices(pb, gt.shape)
    coverage = float(gt[rows, cols].sum() / gt.sum())

    n_box = (rows.stop - rows.start) * (cols.stop - cols.start)
    contamination = 0.0
    if n_box:
        others = np.zeros((rows.stop - rows.start, cols.stop - cols.start), dtype=bool)
        for nb in neighbors:
            if nb.obj_id != obj.obj_id:
                others |= _mask(nb, mask_cache)[rows, cols]
        contamination = float((others & ~gt[rows, cols]).sum() / n_box)

    gw, gh = gb[2] - gb[0], gb[3] - gb[1]
    pw, ph = max(pb[2] - pb[0], 1e-6), max(pb[3] - pb[1], 1e-6)
    delta = (((pb[0] + pb[2]) - (gb[0] + gb[2])) / 2 / gw, ((pb[1] + pb[3]) - (gb[1] + gb[3])) / 2 / gh,
             math.log(pw / gw), math.log(ph / gh))
    return {"box_iou": box_iou(pb, gb), "coverage": coverage, "contamination": contamination,
            "delta": tuple(float(d) for d in delta)}


def severity_columns(sev: dict) -> dict:
    """`describe()` output -> the ResultRow column names (box_iou_in, coverage, ..., delta_log_sh)."""
    return dict(zip(COLUMNS, (sev["box_iou"], sev["coverage"], sev["contamination"], *sev["delta"])))
