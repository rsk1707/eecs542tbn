"""Ideal-click simulator (the RITM / SimpleClick protocol). Owner: P3.

Next click: distance-transform both error regions (false negatives = GT not covered, false positives
= covered but not GT), with the image border counted as boundary. Click at the pixel deepest inside
whichever error is deeper: a positive click for a false negative, a negative one for a false positive.
Pixels already clicked are excluded. The corrector's training target (P5) uses the same rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import distance_transform_edt

from promptcorrect.eval.metrics import iou
from promptcorrect.types import Prompt, SegOutput


def _depth(region: np.ndarray) -> np.ndarray:
    return distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]


def ideal_click(pred: np.ndarray, gt: np.ndarray, clicked=()) -> tuple[float, float, int] | None:
    """(x, y, label) of the next ideal click, or None if the mask is exact. `clicked`: previous (x, y, label)."""
    fn, fp = gt & ~pred, pred & ~gt
    if clicked:
        done = np.zeros_like(gt)
        for x, y, _ in clicked:
            done[int(y), int(x)] = True
        fn, fp = fn & ~done, fp & ~done
    d_fn, d_fp = _depth(fn), _depth(fp)
    positive = d_fn.max() > d_fp.max()  # RITM: ties go to the negative click
    d = d_fn if positive else d_fp
    if d.max() == 0:
        return None
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return float(x), float(y), int(positive)


@dataclass
class OracleTrace:
    """Step 0 is the starting prompt; step k has k ideal clicks."""
    ious: list[float]
    pred_ious: list[float]                     # SAM's own predicted IoU per step
    latency_ms: list[float]                    # per step (step 0 as passed in)
    clicks: list[tuple[float, float, int]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)


def run_oracle(segmenter, prompt: Prompt, gt: np.ndarray, max_clicks: int = 5, start: SegOutput | None = None,
               start_ms: float = float("nan"), use_prev_mask: bool = True, keep_masks: bool = False) -> OracleTrace:
    """Add up to `max_clicks` ideal clicks to `prompt` (box and/or points), one SAM call per click.

    `segmenter.set_image` must already be called. `start` reuses an existing step-0 output (e.g. from a
    batched box call). With `use_prev_mask` each call also gets the previous low-res logits, as in SAM's
    interactive mode. Stops early when the mask is exact. IoU is not always monotone in the number of
    clicks; use `best_so_far` for headroom curves.
    """
    if start is None:
        t = time.perf_counter()
        start = segmenter.predict([prompt])[0]
        start_ms = (time.perf_counter() - t) * 1000
    out, mask = start, segmenter.full_res_mask(start)
    tr = OracleTrace([iou(mask, gt)], [out.iou_score], [start_ms], masks=[mask] if keep_masks else [])
    points = list(prompt.points)
    for _ in range(max_clicks):
        click = ideal_click(mask, gt, tr.clicks)
        if click is None:
            break
        tr.clicks.append(click)
        points.append(click)
        p = replace(prompt, points=list(points), mask_logits=out.mask_logits if use_prev_mask else None,
                    origin="oracle")
        t = time.perf_counter()
        out = segmenter.predict([p])[0]
        tr.latency_ms.append((time.perf_counter() - t) * 1000)
        mask = segmenter.full_res_mask(out)
        tr.ious.append(iou(mask, gt))
        tr.pred_ious.append(out.iou_score)
        if keep_masks:
            tr.masks.append(mask)
    return tr


def best_so_far(ious: list[float]) -> list[float]:
    return np.maximum.accumulate(ious).tolist()


def noc(ious: list[float], thr: float, max_clicks: int) -> int:
    """Clicks needed to reach IoU >= thr (0 = the starting prompt is enough). Not reached -> max_clicks + 1,
    so averages are a lower bound; report the not-reached rate next to it."""
    for k, v in enumerate(ious):
        if v >= thr:
            return k
    return max_clicks + 1
