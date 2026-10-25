"""Prediction-to-GT matching (docs/interfaces.md section 7). Owner: P3.

Call once per (image_id, phrase): predictions for that phrase against the GT objects of that category.
Ignore masks are the annotations of the image that are *not* in the object records (crowd or
size-filtered); build them from the original COCO annotations with `data.records.annotation_mask`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (pred_idx, gt_idx, mask IoU), by pred_idx
    unmatched_preds: list[int] = field(default_factory=list)           # false positives
    unmatched_gts: list[int] = field(default_factory=list)             # missed objects
    ignored_preds: list[int] = field(default_factory=list)             # mostly inside an ignore region


def iou_matrix(preds: list[np.ndarray], gts: list[np.ndarray]) -> np.ndarray:
    """[n_pred, n_gt] mask IoU."""
    P = np.stack([p.reshape(-1) for p in preds]).astype(np.float32)
    G = np.stack([g.reshape(-1) for g in gts]).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(pred_masks, gt_masks, ignore_masks=(), min_iou: float = 0.1, ignore_frac: float = 0.5) -> MatchResult:
    """One-to-one Hungarian matching maximizing total mask IoU.

    A prediction with at least `ignore_frac` of its area inside the union of `ignore_masks` is ignored
    (neither a match nor a false positive, as COCO does for crowd regions). Assigned pairs with IoU below
    `min_iou` are dropped and both sides count as unmatched. Indices refer to the input lists.
    """
    preds = [np.asarray(m, dtype=bool) for m in pred_masks]
    gts = [np.asarray(m, dtype=bool) for m in gt_masks]
    shapes = {m.shape for m in preds + gts + [np.asarray(m) for m in ignore_masks]}
    if len(shapes) > 1:
        raise ValueError(f"all masks must have the same shape, got {shapes}")

    res = MatchResult()
    ignore = np.logical_or.reduce([np.asarray(m, dtype=bool) for m in ignore_masks]) if len(ignore_masks) else None
    active = []
    for i, p in enumerate(preds):
        area = p.sum()
        if ignore is not None and area and (p & ignore).sum() >= ignore_frac * area:
            res.ignored_preds.append(i)
        else:
            active.append(i)

    if active and gts:
        ious = iou_matrix([preds[i] for i in active], gts)
        for r, c in zip(*linear_sum_assignment(ious, maximize=True)):
            if ious[r, c] >= min_iou:
                res.pairs.append((active[r], int(c), float(ious[r, c])))
    matched_p = {p for p, _, _ in res.pairs}
    matched_g = {g for _, g, _ in res.pairs}
    res.unmatched_preds = [i for i in active if i not in matched_p]
    res.unmatched_gts = [j for j in range(len(gts)) if j not in matched_g]
    return res
