"""Stand-in for P3's `eval.matching.match`, following the contract published in docs/interfaces.md section 7.

Owner: P1, TEMPORARY. Used by the RQ0 runner only while `promptcorrect.eval.matching.match` raises
NotImplementedError; the run manifest records which implementation produced the numbers. Once P3's
version lands it is picked up automatically and this file can be deleted.

Contract: one-to-one Hungarian matching that maximizes total mask IoU among non-ignored predictions;
assigned pairs below `min_iou` are dropped and both sides count as unmatched; a prediction is ignored
if at least `ignore_frac` of its area lies inside the union of `ignore_masks` (COCO-style crowd handling).
All indices refer to the input lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (pred_idx, gt_idx, mask IoU)
    unmatched_preds: list[int] = field(default_factory=list)           # false positives
    unmatched_gts: list[int] = field(default_factory=list)             # missed objects
    ignored_preds: list[int] = field(default_factory=list)             # mostly inside crowd / filtered objects


def _iou_matrix(preds: list[np.ndarray], gts: list[np.ndarray]) -> np.ndarray:
    P = np.stack([p.reshape(-1) for p in preds]).astype(np.float32)
    G = np.stack([g.reshape(-1) for g in gts]).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(pred_masks, gt_masks, ignore_masks=(), min_iou: float = 0.1, ignore_frac: float = 0.5) -> MatchResult:
    pred_masks = [np.asarray(m, dtype=bool) for m in pred_masks]
    gt_masks = [np.asarray(m, dtype=bool) for m in gt_masks]
    res = MatchResult()
    ignore = None
    if len(ignore_masks):
        ignore = np.logical_or.reduce([np.asarray(m, dtype=bool) for m in ignore_masks])
    active = []
    for i, p in enumerate(pred_masks):
        area = p.sum()
        if ignore is not None and area > 0 and np.logical_and(p, ignore).sum() >= ignore_frac * area:
            res.ignored_preds.append(i)
        else:
            active.append(i)
    if not active or not gt_masks:
        res.unmatched_preds = list(active)
        res.unmatched_gts = list(range(len(gt_masks)))
        return res
    iou = _iou_matrix([pred_masks[i] for i in active], gt_masks)
    rows, cols = linear_sum_assignment(-iou)
    used_p, used_g = set(), set()
    for r, c in zip(rows, cols):
        if iou[r, c] >= min_iou:
            res.pairs.append((active[r], int(c), float(iou[r, c])))
            used_p.add(active[r]); used_g.add(int(c))
    res.pairs.sort()
    res.unmatched_preds = [i for i in active if i not in used_p]
    res.unmatched_gts = [j for j in range(len(gt_masks)) if j not in used_g]
    return res
