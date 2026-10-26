"""Predicted error maps -> one corrective click. Owner: P5.

The rule is the standard ideal-click convention (RITM/SimpleClick; docs/demo_sprint.md: click at the point
farthest from the boundary of the largest error region), applied to the corrector's thresholded
*predictions* instead of the truth: pick the channel with more predicted error mass (FN -> positive click,
FP -> negative click), take its largest connected component (by probability mass), click at the point
farthest from its boundary. If no
pixel passes the threshold, or the region is tiny, the corrector abstains: this is what lets it leave good
prompts alone (RQ4).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from promptcorrect.corrector.model import FN, FP


def propose_click(probs: np.ndarray, threshold: float = 0.5, min_area: int = 4) -> tuple[float, float, int] | None:
    """probs float [2, S, S] (FP, FN) -> (u, v, label) at a crop pixel centre, or None (abstain)."""
    best = None
    for ch, label in ((FN, 1), (FP, 0)):
        region = probs[ch] > threshold
        if region.sum() < min_area:
            continue
        comp, n = ndimage.label(region)
        mass = ndimage.sum(probs[ch], comp, range(1, n + 1))
        k = int(np.argmax(mass))
        if best is None or mass[k] > best[0]:
            best = (float(mass[k]), comp == k + 1, label)
    if best is None or best[1].sum() < min_area:
        return None
    _, region, label = best
    dist = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    return float(j) + 0.5, float(i) + 0.5, label


def click_hits(click: tuple[float, float, int] | None, fp: np.ndarray, fn: np.ndarray) -> bool | None:
    """True if a positive click lands on a true FN pixel or a negative click on a true FP pixel; None on abstain."""
    if click is None:
        return None
    u, v, label = click
    S = fp.shape[0]
    i, j = min(int(v), S - 1), min(int(u), S - 1)
    return bool(fn[i, j]) if label == 1 else bool(fp[i, j])
