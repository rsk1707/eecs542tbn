"""Prediction-to-GT matching. Owner: P3."""

from __future__ import annotations

import numpy as np


def match(preds: list[np.ndarray], gts: list[np.ndarray]):
    """TODO(P3): Hungarian matching on mask IoU.

    Document and fix the output contract (docs/interfaces.md, issue O12): matched pairs with IoU, plus
    unmatched predictions and unmatched GTs. Ignore predictions that overlap filtered/crowd objects (O1).
    """
    raise NotImplementedError
