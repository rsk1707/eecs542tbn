"""Single implementation of IoU metrics. Owner: P3. Nobody else re-implements these."""

from __future__ import annotations

import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """TODO(P3): IoU of two boolean masks [H, W] at original resolution."""
    raise NotImplementedError


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """TODO(P3): Boundary IoU (Cheng et al. 2021). Fix and document the dilation ratio."""
    raise NotImplementedError
