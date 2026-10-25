"""Mask metrics. The only implementation of iou() and boundary_iou() in the repo.

Masks are boolean arrays at original image resolution. Callers convert logits
with `segmenter.to_full_res(logits) > 0` first; float or int masks are rejected
so logits can't be passed by accident.
"""

import numpy as np
from scipy.ndimage import binary_erosion

BOUNDARY_DILATION_RATIO = 0.02  # Cheng et al., CVPR 2021 default


def _check(pred, gt) -> tuple[np.ndarray, np.ndarray]:
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.dtype != bool or gt.dtype != bool:
        raise TypeError(f"masks must be bool, got {pred.dtype} and {gt.dtype}")
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"masks must be 2-D with equal shapes, got {pred.shape} and {gt.shape}")
    if not gt.any():
        raise ValueError("GT mask is empty; such objects should be filtered out of the records")
    return pred, gt


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def iou(pred, gt) -> float:
    """Mask IoU. An empty prediction scores 0."""
    return _iou(*_check(pred, gt))


def mask_boundary(mask: np.ndarray, dilation_ratio: float = BOUNDARY_DILATION_RATIO) -> np.ndarray:
    """Pixels of `mask` within d pixels of its contour, d = ratio * image diagonal (min 1).

    Same as the official boundary-iou-api: erosion with a 3x3 kernel, d iterations,
    treating everything outside the image as background.
    """
    h, w = mask.shape
    d = max(1, int(round(dilation_ratio * np.hypot(h, w))))
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=d)


def boundary_iou(pred, gt, dilation_ratio: float = BOUNDARY_DILATION_RATIO) -> float:
    """Boundary IoU (Cheng et al., CVPR 2021). An empty prediction scores 0."""
    pred, gt = _check(pred, gt)
    return _iou(mask_boundary(pred, dilation_ratio), mask_boundary(gt, dilation_ratio))
