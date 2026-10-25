import numpy as np
import pytest

from promptcorrect.eval.metrics import boundary_iou, iou, mask_boundary


def square(r0, c0, size, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[r0:r0 + size, c0:c0 + size] = True
    return m


GT = square(10, 10, 20)                # rows/cols 10..29
SHIFTED = np.roll(GT, 3, axis=1)       # cols 13..32


def test_iou_identical_and_disjoint():
    assert iou(GT, GT) == 1.0
    assert iou(square(60, 60, 20), GT) == 0.0


def test_iou_hand_computed():
    # overlap 20x17 = 340, union 400 + 400 - 340 = 460
    assert iou(SHIFTED, GT) == pytest.approx(340 / 460)


def test_empty_prediction_scores_zero():
    empty = np.zeros_like(GT)
    assert iou(empty, GT) == 0.0
    assert boundary_iou(empty, GT) == 0.0


def test_boundary_width():
    # 100x100 image: d = round(0.02 * 141.4) = 3, so the boundary is a 3-px ring: 400 - 14*14
    assert mask_boundary(GT).sum() == 400 - 196


def test_boundary_iou_hand_computed():
    # Overlap region rows 10..29, cols 13..29 = 340 px. Inside it, the union of the two
    # eroded interiors is rows 13..26, cols 13..29 = 238 px, so the shared ring is 102 px.
    # Each ring is 204 px -> union 306 -> 1/3. Mask IoU is 0.74 for the same pair.
    assert boundary_iou(SHIFTED, GT) == pytest.approx(1 / 3)
    assert boundary_iou(GT, GT) == 1.0


def test_boundary_iou_counts_edges_of_interior_holes():
    # A hole deep inside the prediction changes mask IoU but adds its own ring, so
    # boundary IoU must drop too. This guards against computing it on the outer contour only.
    holed = GT.copy()
    holed[18:22, 18:22] = False
    assert boundary_iou(holed, GT) < 1.0


def test_rejects_non_bool_masks():
    logits = GT.astype(np.float32) * 5 - 2
    with pytest.raises(TypeError):
        iou(logits, GT)


def test_rejects_shape_mismatch_and_empty_gt():
    with pytest.raises(ValueError):
        iou(GT, GT[:50])
    with pytest.raises(ValueError):
        iou(GT, np.zeros_like(GT))
