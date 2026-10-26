import numpy as np
import pytest

from promptcorrect.eval.matching import match


def _box(x0, y0, x1, y1, shape=(50, 50)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


A, B = _box(0, 0, 10, 10), _box(20, 20, 30, 30)


def test_one_to_one_by_total_iou():
    # pred 0 overlaps both GTs; greedy would give it to A and leave pred 1 unmatched
    p0, p1 = _box(0, 0, 10, 12), _box(0, 0, 10, 10)
    res = match([p0, p1, B], [A, B])
    assert [(p, g) for p, g, _ in res.pairs] == [(1, 0), (2, 1)]
    assert res.pairs[0][2] == 1.0
    assert res.unmatched_preds == [0] and res.unmatched_gts == []


def test_low_iou_pairs_are_dropped():
    far = _box(40, 40, 50, 50)
    res = match([_box(9, 9, 19, 19)], [A], min_iou=0.1)  # IoU 1/199
    assert res.pairs == [] and res.unmatched_preds == [0] and res.unmatched_gts == [0]
    assert match([far], [A]).unmatched_gts == [0]


def test_predictions_on_crowd_regions_are_ignored():
    crowd = _box(30, 0, 50, 20)
    on_crowd, half_on = _box(32, 2, 42, 12), _box(25, 0, 35, 10)  # 100% and 50% inside
    res = match([A, on_crowd, half_on], [A], ignore_masks=[crowd])
    assert res.ignored_preds == [1, 2] and res.unmatched_preds == [] and len(res.pairs) == 1


def test_empty_inputs():
    assert match([], [A]).unmatched_gts == [0]
    assert match([A], []).unmatched_preds == [0]
    with pytest.raises(ValueError, match="same shape"):
        match([A], [_box(0, 0, 5, 5, shape=(40, 40))])
