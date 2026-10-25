import math

import numpy as np
import pytest

from promptcorrect.data.geometry import mask_to_box, rle_encode
from promptcorrect.perturb.severity import COLUMNS, box_iou, describe, severity_columns
from promptcorrect.types import ObjectRecord, Prompt


def _rec(obj_id: str, mask: np.ndarray) -> ObjectRecord:
    return ObjectRecord(obj_id=obj_id, image_id=1, source="coco", category_id=1, category_name="cat",
                        novelty="seen", gt_box=mask_to_box(mask), gt_mask=rle_encode(mask), area=float(mask.sum()),
                        size_bin="medium", thinness=0.0, n_neighbors=0)


@pytest.fixture
def scene():
    a = np.zeros((100, 200), bool); a[20:60, 20:80] = True      # target: box [20, 20, 80, 60]
    b = np.zeros((100, 200), bool); b[20:60, 80:120] = True     # neighbour right next to it
    return _rec("coco:1", a), _rec("coco:2", b)


def test_gt_box_is_perfect(scene):
    obj, nb = scene
    sev = describe(Prompt(box=obj.gt_box), obj, [obj, nb])
    assert sev["box_iou"] == 1 and sev["coverage"] == 1 and sev["contamination"] == 0
    assert sev["delta"] == pytest.approx((0, 0, 0, 0))


def test_truncation_lowers_coverage(scene):
    obj, nb = scene
    sev = describe(Prompt(box=[50, 20, 80, 60]), obj, [nb])     # left half cut
    assert sev["coverage"] == pytest.approx(0.5) and sev["box_iou"] == pytest.approx(0.5)
    assert sev["delta"][0] == pytest.approx(0.25) and sev["delta"][2] == pytest.approx(math.log(0.5))


def test_shift_into_neighbour_is_contamination(scene):
    obj, nb = scene
    sev = describe(Prompt(box=[50, 20, 110, 60]), obj, [obj, nb])  # half on the target, half on b
    assert sev["contamination"] == pytest.approx(0.5) and sev["coverage"] == pytest.approx(0.5)
    assert describe(Prompt(box=[50, 20, 110, 60]), obj, [])["contamination"] == 0


def test_pixel_centre_convention_and_cache(scene):
    obj, nb = scene
    cache = {}
    sev = describe(Prompt(box=[19.6, 19.6, 80.4, 60.4]), obj, [nb], mask_cache=cache)  # same pixels
    assert sev["coverage"] == 1 and set(cache) == {"coco:1", "coco:2"}


def test_point_only_prompt_gives_nan(scene):
    obj, nb = scene
    sev = describe(Prompt(points=[(30, 30, 1)]), obj, [nb])
    cols = severity_columns(sev)
    assert list(cols) == list(COLUMNS) and all(math.isnan(v) for v in cols.values())


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0
