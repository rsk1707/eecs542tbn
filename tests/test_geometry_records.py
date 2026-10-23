import json

import numpy as np

from promptcorrect.data.geometry import mask_to_box, rle_decode, rle_encode, xywh_to_xyxy
from promptcorrect.data.records import group_by_image, load_objects, save_objects


def test_xywh_to_xyxy():
    assert xywh_to_xyxy([10, 20, 30, 40]) == [10.0, 20.0, 40.0, 60.0]


def test_rle_roundtrip_counts_is_str():
    m = np.zeros((50, 60), dtype=bool)
    m[10:30, 5:25] = True
    rle = rle_encode(m)
    assert isinstance(rle["counts"], str) and rle["size"] == [50, 60]
    json.dumps(rle)  # JSON-safe
    assert np.array_equal(rle_decode(rle), m)


def test_mask_to_box():
    m = np.zeros((50, 60), dtype=bool)
    m[10:30, 5:25] = True
    assert mask_to_box(m) == [5.0, 10.0, 25.0, 30.0]


def test_objects_jsonl_roundtrip(tmp_path, objects):
    path = tmp_path / "objects_val.jsonl"
    assert save_objects(path, objects) == len(objects)
    loaded = load_objects(path)
    assert loaded == objects
    assert set(group_by_image(loaded)) == {1, 2, 3}
