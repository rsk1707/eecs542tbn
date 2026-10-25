import json

import numpy as np
import pytest

from promptcorrect.data.records import (
    annotation_mask, build_coco_records, decode_rle, encode_rle,
    mask_thinness, read_records, write_records, xywh_to_xyxy,
)


def coco_fixture():
    def ann(ident, box, *, category=1, crowd=0, area=None):
        x, y, w, h = box
        return {"id": ident, "image_id": 10, "category_id": category,
                "bbox": box, "area": w * h if area is None else area,
                "iscrowd": crowd,
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]}

    return {"images": [{"id": 10, "width": 160, "height": 128}],
            "categories": [{"id": 1, "name": "test_object"}, {"id": 2, "name": "other"}],
            "annotations": [ann(1, [0, 0, 32, 32]), ann(2, [32, 0, 32, 32]),
                            ann(3, [0, 64, 8, 40]), ann(4, [80, 0, 31, 31]),
                            ann(5, [0, 0, 40, 40], crowd=1),
                            ann(6, [0, 0, 32, 32], category=2),
                            ann(7, [64, 32, 96, 96])]}


def test_records_filter_geometry_and_round_trip(tmp_path):
    records = list(build_coco_records(coco_fixture()))
    assert [r.obj_id for r in records] == ["coco:1", "coco:2", "coco:3", "coco:6", "coco:7"]
    first, second, thin, other, large = records
    assert first.gt_box == [0., 0., 32., 32.]
    assert first.n_neighbors == second.n_neighbors == 1
    assert other.n_neighbors == 0
    assert first.category_name == "test object"
    assert first.novelty == "seen"
    assert first.size_bin == "medium" and thin.size_bin == "small"
    assert large.size_bin == "large"
    assert first.thinness == 16
    assert thin.thinness == 96**2 / 320
    assert decode_rle(first.gt_mask).sum() == 1024
    assert isinstance(first.gt_mask["counts"], str)
    path = tmp_path / "objects.jsonl"
    write_records(iter(records), path)
    assert list(read_records(path)) == records
    with pytest.raises(FileExistsError):
        write_records(records, path)


def test_polygon_and_compressed_and_uncompressed_rle():
    expected = np.zeros((4, 5), dtype=bool)
    expected[1:3, 1:4] = True
    polygon = [[1, 1, 4, 1, 4, 3, 1, 3]]
    np.testing.assert_array_equal(annotation_mask(polygon, 4, 5), expected)
    rle = encode_rle(expected)
    np.testing.assert_array_equal(annotation_mask(rle, 4, 5), expected)
    rle["counts"] = rle["counts"].encode("ascii")
    np.testing.assert_array_equal(decode_rle(rle), expected)
    flat = expected.flatten(order="F")
    runs, previous, length = [], False, 0
    for pixel in flat:
        if pixel != previous:
            runs.append(length)
            length, previous = 0, pixel
        length += 1
    runs.append(length)
    np.testing.assert_array_equal(decode_rle({"size": [4, 5], "counts": runs}), expected)
    multipart = [[0, 0, 1, 0, 1, 1, 0, 1], [4, 3, 5, 3, 5, 4, 4, 4]]
    assert annotation_mask(multipart, 4, 5).sum() == 2
    with pytest.raises(ValueError, match="dimensions"):
        annotation_mask(rle, 5, 4)


def test_thinness_counts_holes_and_image_edges():
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    assert mask_thinness(mask) == 16**2 / 8
    with pytest.raises(ValueError):
        mask_thinness(np.zeros((3, 3), dtype=bool))


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, -1, 2], [0, 0, float("nan"), 1]])
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        xywh_to_xyxy(box)


def test_subset_and_empty_mask_fail_loudly():
    assert list(build_coco_records(coco_fixture(), [])) == []
    with pytest.raises(ValueError, match="absent"):
        list(build_coco_records(coco_fixture(), [999]))
    annotations = coco_fixture()
    annotations["annotations"][0]["segmentation"] = []
    with pytest.raises(ValueError, match="coco:1"):
        list(build_coco_records(annotations))


def test_record_write_failure_is_atomic(tmp_path):
    def broken():
        yield next(build_coco_records(coco_fixture()))
        raise RuntimeError("interrupted")

    path = tmp_path / "objects.jsonl"
    with pytest.raises(RuntimeError):
        write_records(broken(), path)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
