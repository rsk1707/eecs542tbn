import json

import pytest
import yaml

from promptcorrect.data.geometry import rle_encode
from promptcorrect.data.prepare import rooted_path
from promptcorrect.detectors.export import export_detections
from promptcorrect.detectors.sam3_text import RecordingSegmenter, is_valid_detection
from promptcorrect.segmenters.dummy import DummySegmenter


class SegmenterWithBadBoxes(DummySegmenter):
    """Adds an edge-clipped zero-width detection and an out-of-range score to every query."""

    last_presence = 0.93

    def detect_text(self, phrase):
        dets = super().detect_text(phrase)
        if dets:
            rle = dets[0][1]
            dets += [([5.0, 5.0, 5.0, 9.0], rle, 0.8), ([5.0, 5.0, 9.0, 9.0], rle, 1.5)]
        return dets


def test_is_valid_detection():
    assert is_valid_detection([1, 1, 5, 5], 0.5)
    assert not is_valid_detection([5, 1, 5, 5], 0.5)          # zero width
    assert not is_valid_detection([1, 1, 5, 5], 1.2)          # score out of range
    assert not is_valid_detection([1, 1, float("nan"), 5], 0.5)


def test_export_survives_invalid_detections_and_records_queries(tmp_path, objects):
    seg = RecordingSegmenter(SegmenterWithBadBoxes(objects))
    out = tmp_path / "val.jsonl"
    count = export_detections(objects, out, segmenter=seg)  # would raise without the filter
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert count == len(rows) > 0 and all(r["box"][2] > r["box"][0] and "mask_rle" in r for r in rows)
    assert seg.queries and all(q["presence"] == 0.93 for q in seg.queries)
    assert sum(q["n_dropped_invalid"] for q in seg.queries) == 2 * len([q for q in seg.queries if q["n_detections"]])
    assert {q["image_id"] for q in seg.queries} == {o.image_id for o in objects}


def test_recording_segmenter_without_presence(objects):
    seg = RecordingSegmenter(DummySegmenter(objects))
    seg.set_image(1)
    seg.detect_text("person")
    assert seg.queries[0]["presence"] is None and seg.queries[0]["image_id"] == 1


def test_rooted_path_accepts_both_key_styles(tmp_path):
    upper = tmp_path / "upper.yaml"
    upper.write_text("DATA_ROOT: /abs/data\nCACHE_ROOT: /abs/cache\n")
    lower = tmp_path / "lower.yaml"
    lower.write_text(yaml.safe_dump({"data_root": "/abs/data", "cache_root": "/abs/cache"}))
    for f in (upper, lower):
        assert rooted_path(f, "data_root", "x.jsonl").as_posix().endswith("/abs/data/x.jsonl")
        assert rooted_path(f, "CACHE_ROOT", "d/y.jsonl").as_posix().endswith("/abs/cache/d/y.jsonl")
    with pytest.raises(KeyError, match="runs_root"):
        rooted_path(lower, "runs_root", "z")
