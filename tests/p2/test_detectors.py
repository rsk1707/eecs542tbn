from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from promptcorrect.data.records import build_coco_records, encode_rle
from promptcorrect.detectors.export import export_detections
from promptcorrect.detectors.grounding_dino import GroundingDinoDetector
from test_records import coco_fixture


def test_export_independent_detector_queries_once_per_gt_phrase(tmp_path):
    records = list(build_coco_records(coco_fixture()))
    calls = []

    class Detector:
        def detect(self, image, phrase):
            calls.append(phrase)
            return [] if phrase == "other" else [([1, 2, 20, 30], 0.9)]

    output = tmp_path / "dino.jsonl"
    count = export_detections(records, output, detector=Detector(),
                              load_image=lambda ident: Image.new("RGB", (160, 128)))
    assert calls == ["other", "test object"]
    assert count == 1
    assert json.loads(output.read_text()) == {
        "image_id": 10, "phrase": "test object", "box": [1, 2, 20, 30], "score": 0.9}


def test_export_through_segmenter_contract_and_json_safe_rle(tmp_path):
    class FakeSegmenter:
        def set_image(self, image_id):
            assert image_id == 10

        def detect_text(self, phrase):
            rle = encode_rle(np.ones((128, 160), dtype=bool))
            rle["counts"] = rle["counts"].encode("ascii")
            return [([0, 0, 160, 128], rle, 0.8)]

    output = tmp_path / "sam.jsonl"
    assert export_detections(build_coco_records(coco_fixture()), output,
                             segmenter=FakeSegmenter()) == 2
    assert all(isinstance(json.loads(line)["mask_rle"]["counts"], str)
               for line in output.read_text().splitlines())


def test_failed_export_is_not_published(tmp_path):
    output = tmp_path / "invalid.jsonl"
    detector = SimpleNamespace(detect=lambda image, phrase: [([2, 0, 1, 9], 0.8)])
    with pytest.raises(ValueError, match="Invalid detection"):
        export_detections(build_coco_records(coco_fixture()), output,
                          detector=detector, load_image=lambda ident: None)
    assert not output.exists()
    with pytest.raises(ValueError, match="exactly one"):
        export_detections([], output)


def test_grounding_dino_freezes_model_and_uses_original_dimensions(monkeypatch):
    """Exercise adapter calls without torch, checkpoints, or network access."""
    calls = {}

    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class Inputs(dict):
        input_ids = "input_ids"

        def to(self, device):
            calls["input_device"] = device
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, ident, revision):
            calls["processor_revision"] = revision
            return cls()

        def __call__(self, **kwargs):
            calls["text"] = kwargs["text"]
            assert kwargs["images"].mode == "RGB"
            return Inputs()

        def post_process_grounded_object_detection(self, outputs, input_ids, **kwargs):
            calls["postprocess"] = kwargs
            return [{"boxes": Tensor([[1, 2, 30, 40]]), "scores": Tensor([0.8])}]

    class Model:
        @classmethod
        def from_pretrained(cls, ident, revision):
            return cls()

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def requires_grad_(self, enabled):
            calls["requires_grad"] = enabled

        def __call__(self, **kwargs):
            return "outputs"

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=Processor, AutoModelForZeroShotObjectDetection=Model))
    detector = GroundingDinoDetector(device="cpu", revision="fixed-revision")
    assert detector.detect(Image.new("L", (200, 100)), "test_object") == [([1, 2, 30, 40], 0.8)]
    assert calls["text"] == "test object."
    assert calls["postprocess"]["target_sizes"] == [(100, 200)]
    assert calls["postprocess"]["threshold"] == 0.35
    assert calls["eval"] and calls["requires_grad"] is False
    assert calls["processor_revision"] == "fixed-revision"
    with pytest.raises(ValueError):
        detector.detect(Image.new("RGB", (5, 5)), " ")
