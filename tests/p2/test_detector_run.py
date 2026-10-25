import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from promptcorrect.data.records import build_coco_records, write_records
from promptcorrect.detectors import run
from test_records import coco_fixture


def test_run_writes_manifest_even_for_zero_detections(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations.json"
    fixture = coco_fixture()
    fixture["images"][0]["file_name"] = "test.png"
    annotations.write_text(json.dumps(fixture))
    records = tmp_path / "objects.jsonl"
    write_records(build_coco_records(fixture), records)
    Image.new("RGB", (160, 128)).save(tmp_path / "test.png")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setattr(run.importlib.metadata, "version", lambda name: "test")

    class Detector:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fixed"))

        def detect(self, image, phrase):
            assert image.size == (160, 128)
            return []

    monkeypatch.setattr(run, "GroundingDinoDetector", Detector)
    output = tmp_path / "detections.jsonl"
    manifest = run.run_grounding_dino(annotations, records, tmp_path, output)
    assert output.read_text() == ""
    assert manifest["n_detections"] == 0
    assert len(manifest["queries"]) == 2
    assert manifest["resolved_revision"] == "fixed"
    assert manifest == json.loads(output.with_suffix(".manifest.json").read_text())
    with pytest.raises(FileExistsError):
        run.run_grounding_dino(annotations, records, tmp_path, output)
