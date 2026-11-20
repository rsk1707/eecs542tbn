from copy import deepcopy
import json
import random
import subprocess
import sys

import pytest

from promptcorrect.data.prepare import file_sha256
from promptcorrect.data.prepare_rq3 import prepare_rq3, write_category_review
from promptcorrect.data.records import read_records
from test_lvis import lvis_fixture, reviewed_fixture
from test_records import coco_fixture


def inputs(tmp_path):
    val = coco_fixture()
    train = deepcopy(val)
    train["images"] = [{**val["images"][0], "id": i} for i in [20, 30, 40]]
    train["annotations"] = [{**a, "image_id": i, "id": a["id"] + i * 10}
                            for i in [20, 30, 40] for a in val["annotations"]]
    lvis = lvis_fixture()
    # Image 20 leaks even though it has NO retained LVIS objects.
    lvis["images"].append({**lvis["images"][0], "id": 20})
    paths = []
    for name, data in [("train", train), ("val", val), ("lvis", lvis)]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        paths.append(path)
    review = reviewed_fixture()
    review["sources_sha256"] = {"coco": file_sha256(paths[1]), "lvis": file_sha256(paths[2])}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    return [*paths, path]


def test_preparation_excludes_empty_lvis_images_and_hashes_artifacts(tmp_path):
    paths = inputs(tmp_path)
    out = tmp_path / "out"
    manifest = prepare_rq3(*paths, out)
    split = json.loads((out / "splits.json").read_text())
    assert split == {"train": [30, 40], "val": [10], "lvis_val_excluded_from_train": [20]}
    train = list(read_records(out / "objects_train.jsonl"))
    assert {r.image_id for r in train} == {30, 40}
    assert len(train) == 10
    lvis = list(read_records(out / "objects_lvis_val.jsonl"))
    assert len(lvis) == 5 and {r.novelty for r in lvis} == {"seen", "unseen"}
    inventory = manifest["inventory"]["objects_lvis_val.jsonl"]
    assert inventory["n_selected_images"] == 2 and inventory["n_images_with_objects"] == 1
    assert manifest == json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["artifacts_sha256"].items():
        assert file_sha256(out / name) == digest
    with pytest.raises(FileExistsError):
        prepare_rq3(*paths, out)


def test_train_sampling_after_exclusion_is_reproducible(tmp_path):
    paths = inputs(tmp_path)
    state = random.getstate()
    for name in ("one", "two"):
        prepare_rq3(*paths, tmp_path / name, train_count=1, seed=4)
    assert random.getstate() == state
    assert (tmp_path / "one/splits.json").read_bytes() == (tmp_path / "two/splits.json").read_bytes()
    assert len(list(read_records(tmp_path / "one/objects_train.jsonl"))) == 5
    for count in (0, 3):
        with pytest.raises(ValueError, match="train_count"):
            prepare_rq3(*paths, tmp_path / "invalid", train_count=count)
        assert not (tmp_path / "invalid").exists()


def test_draft_review_and_cli_reject_unreviewed_categories(tmp_path):
    paths = inputs(tmp_path)
    synsets = tmp_path / "synsets.json"
    synsets.write_text(json.dumps({
        "test_object": {"coco_cat_id": 1, "synset": "test.n.01"},
        "other": {"coco_cat_id": 2, "synset": "other.n.01"},
    }))
    draft = tmp_path / "draft.json"
    summary = write_category_review(paths[1], paths[2], synsets, draft)
    assert summary == {"n_categories": 2, "n_exact_synset_candidates": 1, "review_complete": False}
    with pytest.raises(ValueError, match="reviewed"):
        prepare_rq3(*paths[:3], draft, tmp_path / "unreviewed")
    assert not (tmp_path / "unreviewed").exists()
    config = tmp_path / "paths.yaml"
    config.write_text(f"data_root: {tmp_path}\n")
    result = subprocess.run([
        sys.executable, "scripts/prepare_p2_rq3.py", "--paths", str(config), "prepare",
        "--coco-train", "train.json", "--coco-val", "val.json", "--lvis-val", "lvis.json",
        "--category-review", "review.json", "--output-dir", "cli", "--train-count", "1",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["inventory"]["objects_train.jsonl"]["n_objects"] == 5


def test_stale_annotations_fail_before_output_creation(tmp_path):
    paths = inputs(tmp_path)
    paths[2].write_text(paths[2].read_text() + "\n")
    with pytest.raises(ValueError, match="different"):
        prepare_rq3(*paths, tmp_path / "stale")
    assert not (tmp_path / "stale").exists()


def test_interruption_never_writes_completion_manifest(tmp_path, monkeypatch):
    import promptcorrect.data.prepare_rq3 as module

    paths = inputs(tmp_path)

    def broken(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "write_records", broken)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_rq3(*paths, tmp_path / "partial")
    assert not (tmp_path / "partial/manifest.json").exists()
