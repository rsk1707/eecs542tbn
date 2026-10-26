import json

import pytest

from promptcorrect.data.prepare import prepare_demo, rooted_path
from promptcorrect.data.splits import demo_split, training_split, write_split
from test_records import coco_fixture


def test_demo_reproducibility_and_no_rng_side_effects():
    import random

    state = random.getstate()
    split = demo_split(range(1000))
    assert split == demo_split(reversed(range(1000)))
    assert len(split["val"]) == len(set(split["val"])) == 500
    assert split != demo_split(range(1000), seed=1)
    assert random.getstate() == state
    with pytest.raises(ValueError):
        demo_split([1, 2], count=500)


def test_training_split_excludes_lvis_at_image_level():
    split = training_split([1, 2, 3], [4, 5], [2, 3, 5, 6])
    assert split == {"train": [1], "val": [4, 5], "lvis_val_excluded_from_train": [2, 3]}
    assert not set(split["train"]) & {2, 3, 5, 6}
    with pytest.raises(ValueError, match="overlap"):
        training_split([1, 2], [2], [])


def test_split_cannot_be_overwritten(tmp_path):
    path = tmp_path / "split.json"
    split = demo_split([1], count=1)
    write_split(split, path)
    with pytest.raises(FileExistsError):
        write_split(demo_split([2], count=1), path)
    assert json.loads(path.read_text()) == split


def test_preparation_provenance_and_fixed_split_reuse(tmp_path):
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps(coco_fixture()))
    split, objects = tmp_path / "split.json", tmp_path / "objects.jsonl"
    manifest = prepare_demo(annotation, split, objects, count=1)
    assert manifest["n_images"] == 1 and manifest["n_objects"] == 5
    assert len(manifest["annotations_sha256"]) == 64
    assert manifest == json.loads(objects.with_suffix(".manifest.json").read_text())
    with pytest.raises(FileExistsError):
        prepare_demo(annotation, split, objects, count=1)
    prepare_demo(annotation, split, tmp_path / "second.jsonl", count=1)
    split.write_text(json.dumps(demo_split([99], count=1)))
    with pytest.raises(ValueError, match="differs"):
        prepare_demo(annotation, split, tmp_path / "third.jsonl", count=1)


def test_rooted_paths(tmp_path):
    paths = tmp_path / "paths.yaml"
    paths.write_text("DATA_ROOT: datasets\nCACHE_ROOT: /tmp/cache\n")
    assert rooted_path(paths, "DATA_ROOT", "objects.jsonl") == tmp_path / "datasets/objects.jsonl"
    assert rooted_path(paths, "CACHE_ROOT", "detections/a.jsonl").as_posix() == "/tmp/cache/detections/a.jsonl"


def test_new_files_get_normal_permissions(tmp_path):
    """mkstemp's 0600 would lock teammates out of the ACL-shared scratch dir."""
    import os

    from promptcorrect.data.io import new_text_file

    old = os.umask(0o002)
    try:
        with new_text_file(tmp_path / "out.jsonl") as f:
            f.write("{}\n")
    finally:
        os.umask(old)
    assert (tmp_path / "out.jsonl").stat().st_mode & 0o777 == 0o664
