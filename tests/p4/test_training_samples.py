"""P4 training-data variants through P5's sample generator, on the dummy segmenter (wiring, not results)."""

import json

import numpy as np
import pytest

from promptcorrect.corrector.crops import box_mask, crop_window
from promptcorrect.corrector.dataset import load_samples, split_by_image
from promptcorrect.corrector.samples import generate_samples
from promptcorrect.perturb.training import SeverityMeta, make_prompt_fn, synthetic_prompts
from promptcorrect.segmenters.dummy import DummySegmenter
from promptcorrect.testing import make_synthetic_objects


def _loader(objects):
    hw = {o.image_id: tuple(o.gt_mask["size"]) for o in objects}
    return lambda image_id: np.random.default_rng(image_id).integers(0, 255, size=(*hw[image_id], 3), dtype=np.uint8)


@pytest.fixture
def small():
    return make_synthetic_objects(n_images=3, objects_per_image=2, size=(120, 160), seed=0)


def test_synthetic_prompts_are_per_object_deterministic(small):
    rec = small[0]
    a, b = synthetic_prompts(rec, seed=0, n_perturbed=3), synthetic_prompts(rec, seed=0, n_perturbed=3)
    assert [p.box for p in a] == [p.box for p in b] and len(a) == 4
    assert a[0].origin == "gt" and a[0].perturbation is None and a[0].box == list(rec.gt_box)
    assert all(p.origin == "perturbed" and p.perturbation["type"] in ("shift", "scale", "truncate") for p in a[1:])
    assert [p.box for p in synthetic_prompts(rec, seed=1, n_perturbed=3)] != [p.box for p in a]
    assert len(synthetic_prompts(rec, n_perturbed=2, include_clean=False)) == 2
    only_shift = synthetic_prompts(rec, n_perturbed=5, types=("shift",))
    assert {p.perturbation["type"] for p in only_shift[1:]} == {"shift"}


def test_other_variants_wait_for_mined_failures():
    with pytest.raises(NotImplementedError):
        make_prompt_fn("real")
    with pytest.raises(ValueError):
        make_prompt_fn("nonsense")


def test_samples_follow_the_prompt_box(tmp_path, small):
    fn = make_prompt_fn("synthetic", seed=0, n_perturbed=2)
    stats = generate_samples(DummySegmenter(small), small, _loader(small), tmp_path / "s", crop_size=32,
                             shard_size=4, log_every=0, prompt_fn=fn, extra_meta=SeverityMeta())
    assert stats["n_samples"] == 3 * len(small)
    ds = load_samples(tmp_path / "s")
    assert len(ds) == 3 * len(small)

    by_obj = {}
    for i, m in enumerate(ds.meta):
        by_obj.setdefault(m["obj_id"], []).append((i, m))
    for obj_id, rows in by_obj.items():
        assert [m["prompt_id"] for _, m in rows] == [0, 1, 2]
        clean = rows[0][1]
        assert clean["origin"] == "gt" and clean["pert_type"] is None and clean["box_iou_in"] == 1.0
        for i, m in rows[1:]:
            assert m["origin"] == "perturbed" and m["pert_type"] in ("shift", "scale", "truncate")
            assert 0 <= m["coverage"] <= 1 and m["box_iou_in"] < 1 and "delta_cx" in m
            # the crop is built around the PROMPT box, never the GT box
            assert m["window"] == pytest.approx(crop_window(m["box"], 0.5))
            assert (ds.box[i].numpy() == box_mask(m["box"], m["window"], 32)).all()
    # meta.jsonl is strict JSON (no NaN) for box prompts
    for line in (tmp_path / "s" / "meta.jsonl").read_text().splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    # all prompts of an image stay on the same side of P5's holdout split
    tr, va = split_by_image(ds, val_frac=0.5)
    assert not ({ds.meta[i]["obj_id"] for i in tr} & {ds.meta[i]["obj_id"] for i in va})


def test_default_is_still_v0(tmp_path, small):
    stats = generate_samples(DummySegmenter(small), small, _loader(small), tmp_path / "v0", crop_size=32,
                             log_every=0)
    ds = load_samples(tmp_path / "v0")
    assert stats["n_samples"] == len(small) and stats["prompts"] == "clean GT box only"
    assert all(m["origin"] == "gt" and m["prompt_id"] == 0 and m["box"] == list(o.gt_box)
               for m, o in zip(ds.meta, small))
