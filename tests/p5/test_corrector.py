"""Corrector v0 on CPU: crop geometry, model/loss, click rule, and the whole chain on the DummySegmenter."""

import numpy as np
import pytest
import torch

from promptcorrect.corrector.clicks import click_hits, propose_click
from promptcorrect.corrector.crops import (box_mask, click_heatmaps, clicks_to_crop, crop_object, crop_to_image,
                                           crop_window, image_to_crop)
from promptcorrect.corrector.dataset import load_samples, split_by_image
from promptcorrect.corrector.figures import gain_and_harm, iou_by_step, training_curves
from promptcorrect.corrector.losses import error_map_loss
from promptcorrect.corrector.model import IN_CHANNELS, CorrectorNet, build_inputs, error_targets, load_checkpoint
from promptcorrect.corrector.rollout import run_rollouts, summarize
from promptcorrect.corrector.samples import generate_samples
from promptcorrect.corrector.standins import PERTURBATIONS, box_iou, perturb_box
from promptcorrect.corrector.train import TrainConfig, auroc, evaluate, train
from promptcorrect.data.geometry import rle_decode
from promptcorrect.segmenters.dummy import DummySegmenter
from promptcorrect.testing import make_synthetic_objects


def _image_loader(objects):
    hw = {o.image_id: tuple(o.gt_mask["size"]) for o in objects}

    def load(image_id):
        rng = np.random.default_rng(image_id)
        return rng.integers(0, 255, size=(*hw[image_id], 3), dtype=np.uint8)
    return load


def test_crop_coordinates_round_trip():
    win = crop_window([10, 20, 50, 100], context=0.5)
    assert win == [-10, -20, 70, 140]
    u, v = image_to_crop(33.0, 47.0, win, 64)
    assert np.allclose(crop_to_image(u, v, win, 64), (33.0, 47.0))
    m = box_mask([10, 20, 50, 100], win, 64)
    assert m[32, 32] and not m[0, 0] and m.sum() == pytest.approx(32 * 32, abs=64)


def test_crop_object_matches_gt_and_marks_outside(objects):
    o = objects[0]
    gt = rle_decode(o.gt_mask)
    H, W = gt.shape
    image = torch.zeros(3, H, W, dtype=torch.uint8)
    logits = torch.where(torch.from_numpy(gt), 5.0, -5.0)
    win = crop_window(o.gt_box, 0.5)
    c = crop_object(image, logits, gt, win, 64)
    assert ((c["logits"] > 0) == c["gt"]).mean() > 0.97
    whole = crop_object(image, logits, gt, [-W, -H, 2 * W, 2 * H], 30)
    assert not whole["valid"][0, 0] and whole["valid"][15, 15] and whole["logits"][0, 0] < -10


def test_click_heatmaps_peak_and_padding():
    clicks = torch.tensor(clicks_to_crop([(10.0, 20.0, 1), (30.0, 5.0, 0)], [0, 0, 64, 64], 64, 4))[None]
    h = click_heatmaps(clicks, 64)
    assert h.shape == (1, 2, 64, 64)
    assert h[0, 0, 19, 9] > 0.9 and h[0, 1, 4, 29] > 0.9 and h[0, 0, 4, 29] < 1e-3
    assert click_heatmaps(torch.full((1, 4, 3), -1.0), 16).abs().sum() == 0


def test_perturbations_reduce_box_iou():
    rng = np.random.default_rng(0)
    box = [100.0, 100.0, 200.0, 180.0]
    for kind in PERTURBATIONS:
        ious = [np.mean([box_iou(perturb_box(box, kind, lv, rng, (480, 640))[0], box) for _ in range(20)])
                for lv in (0.1, 0.3, 0.5)]
        assert ious[0] > ious[1] > ious[2], (kind, ious)


def test_model_overfits_one_batch():
    torch.manual_seed(0)
    S, N = 32, 4
    rgb = torch.randint(0, 255, (N, 3, S, S), dtype=torch.uint8)
    logits = torch.randn(N, S, S) * 4
    gt = torch.zeros(N, S, S, dtype=torch.bool)
    gt[:, 8:24, 8:24] = True
    box = torch.zeros(N, S, S, dtype=torch.bool)
    valid = torch.ones(N, S, S, dtype=torch.bool)
    x = build_inputs(rgb, logits, box, valid, torch.full((N, 2, 3), -1.0))
    y = error_targets(logits, gt)
    assert x.shape == (N, IN_CHANNELS, S, S) and y.shape == (N, 2, S, S)
    model = CorrectorNet(base=8, depth=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        loss, _ = error_map_loss(model(x), y, valid)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.25 * first


def test_propose_click_picks_larger_region_and_abstains():
    probs = np.zeros((2, 32, 32), dtype=np.float32)
    assert propose_click(probs) is None
    probs[1, 4:20, 4:20] = 0.9   # FN block -> positive click in its middle
    probs[0, 25:28, 25:28] = 0.9  # smaller FP block
    u, v, label = propose_click(probs)
    assert label == 1 and 10 <= u <= 14 and 10 <= v <= 14
    fn = probs[1] > 0.5
    assert click_hits((u, v, label), np.zeros_like(fn), fn) is True
    assert click_hits(None, fn, fn) is None


def test_auroc():
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.75)


def test_full_chain_on_dummy(tmp_path):
    objects = make_synthetic_objects(n_images=4, objects_per_image=2, size=(120, 160), seed=0)
    seg = DummySegmenter(objects)
    load = _image_loader(objects)
    stats = generate_samples(seg, objects, load, tmp_path / "samples", n_perturbed=2, click_state_prob=1.0,
                             crop_size=32, shard_size=10, log_every=0)
    assert stats["n_samples"] == stats["states"]["step0"] + stats["states"]["clicks"]
    assert stats["states"]["step0"] == len(objects) * 3 and stats["n_shards"] >= 2

    ds = load_samples(tmp_path / "samples")
    assert len(ds) == stats["n_samples"] and ds.rgb.shape[-1] == 32
    assert (ds.clicks[:, :, 2] >= 0).any()
    tr, va = split_by_image(ds, val_frac=0.5)
    assert not ({ds.meta[i]["image_id"] for i in tr} & {ds.meta[i]["image_id"] for i in va})

    cfg = TrainConfig(epochs=2, batch_size=8, base=8, amp=False, click_eval_samples=10)
    result = train(ds.subset(tr), ds.subset(va), cfg, tmp_path / "train", "cpu")
    assert len(result["history"]) == 2 and (tmp_path / "train" / "best.pt").exists()
    model, _ = load_checkpoint(tmp_path / "train" / "best.pt")
    metrics = evaluate(model, ds.subset(va), cfg, "cpu")
    assert {"fp_auroc", "fn_auroc", "click_hit_rate", "abstain_rate"} <= metrics.keys()

    df = run_rollouts(seg, model, objects, load, run_id="test", steps=2, crop_size=32, threshold=0.0,
                      log_every=0)
    assert set(df.method) == {"b0_none", "corrector_v0", "b1_random_k", "b3_box_refit", "oracle_clicks"}
    assert df.groupby(["obj_id", "prompt_id", "method", "step"]).size().max() == 1
    assert df.step.max() <= 2 and df.prompt_id.nunique() >= 4
    oracle = df[df.method == "oracle_clicks"].groupby(["obj_id", "prompt_id"]).iou
    assert (oracle.max() >= oracle.first() - 1e-9).all()

    tables = summarize(df, steps=2)
    assert "final_clean" in tables and "corrector_v0" in tables["final_perturbed"].index
    for fig in (training_curves(result["history"], tmp_path / "c.png"), iou_by_step(tables, tmp_path / "s.png"),
                gain_and_harm(tables, tmp_path / "g.png")):
        assert fig.stat().st_size > 0
