"""Corrector v1 on CPU: quality head, P4-perturbed samples, calibration and the full eval on the DummySegmenter."""

import numpy as np
import pandas as pd
import pytest
import torch

from promptcorrect.corrector.calibrate import choose_thresholds, run_calibration, threshold_curve
from promptcorrect.corrector.dataset import load_samples, split_by_image
from promptcorrect.corrector.evaluate_v1 import BASELINES, eval_prompts, go_no_go, iou_by_step, run_eval_v1, summary_table
from promptcorrect.corrector.model import CorrectorNet, load_checkpoint, save_checkpoint
from promptcorrect.corrector.policy import CorrectorPolicy
from promptcorrect.corrector.samples import generate_samples
from promptcorrect.corrector.train import TrainConfig, evaluate, quality_metrics, train
from promptcorrect.eval.results import read_results, write_results
from promptcorrect.perturb.training import SeverityMeta, make_prompt_fn
from promptcorrect.segmenters.dummy import DummySegmenter
from promptcorrect.testing import make_synthetic_objects


def _image_loader(objects):
    hw = {o.image_id: tuple(o.gt_mask["size"]) for o in objects}

    def load(image_id):
        rng = np.random.default_rng(image_id)
        return rng.integers(0, 255, size=(*hw[image_id], 3), dtype=np.uint8)
    return load


def test_quality_head_shapes_and_v0_checkpoints_still_load(tmp_path):
    x = torch.randn(3, 9, 32, 32)
    v1 = CorrectorNet(base=8, quality_head=True)
    maps, q = v1.forward_all(x)
    assert maps.shape == (3, 2, 32, 32) and q.shape == (3,)
    assert torch.equal(v1(x), maps)
    v0 = CorrectorNet(base=8)
    assert v0.forward_all(x)[1] is None and "quality_head" not in v0.config
    save_checkpoint(v0, tmp_path / "v0.pt")
    m, _ = load_checkpoint(tmp_path / "v0.pt")
    assert not m.has_quality
    save_checkpoint(v1, tmp_path / "v1.pt")
    assert load_checkpoint(tmp_path / "v1.pt")[0].has_quality


def test_quality_metrics():
    true = torch.tensor([0.2, 0.5, 0.8, 0.95])
    m = quality_metrics(true.clone(), true)
    assert m["q_mae"] == 0 and m["q_pearson"] == pytest.approx(1) and m["q_fail_auroc"] == 1


def test_gate_abstains_on_high_predicted_iou():
    class Fixed(CorrectorNet):
        def forward_all(self, x):
            maps = torch.full((x.shape[0], 2, x.shape[-1], x.shape[-1]), -10.0)
            maps[:, 1, 8:20, 8:20] = 10.0  # a clear FN region -> a positive click
            return maps, torch.full((x.shape[0],), 3.0)  # sigmoid(3) = 0.95
    image = torch.zeros(3, 64, 64, dtype=torch.uint8)
    logits = torch.full((64, 64), -5.0)
    pol = CorrectorPolicy(Fixed(base=8), "cpu", crop_size=32, quality_threshold=0.9)
    v = pol.view(image, logits, [16, 16, 48, 48])
    assert v.q == pytest.approx(0.9526, abs=1e-3) and pol.candidate(v)[2] == 1
    assert pol.decide(v) is None                       # predicted IoU 0.95 >= 0.9: abstain
    pol.quality_threshold = 0.99
    assert pol.decide(v) is not None
    pol.quality_threshold = None                       # no gate = v0 behaviour
    assert pol.decide(v) is not None


def test_threshold_choice_respects_harm():
    # 4 good masks (two get hurt by a click), 4 bad ones (all fixed); q separates them perfectly
    df = pd.DataFrame(dict(iou0=[0.9] * 4 + [0.4] * 4, q=[0.9] * 4 + [0.4] * 4, err_threshold=0.5, feedback=True,
                           has_click=True, iou1=[0.95, 0.95, 0.5, 0.5] + [0.8] * 4))
    curve = threshold_curve(df, True)
    never = curve[curve.tau == 0].iloc[0]
    assert never.clicked_pct == 0 and never.delta_all == 0
    best = choose_thresholds(curve, max_harm=0.05)
    assert best["harm_good_pct"] == 0 and best["clicked_good_pct"] == 0 and best["delta_fail"] == pytest.approx(0.4)
    assert 0.4 < best["tau"] <= 0.9
    always = choose_thresholds(curve, max_harm=1.0)    # harm allowed: clicking everything has lower mean gain
    assert always["delta_all"] >= best["delta_all"] - 1e-9


def test_v1_chain_on_dummy(tmp_path):
    objects = make_synthetic_objects(n_images=4, objects_per_image=2, size=(120, 160), seed=0)
    seg = DummySegmenter(objects)
    load = _image_loader(objects)
    prompt_fn = make_prompt_fn("synthetic", seed=0, n_perturbed=2)
    stats = generate_samples(seg, objects, load, tmp_path / "samples", crop_size=32, shard_size=5, log_every=0,
                             prompt_fn=prompt_fn, extra_meta=SeverityMeta())
    assert stats["n_samples"] == 3 * len(objects)
    ds = load_samples(tmp_path / "samples")
    assert ds.iou is not None and ds.iou.shape == (len(ds),)
    assert torch.allclose(ds.iou, torch.tensor([m["iou"] for m in ds.meta]))
    tr, va = split_by_image(ds, val_frac=0.5)
    assert len(ds.subset(va).iou) == len(va)

    cfg = TrainConfig(epochs=2, batch_size=4, base=8, amp=False, click_eval_samples=10, quality_head=True)
    result = train(ds.subset(tr), ds.subset(va), cfg, tmp_path / "train", "cpu")
    assert "train_quality" in result["history"][0] and "q_mae" in result["history"][0]
    model, _ = load_checkpoint(tmp_path / "train" / "best.pt")
    assert model.has_quality and "q_fail_auroc" in evaluate(model, ds.subset(va), cfg, "cpu")

    # calibration replays held-out samples with their own (perturbed) boxes
    policy = CorrectorPolicy(model, "cpu", crop_size=32)
    held = [ds.meta[i] for i in va]
    calib = run_calibration(seg, policy, held, {o.obj_id: o for o in objects}, load,
                            err_thresholds=(0.0, 0.5), log_every=0)
    assert len(calib) == len(held) * 2 * 2 and calib.q.between(0, 1).all()
    assert calib[calib.err_threshold == 0.0].has_click.all()   # threshold 0 always proposes a click
    chosen = choose_thresholds(threshold_curve(calib, True), max_harm=0.05)
    assert chosen["harm_good_pct"] <= 5

    # eval: all methods share step 0; rows are valid ResultRows
    pols = {"corrector_v1": CorrectorPolicy(model, "cpu", crop_size=32, err_threshold=0.0, quality_threshold=None,
                                            feedback=True)}
    rows = run_eval_v1(seg, objects, load, run_id="p5v1_eval_test", policies=pols, steps=2, n_perturbed=1,
                       log_every=0)
    write_results(rows, tmp_path / "p5v1_eval_test")
    df = read_results(tmp_path / "p5v1_eval_test")
    assert set(df.method) == {"b0_none", "corrector_v1", "oracle_clicks", *BASELINES}
    assert set(df.prompt_origin) == {"gt", "perturbed"} and df.box_iou_in.notna().all()
    s0 = df[df.step == 0].groupby(["obj_id", "prompt_origin"]).iou.nunique()
    assert (s0 == 1).all()
    assert (df[df.method == "corrector_v1"].step == 2).any()
    table = summary_table(df)
    assert ("gt", "all", "corrector_v1") in table.index
    assert table.loc[("all", "all", "b0_none"), "harm_pct"] == 0
    assert table.loc[("all", "all", "corrector_v1"), "clicked_pct"] == 100
    gng = go_no_go(table, "corrector_v1")
    assert {"beats_rules", "harm_ok", "go"} <= set(gng.columns)
    assert iou_by_step(df, 2).shape[1] == 3


def test_eval_prompts_are_independent_of_training_draws(objects):
    rec = objects[0]
    a, b = eval_prompts(rec, seed=0, n_perturbed=2), eval_prompts(rec, seed=0, n_perturbed=2)
    assert a[0].origin == "gt" and [p.box for p in a] == [p.box for p in b]
    train_boxes = [p.box for p in make_prompt_fn("synthetic", seed=0, n_perturbed=2)(rec, objects)]
    assert a[1].box != train_boxes[1]
