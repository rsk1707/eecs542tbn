"""Sensitivity sweep runner and summaries on the dummy segmenter (wiring, not results)."""

import pandas as pd
import pytest

from promptcorrect.perturb.box import SWEEP_LEVELS
from promptcorrect.perturb.sweep import METHODS, by_box_iou, method_table, run_sweep, sensitivity, summarize
from promptcorrect.segmenters.dummy import DummySegmenter

SMALL = {"shift": (0.1, 0.3), "scale": (0.5, 1.5), "truncate": (0.2, 0.4)}


def test_b0_sweep_rows(objects):
    df = run_sweep(DummySegmenter(objects), objects, run_id="t", log_every=0)
    per_obj = 1 + sum(len(v) for v in SWEEP_LEVELS.values())
    assert len(df) == len(objects) * per_obj and (df.method == "b0_none").all() and (df.step == 0).all()
    clean = df[df.pert_type.isna()]
    assert len(clean) == len(objects) and (clean.prompt_origin == "gt").all() and (clean.box_iou_in == 1).all()
    assert df.boundary_iou.notna().all() and df.iou.between(0, 1).all()
    assert {"delta_cx", "coverage", "contamination", "pert_dir", "pred_iou"} <= set(df.columns)
    t = sensitivity(df)
    trunc = t.loc["truncate"].mean_iou
    assert trunc.is_monotonic_decreasing                      # the dummy clips its mask to the box


def test_same_seed_same_rows(objects):
    a = run_sweep(DummySegmenter(objects), objects, run_id="t", levels=SMALL, log_every=0)
    b = run_sweep(DummySegmenter(objects), objects, run_id="t", levels=SMALL, log_every=0)
    pd.testing.assert_frame_equal(a.drop(columns="latency_ms"), b.drop(columns="latency_ms"))


def test_methods_and_oracle(objects):
    df = run_sweep(DummySegmenter(objects), objects, run_id="t", levels=SMALL, methods=METHODS, steps=3,
                   log_every=0)
    assert set(df.method) == set(METHODS)
    s0 = df[df.step == 0].groupby(["obj_id", "pert_id"]).iou.nunique()
    assert (s0 == 1).all()                                    # every method starts from the same mask
    assert df[df.method == "b0_none"].step.max() == 0 and df[df.method == "b2_multimask"].step.max() == 1
    tab = method_table(df, "perturbed")
    assert list(tab.index) == list(METHODS)
    assert tab.loc["oracle_clicks", "delta_iou"] > 0 and tab.loc["oracle_clicks", "harm_%"] == 0
    assert tab.loc["b0_none", "delta_iou"] == 0
    tables = summarize(df)
    assert {"sensitivity", "iou_by_box_iou", "methods_perturbed", "methods_clean"} <= set(tables)
    assert not by_box_iou(df).empty


def test_unknown_method_rejected(objects):
    with pytest.raises(ValueError):
        run_sweep(DummySegmenter(objects), objects, run_id="t", methods=("magic",))
