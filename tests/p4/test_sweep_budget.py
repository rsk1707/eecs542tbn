"""Timing safeguards of the sensitivity sweep: time budget, progress, phase timing, sparse-bin masking."""

import time

import numpy as np
import pytest

from promptcorrect.perturb.sweep import MIN_BIN_N, by_box_iou, run_sweep, summarize
from promptcorrect.segmenters.dummy import DummySegmenter


class SlowSegmenter(DummySegmenter):
    """Each image takes a visible amount of time, so a small budget stops the sweep after the first image."""

    def set_image(self, image_id):
        time.sleep(0.15)
        super().set_image(image_id)


def test_time_budget_keeps_finished_images(objects):
    df = run_sweep(SlowSegmenter(objects), objects, run_id="t", log_every=0, time_budget_s=0.1)
    assert df.attrs["stopped_early"] is True
    assert df.attrs["images_done"] == 1 and df.attrs["images_total"] == 3
    assert df.image_id.nunique() == 1 and len(df) == 2 * 18          # two objects x 18 prompts of the first image


def test_no_budget_runs_everything_and_reports_timing(objects, capsys):
    df = run_sweep(DummySegmenter(objects), objects, run_id="t", log_every=1)
    assert df.attrs["stopped_early"] is False and df.attrs["images_done"] == 3
    assert set(df.attrs["timing"]) == {"encode", "predict", "to_full_res", "metrics", "rollout"}
    out = capsys.readouterr().out
    assert "[sweep] 3/3 images" in out and "ETA" in out


def test_zero_budget_returns_an_empty_frame_without_crashing(objects):
    df = run_sweep(DummySegmenter(objects), objects, run_id="t", log_every=0, time_budget_s=0.0)
    assert df.empty and df.attrs["stopped_early"] is True and df.attrs["images_done"] == 0


def test_boundary_can_be_skipped(objects):
    df = run_sweep(DummySegmenter(objects), objects, run_id="t", log_every=0, boundary=False)
    assert df.boundary_iou.isna().all()
    df2 = run_sweep(DummySegmenter(objects), objects, run_id="t", log_every=0)
    assert df2[df2.step == 0].boundary_iou.notna().all()
    assert np.allclose(df.iou, df2.iou)                             # skipping it changes nothing else


def test_sparse_bins_are_blanked(objects):
    import pandas as pd

    from promptcorrect.perturb.sweep import BOX_IOU_BINS, _step0

    df = run_sweep(DummySegmenter(objects), objects, run_id="t", log_every=0)
    loose, strict = by_box_iou(df, min_n=1), by_box_iou(df, min_n=10_000)
    assert loose.notna().any().any() and strict.isna().all().all()

    g = _step0(df)
    g["bin"] = pd.cut(g.box_iou_in, BOX_IOU_BINS, right=False)
    counts = g.pivot_table(index="bin", columns="pert_type", values="iou", aggfunc="count", observed=False)
    table = summarize(df)["iou_by_box_iou"]

    def shown_counts(tab):
        vals = counts.reindex_like(tab).where(tab.notna()).to_numpy(dtype=float).ravel()
        return vals[~np.isnan(vals)]

    assert len(shown_counts(table)) > 0 and (shown_counts(table) >= MIN_BIN_N).all()   # every shown cell has enough prompts
    assert len(shown_counts(loose)) >= len(shown_counts(table))                        # blanking only removes cells
