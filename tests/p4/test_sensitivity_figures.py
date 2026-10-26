"""The sweep figure code runs on the rows a real sweep produces (dummy segmenter; the pictures mean nothing)."""

import pytest

pytest.importorskip("matplotlib")

from promptcorrect.perturb.sweep import METHODS, run_sweep  # noqa: E402
from promptcorrect.segmenters.dummy import DummySegmenter  # noqa: E402
from promptcorrect.viz import sensitivity_figures  # noqa: E402


def test_make_all_writes_three_figures(tmp_path, objects):
    df = run_sweep(DummySegmenter(objects), objects, run_id="t", methods=METHODS, steps=2,
                   levels={"shift": (0.1, 0.3), "scale": (0.5, 1.5), "truncate": (0.2, 0.4)}, log_every=0)
    files = sensitivity_figures.make_all(df, tmp_path)
    assert {f.name for f in files} == {"sensitivity_curves.png", "sensitivity_by_box_iou.png", "baselines_vs_oracle.png"}
    assert all(f.stat().st_size > 5000 for f in files)


def test_b0_only_sweep_skips_the_method_figure(tmp_path, objects):
    df = run_sweep(DummySegmenter(objects), objects, run_id="t", log_every=0)
    assert len(sensitivity_figures.make_all(df, tmp_path)) == 2
