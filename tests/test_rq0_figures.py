"""The RQ0 figure code runs on the rows a real run produces (dummy segmenter; the pictures mean nothing)."""

import pytest

pytest.importorskip("matplotlib")

from promptcorrect.experiments.rq0 import run_rq0  # noqa: E402
from promptcorrect.segmenters.dummy import DummySegmenter  # noqa: E402
from promptcorrect.viz import rq0_figures  # noqa: E402
from tests.test_rq0_runner import NO_ANNS, make_detections  # noqa: E402


def test_make_all_writes_four_figures(tmp_path, objects):
    gd, sam = make_detections(objects)
    df, _ = run_rq0(DummySegmenter(objects), objects, NO_ANNS, gd, sam, run_id="t", oracle_steps=3, log_every=0)
    files = rq0_figures.make_all(df, tmp_path / "figures")
    assert {f.name for f in files} == {"rq0_failure_rates.png", "rq0_clicks.png", "rq0_overconfidence.png",
                                       "rq0_by_size.png"}
    assert all(f.exists() and f.stat().st_size > 5000 for f in files)


def test_ceiling_only_run_still_plots(tmp_path, objects):
    df, _ = run_rq0(DummySegmenter(objects), objects, NO_ANNS, {}, {}, run_id="t", sources=("gt_box",),
                    oracle_steps=2, log_every=0)
    files = rq0_figures.make_all(df, tmp_path)
    assert len(files) == 4


def test_colors_are_fixed_per_source():
    assert list(rq0_figures.COLORS) == ["gt_box", "grounding_dino_box", "sam3_text_box", "sam3_text_mask"]
    assert len(set(rq0_figures.COLORS.values())) == 4
