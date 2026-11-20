import pytest

from promptcorrect.eval.aggregate import all_tables, click_curve, failure_table, harm_table, noc_table, step_curve
from promptcorrect.eval.results import to_frame
from promptcorrect.types import ResultRow


def _r(obj, origin, method, step, iou, **kw):
    return ResultRow(run_id="r", obj_id=obj, image_id=1, method=method, step=step, iou=iou, boundary_iou=iou,
                     prompt_origin=origin, size_bin="small", novelty="seen", **kw)


def _df():
    rows = [
        # object a, GT box: 0.6 -> 0.9 -> 0.8 (non-monotone)
        _r("a", "gt", "b0_none", 0, 0.6, pred_iou=0.9),
        *[_r("a", "gt", "oracle_clicks", k, v) for k, v in enumerate([0.6, 0.9, 0.8])],
        # object b, GT box: 0.95, stops at once (mask exact after 1 click)
        _r("b", "gt", "b0_none", 0, 0.95, pred_iou=0.95),
        *[_r("b", "gt", "oracle_clicks", k, v) for k, v in enumerate([0.95, 1.0])],
        # detector: a detected at 0.4 and the click hurts; b missed
        _r("a", "gd", "b0_none", 0, 0.4, pred_iou=0.8, det_score=0.5),
        *[_r("a", "gd", "oracle_clicks", k, v) for k, v in enumerate([0.4, 0.3, 0.35])],
        _r("b", "gd", "b0_none", 0, 0.0, detected=False),
    ]
    return to_frame(rows)


def test_failure_table_counts_missed_as_failures():
    t = failure_table(_df())
    assert t.loc["gd", "detected"] == 0.5 and t.loc["gd", "mean_iou"] == pytest.approx(0.2)
    assert t.loc["gd", "mean_iou_detected"] == pytest.approx(0.4) and t.loc["gd", "fail@0.5"] == 1.0
    assert t.loc["gt", "fail@0.75"] == 0.5 and t.loc["gt", "fail@0.9"] == 0.5
    assert t.loc["gt", "overconfidence"] == pytest.approx((0.9 + 0.95) / 2 - (0.6 + 0.95) / 2)


def test_click_curve_is_best_so_far_and_fills_stopped_traces():
    c = click_curve(_df())
    assert c.loc["gt", 2] == pytest.approx((0.9 + 1.0) / 2)   # a keeps its 0.9; b stays at 1.0
    assert c.loc["gd", 1] == pytest.approx(0.4) and c.loc["gt", "n"] == 2


def test_noc_and_harm():
    n = noc_table(_df(), thresholds=(0.85,))
    assert n.loc["gt", "noc@85"] == pytest.approx((1 + 0) / 2) and n.loc["gt", "not_reached@85"] == 0
    assert n.loc["gd", "noc@85"] == 3 and n.loc["gd", "not_reached@85"] == 1   # budget 2 -> counted as 3
    h = harm_table(_df()).xs("oracle_clicks", level="method")
    assert h.loc["gd", "harm_rate"] == 1.0 and h.loc["gt", "harm_rate"] == 0.0
    assert h.loc["gt", "mean_gain"] == pytest.approx((0.2 + 0.05) / 2)


def test_step_curve_is_raw_not_best_so_far():
    c = step_curve(_df())
    assert c.loc[("gt", "oracle_clicks"), 2] == pytest.approx((0.8 + 1.0) / 2)  # a drops back to 0.8
    assert c.loc[("gd", "oracle_clicks"), 1] == pytest.approx(0.3)


def test_all_tables_runs():
    assert {"failure", "noc_failures", "harm", "click_curve_failures", "methods"} <= set(all_tables(_df()))
