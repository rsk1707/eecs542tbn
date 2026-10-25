"""Tests of the stand-in match (follows P3's published contract) and of the runner's choice between the two."""

import itertools

import numpy as np
import pytest

from promptcorrect.eval.metrics import iou
from promptcorrect.experiments import match_fallback
from promptcorrect.experiments.match_fallback import match
from promptcorrect.experiments.rq0 import resolve_match, run_rq0
from promptcorrect.segmenters.dummy import DummySegmenter
from tests.test_rq0_runner import NO_ANNS, make_detections


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_perfect_one_to_one_and_extra_prediction_is_a_false_positive():
    g0, g1 = box(50, 50, 0, 20, 0, 20), box(50, 50, 30, 50, 30, 50)
    r = match([g1, g0, box(50, 50, 22, 28, 22, 28)], [g0, g1])
    assert [(p, g) for p, g, _ in r.pairs] == [(0, 1), (1, 0)] and all(v == 1.0 for *_, v in r.pairs)
    assert r.unmatched_preds == [2] and r.unmatched_gts == [] and r.ignored_preds == []


def test_two_predictions_one_gt_keeps_the_better_one():
    gt = box(40, 40, 0, 20, 0, 20)
    good, worse = box(40, 40, 0, 20, 0, 18), box(40, 40, 0, 20, 0, 12)
    r = match([worse, good], [gt])
    assert [(p, g) for p, g, _ in r.pairs] == [(1, 0)] and r.unmatched_preds == [0]


def test_min_iou_drops_weak_pairs_on_both_sides():
    gt = box(40, 40, 0, 20, 0, 20)
    weak = box(40, 40, 0, 20, 18, 40)           # IoU well below 0.1
    r = match([weak], [gt], min_iou=0.1)
    assert r.pairs == [] and r.unmatched_preds == [0] and r.unmatched_gts == [0]


def test_ignore_masks_exclude_predictions_mostly_inside_crowds():
    gt = box(40, 40, 0, 10, 0, 10)
    crowd = box(40, 40, 20, 40, 20, 40)
    inside = box(40, 40, 22, 38, 22, 38)
    r = match([inside, gt], [gt], ignore_masks=[crowd])
    assert r.ignored_preds == [0] and [(p, g) for p, g, _ in r.pairs] == [(1, 0)] and r.unmatched_preds == []


def test_empty_inputs():
    gt = box(10, 10, 0, 5, 0, 5)
    assert match([], [gt]).unmatched_gts == [0]
    r = match([gt], [])
    assert r.unmatched_preds == [0] and r.pairs == []
    assert match([], []).pairs == []


def test_total_iou_equals_the_optimum_over_all_assignments():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n, m = (int(x) for x in rng.integers(1, 4, 2))

        def rects(k):
            out = []
            for _ in range(k):
                y0, x0 = rng.integers(0, 20, 2)
                out.append(box(40, 40, y0, y0 + rng.integers(8, 20), x0, x0 + rng.integers(8, 20)))
            return out

        preds, gts = rects(n), rects(m)
        got = sum(v for *_, v in match(preds, gts, min_iou=0.0).pairs)
        k = min(n, m)
        best = max(sum(iou(preds[p], gts[g]) for p, g in zip(ps, gs))
                   for ps in itertools.permutations(range(n), k) for gs in itertools.permutations(range(m), k))
        assert got == pytest.approx(best, abs=1e-6)


def test_resolver_uses_the_stand_in_while_p3_match_is_a_stub():
    fn, name = resolve_match()
    assert "P1 stand-in" in name and fn is match_fallback.match


def test_runner_works_end_to_end_with_the_stand_in(objects):
    seg = DummySegmenter(objects)
    gd, sam = make_detections(objects)
    del sam[(1, "person")]
    df, queries = run_rq0(seg, objects, NO_ANNS, gd, sam, run_id="t", oracle_steps=2, log_every=0)
    assert "P1 stand-in" in df.attrs["match_impl"]
    missed = df[(df.obj_id == "coco:100") & df.source.str.startswith("sam3_text") & (df.method == "b0_none")]
    assert (~missed.detected).all() and (df[df.detected & (df.method == "b0_none")].iou > 0).all()
    assert (queries.n_matched <= queries.n_gt).all()
