"""Baselines on the dummy segmenter: logic only (the dummy's IoUs are not results)."""

import numpy as np
import pytest

from promptcorrect.baselines.rules import box_refit, rollout, uncertainty_click
from promptcorrect.eval.metrics import iou
from promptcorrect.perturb.box import truncate_box
from promptcorrect.types import Prompt


def _gt(seg, objects):
    return seg._gt[0], [o for o in objects if o.image_id == 1][0]


@pytest.mark.parametrize("method", ["b0_none", "b1_random_k", "b2_multimask", "b3_box_refit", "b3_unc_click"])
def test_rollout_shapes_and_bookkeeping(seg, objects, method):
    _, obj = _gt(seg, objects)
    traj = rollout(seg, method, Prompt(box=obj.gt_box), steps=3, rng=np.random.default_rng(0))
    assert traj[0].step == 0 and traj[0].n_decoder_calls == 1 and not traj[0].prompt.points
    assert [s.step for s in traj] == list(range(len(traj))) and len(traj) <= 4
    assert [s.n_decoder_calls for s in traj] == list(range(1, len(traj) + 1))
    assert all(s.mask.dtype == bool and s.mask.shape == traj[0].mask.shape for s in traj)
    if method == "b0_none":
        assert len(traj) == 1
    if method == "b2_multimask":
        assert len(traj) == 2
    if method in ("b1_random_k", "b3_unc_click"):
        assert [len(s.prompt.points) for s in traj] == list(range(len(traj)))
        assert all(s.prompt.origin == f"baseline:{method}" for s in traj[1:])


def test_random_clicks_are_inside_the_box_and_seeded(seg, objects):
    _, obj = _gt(seg, objects)
    a = rollout(seg, "b1_random_k", Prompt(box=obj.gt_box), 3, np.random.default_rng(1))
    b = rollout(seg, "b1_random_k", Prompt(box=obj.gt_box), 3, np.random.default_rng(1))
    assert a[-1].prompt.points == b[-1].prompt.points
    x0, y0, x1, y1 = obj.gt_box
    assert all(x0 <= x <= x1 and y0 <= y <= y1 and lab == 1 for x, y, lab in a[-1].prompt.points)


def test_box_refit_fixes_a_loose_box_and_then_stops(seg, objects):
    gt, obj = _gt(seg, objects)
    x0, y0, x1, y1 = obj.gt_box
    loose = Prompt(box=[x0 - 30, y0 - 20, x1 + 30, y1 + 20])
    traj = rollout(seg, "b3_box_refit", loose, 5)
    assert len(traj) < 6                                   # converges, stops early
    assert traj[1].prompt.box[0] > loose.box[0]            # box got tighter
    assert box_refit(traj[-1].prompt, None, traj[-1].mask, None) is None


def test_box_refit_with_margin_grows_a_truncated_box(seg, objects):
    gt, obj = _gt(seg, objects)
    p = truncate_box(obj.gt_box, 0.4, "right", tuple(obj.gt_mask["size"]))
    traj = rollout(seg, "b3_box_refit", p, 3, margin=0.1)
    assert traj[1].prompt.box[2] > p.box[2]
    assert iou(traj[-1].mask, gt) > iou(traj[0].mask, gt)


def test_uncertainty_click_targets_low_logits_and_labels_by_box():
    logits = np.full((50, 50), -8.0)
    logits[10:40, 10:40] = 8.0
    logits[10:40, 30:40] = 0.2                              # uncertain strip on the right of the mask
    p = uncertainty_click(Prompt(box=[10, 10, 40, 40]), logits, logits > 0, None)
    x, y, label = p.points[-1]
    assert 30 <= x <= 40 and 10 <= y <= 40 and label == 1
    p = uncertainty_click(Prompt(box=[10, 10, 28, 40]), logits, logits > 0, None)
    assert p.points[-1][2] == 0                             # outside the prompt box -> negative
    assert uncertainty_click(Prompt(box=[0, 0, 50, 50]), np.full((50, 50), 9.0), np.ones((50, 50), bool), None) is None


def test_uncertainty_click_does_not_repeat_a_click():
    logits = np.full((50, 50), -8.0)
    logits[20:25, 20:25] = 0.1
    p1 = uncertainty_click(Prompt(box=[0, 0, 50, 50]), logits, logits > 0, None)
    assert uncertainty_click(p1, logits, logits > 0, None) is None


def test_unknown_method():
    with pytest.raises(ValueError):
        rollout(None, "b9_magic", Prompt(box=[0, 0, 1, 1]), 1)
