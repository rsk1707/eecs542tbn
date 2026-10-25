import math

import numpy as np
import pytest

from promptcorrect.perturb.box import (SWEEP_LEVELS, clip_box, object_rng, perturb, sample_perturbation,
                                       scale_box, shift_box, sweep_prompts, truncate_box)
from promptcorrect.perturb.severity import box_iou

HW = (400, 600)
BOX = [100.0, 100.0, 200.0, 300.0]  # w = 100, h = 200


def test_shift_is_relative_to_box_size():
    p = shift_box(BOX, 0.2, 0.0, HW)
    assert p.box == pytest.approx([120, 100, 220, 300])
    p = shift_box(BOX, 0.1, math.pi / 2, HW)
    assert p.box == pytest.approx([100, 120, 200, 320])
    assert p.origin == "perturbed" and p.perturbation["type"] == "shift" and p.perturbation["level"] == 0.1
    assert not p.perturbation["params"]["clipped"]


def test_scale_keeps_centre_and_truncate_cuts_one_side():
    assert scale_box(BOX, 2.0, HW).box == pytest.approx([50, 0, 250, 400])
    assert scale_box(BOX, 0.5, HW).box == pytest.approx([125, 150, 175, 250])
    assert truncate_box(BOX, 0.3, "left", HW).box == pytest.approx([130, 100, 200, 300])
    assert truncate_box(BOX, 0.25, "bottom", HW).box == pytest.approx([100, 100, 200, 250])
    with pytest.raises(ValueError):
        truncate_box(BOX, 1.0, "left", HW)
    with pytest.raises(ValueError):
        truncate_box(BOX, 0.2, "middle", HW)


def test_clipping_never_gives_an_empty_or_outside_box():
    p = shift_box([500, 300, 590, 390], 0.5, 0.0, HW)
    assert p.perturbation["params"]["clipped"] and p.box[2] <= 600
    for box in ([-50, -50, -10, -10], [700, 500, 800, 600]):
        x0, y0, x1, y1 = clip_box(box, HW)
        assert 0 <= x0 < x1 <= 600 and 0 <= y0 < y1 <= 400 and x1 - x0 >= 1 and y1 - y0 >= 1


@pytest.mark.parametrize("ptype", ["shift", "scale", "truncate"])
def test_box_iou_falls_as_level_grows(ptype):
    levels = SWEEP_LEVELS[ptype]
    if ptype == "scale":  # severity grows away from 1 in both directions
        down, up = [lv for lv in levels if lv < 1][::-1], [lv for lv in levels if lv > 1]
        for side in (down, up):
            ious = [box_iou(perturb(BOX, "scale", lv, HW, object_rng(0, "a")).box, BOX) for lv in side]
            assert all(a > b for a, b in zip(ious, ious[1:]))
        return
    ious = [box_iou(perturb(BOX, ptype, lv, HW, object_rng(0, "a")).box, BOX) for lv in levels]
    assert all(a > b for a, b in zip(ious, ious[1:]))


def test_rng_makes_sweeps_reproducible_and_order_independent():
    a = sweep_prompts(BOX, HW, object_rng(0, "coco:1"))
    b = sweep_prompts(BOX, HW, object_rng(0, "coco:1"))
    c = sweep_prompts(BOX, HW, object_rng(1, "coco:1"))
    assert [p.box for p in a] == [p.box for p in b] and [p.box for p in a] != [p.box for p in c]
    assert a[0].origin == "gt" and a[0].perturbation is None and a[0].box == BOX
    assert len(a) == 1 + sum(len(v) for v in SWEEP_LEVELS.values())


def test_sample_perturbation_stays_in_range():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        p = sample_perturbation(BOX, HW, rng)
        t, lv = p.perturbation["type"], p.perturbation["level"]
        seen.add(t)
        lo, hi = {"shift": (0, 0.5), "scale": (0.5, 2.0), "truncate": (0, 0.5)}[t]
        assert lo <= lv <= hi
    assert seen == {"shift", "scale", "truncate"}
