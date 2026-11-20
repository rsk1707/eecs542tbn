import numpy as np

from promptcorrect.data.records import decode_rle
from promptcorrect.eval.oracle_clicks import best_so_far, ideal_click, noc, run_oracle
from promptcorrect.types import Prompt


def test_positive_click_at_centre_of_missed_region():
    gt = np.zeros((40, 40), bool); gt[10:30, 10:30] = True
    pred = gt.copy(); pred[10:30, 20:30] = False             # right half missed: 20 x 10 strip
    x, y, label = ideal_click(pred, gt)
    assert label == 1 and 20 <= x <= 29 and 14 <= y <= 25    # deepest point of the strip
    assert ideal_click(gt, gt) is None


def test_negative_click_when_false_positive_is_deeper():
    gt = np.zeros((40, 40), bool); gt[10:20, 10:20] = True
    pred = gt.copy(); pred[0:40, 25:40] = True                # big spill on the right
    x, y, label = ideal_click(pred, gt)
    assert label == 0 and x >= 25 and not gt[int(y), int(x)]


def test_already_clicked_pixel_is_not_repeated():
    gt = np.zeros((9, 9), bool); gt[4, 4] = True
    assert ideal_click(np.zeros_like(gt), gt) == (4.0, 4.0, 1)
    assert ideal_click(np.zeros_like(gt), gt, [(4.0, 4.0, 1)]) is None


def test_oracle_on_dummy_improves_and_records_each_step(seg, objects):
    obj = next(o for o in objects if o.image_id == 1)
    gt = decode_rle(obj.gt_mask)
    tr = run_oracle(seg, Prompt(box=list(obj.gt_box)), gt, max_clicks=3)
    assert len(tr.ious) == len(tr.pred_ious) == len(tr.latency_ms) == len(tr.clicks) + 1
    assert max(tr.ious[1:]) > tr.ious[0]
    for x, y, label in tr.clicks:
        assert bool(gt[int(y), int(x)]) == bool(label)        # clicks agree with the GT


def test_noc_and_best_so_far():
    ious = [0.6, 0.86, 0.8, 0.91]
    assert best_so_far(ious) == [0.6, 0.86, 0.86, 0.91]
    assert noc(ious, 0.85, 3) == 1 and noc(ious, 0.9, 3) == 3 and noc(ious, 0.95, 3) == 4
