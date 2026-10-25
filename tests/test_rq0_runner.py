"""CPU tests of the RQ0 runner and summaries on the dummy segmenter.

`match` (P3) is replaced by a small greedy stand-in with the same contract, so these tests check the
runner's wiring and bookkeeping, not P3's matching. DummySegmenter IoUs are not scientific results.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from promptcorrect.data.geometry import rle_decode
from promptcorrect.eval import matching
from promptcorrect.eval.metrics import iou
from promptcorrect.experiments.rq0 import ORACLE_SOURCES, SOURCES, load_detections, oracle_click, run_rq0
from promptcorrect.experiments.rq0_summary import summarize
from promptcorrect.segmenters.dummy import DummySegmenter

NO_ANNS = {"annotations": []}


def greedy_match(pred_masks, gt_masks, ignore_masks=(), min_iou=0.1, ignore_frac=0.5):
    cand = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(pred_masks) for gi, g in enumerate(gt_masks)), reverse=True)
    pairs, used_p, used_g = [], set(), set()
    for v, pi, gi in cand:
        if v < min_iou:
            break
        if pi not in used_p and gi not in used_g:
            pairs.append((pi, gi, v)); used_p.add(pi); used_g.add(gi)
    return SimpleNamespace(pairs=pairs, unmatched_preds=[i for i in range(len(pred_masks)) if i not in used_p],
                           unmatched_gts=[i for i in range(len(gt_masks)) if i not in used_g], ignored_preds=[])


def make_detections(objects):
    gd, sam = {}, {}
    seg = DummySegmenter(objects)
    for img in sorted({o.image_id for o in objects}):
        seg.set_image(img)
        for rec in [o for o in objects if o.image_id == img]:
            x0, y0, x1, y1 = rec.gt_box
            shifted = [x0 + 0.05 * (x1 - x0), y0, x1 + 0.05 * (x1 - x0), y1]
            gd[(img, rec.category_name)] = [{"image_id": img, "phrase": rec.category_name, "box": shifted, "score": 0.9}]
            sam[(img, rec.category_name)] = [
                {"image_id": img, "phrase": rec.category_name, "box": d[0], "score": d[2], "mask_rle": d[1]}
                for d in seg.detect_text(rec.category_name)]
    return gd, sam


@pytest.fixture
def seg(objects):
    return DummySegmenter(objects)


def test_oracle_click_picks_largest_error_region():
    gt = np.zeros((40, 40), bool); gt[5:35, 5:35] = True
    pred = np.zeros_like(gt); pred[5:35, 5:20] = True          # right half missing
    x, y, label = oracle_click(pred, gt)
    assert label == 1 and x >= 20 and gt[int(y), int(x)] and not pred[int(y), int(x)]
    over = gt.copy(); over[0:5, :] = True                       # false positives on top
    x, y, label = oracle_click(over, gt)
    assert label == 0 and y < 5
    assert oracle_click(gt, gt) is None


def test_gt_box_ceiling_needs_no_matching_and_clicks_help(seg, objects):
    df, queries = run_rq0(seg, objects, NO_ANNS, {}, {}, run_id="t", sources=("gt_box",), oracle_steps=3, log_every=0)
    assert queries.empty and set(df.source) == {"gt_box"} and set(df.prompt_origin) == {"gt"}
    b0 = df[df.method == "b0_none"]
    assert len(b0) == len(objects) and (b0.step == 0).all() and b0.detected.all()
    orc = df[df.method == "oracle_clicks"]
    assert orc.step.max() <= 3 and (orc[orc.step == 0].shape[0] == len(objects))
    assert orc[orc.step == 0].boundary_iou.notna().all() and orc[orc.step > 0].boundary_iou.isna().all()
    first = orc[orc.step == 0].set_index("obj_id").iou
    last = orc.sort_values("step").groupby("obj_id").iou.last()
    assert (last >= first - 1e-9).all() and last.mean() > first.mean()   # the dummy rewards ideal clicks
    assert df.run_id.eq("t").all() and df.n_decoder_calls[df.step == 0].eq(1).all()


def test_all_sources_with_missed_object_and_false_positive(monkeypatch, seg, objects):
    monkeypatch.setattr(matching, "match", greedy_match)
    gd, sam = make_detections(objects)
    del sam[(1, "person")]                                             # SAM 3 misses this object
    gd[(2, "cat")].append({"image_id": 2, "phrase": "cat", "box": [250.0, 5.0, 300.0, 30.0], "score": 0.4})  # a false positive
    df, queries = run_rq0(seg, objects, NO_ANNS, gd, sam, run_id="t", oracle_steps=2, log_every=0)

    assert set(df.source) == set(SOURCES)
    missed = df[(df.obj_id == "coco:100") & df.source.str.startswith("sam3_text")]
    assert not missed.empty and (~missed.detected).all() and (missed.iou == 0).all() and (missed.method == "b0_none").all()
    assert df[(df.obj_id == "coco:100") & (df.source == "grounding_dino_box")].detected.all()

    text_mask = df[(df.source == "sam3_text_mask") & df.detected]
    assert (text_mask.n_decoder_calls == 0).all() and text_mask.pred_iou.isna().all()
    assert set(df[df.method == "oracle_clicks"].source) <= set(ORACLE_SOURCES)       # no oracle on raw text masks
    q = queries[(queries.source == "grounding_dino_box") & (queries.image_id == 2) & (queries.phrase == "cat")].iloc[0]
    assert q.n_pred == 2 and q.n_matched == 1 and q.n_fp == 1


def test_unknown_source_raises(seg, objects):
    with pytest.raises(ValueError, match="unknown sources"):
        run_rq0(seg, objects, NO_ANNS, {}, {}, run_id="t", sources=("nope",))


def test_load_detections_sorts_by_score(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"image_id": 1, "phrase": "cat", "box": [0,0,1,1], "score": 0.3}\n'
                 '{"image_id": 1, "phrase": "cat", "box": [0,0,2,2], "score": 0.9}\n')
    assert [r["score"] for r in load_detections(p)[(1, "cat")]] == [0.9, 0.3]


def test_summaries(monkeypatch, seg, objects):
    monkeypatch.setattr(matching, "match", greedy_match)
    gd, sam = make_detections(objects)
    del sam[(1, "person")]
    df, _ = run_rq0(seg, objects, NO_ANNS, gd, sam, run_id="t", oracle_steps=3, log_every=0)
    tables = summarize(df)
    h = tables["headline"].set_index("source")
    assert set(h.index) == set(SOURCES) and (h["missed_%"] >= 0).all()
    assert h.loc["sam3_text_mask", "missed_%"] > 0 and h.loc["gt_box", "missed_%"] == 0
    for col in [c for c in h.columns if c.startswith("fail<")]:
        assert h[col].dropna().between(0, 100).all()
    reached = tables["oracle_reached_0.85_%_by_clicks"]
    assert (reached.diff(axis=1).iloc[:, 1:].fillna(0) >= -1e-9).all().all()           # cumulative: never decreases
    assert "calibration" in tables and set(tables["calibration"].source) == set(ORACLE_SOURCES)
    assert not tables["failure_rate_by_size_bin_iou<0.75"].empty
