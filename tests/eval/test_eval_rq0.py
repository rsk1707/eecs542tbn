import numpy as np
import pandas as pd

from promptcorrect.data.records import encode_rle
from promptcorrect.eval.results import read_results, write_results
from promptcorrect.eval.rq0 import run_rq0


def _setup(objects):
    anns = [dict(id=int(o.obj_id.split(":")[1]), image_id=o.image_id, category_id=o.category_id,
                 segmentation=o.gt_mask) for o in objects]
    crowd = np.zeros((240, 320), bool); crowd[0:20, 0:60] = True        # a person crowd in image 1
    anns.append(dict(id=999, image_id=1, category_id=1, segmentation=encode_rle(crowd)))
    person = next(o for o in objects if o.image_id == 1 and o.category_name == "person")
    x0, y0, x1, y1 = person.gt_box
    gd = {(1, "person"): [dict(box=[x0 + 4, y0 + 4, x1 - 4, y1 - 4], score=0.8)]}  # image 1 cat: no detection
    sam = {(1, "person"): [dict(box=person.gt_box, score=0.9, mask_rle=person.gt_mask),
                           dict(box=[0, 0, 60, 20], score=0.4, mask_rle=encode_rle(crowd))]}
    return {"annotations": anns}, {"grounding_dino": gd, "sam3_text": sam}, person


def test_rq0_end_to_end_on_dummy(seg, objects, tmp_path):
    annotations, dets, person = _setup(objects)
    rows = run_rq0(seg, objects, annotations, dets, run_id="demo_t", max_clicks=3, log_every=0)
    write_results(rows, tmp_path / "demo_t")                          # validates registry + unique keys
    df = read_results(tmp_path / "demo_t")
    b0 = df[(df.method == "b0_none") & (df.step == 0)].set_index(["prompt_origin", "obj_id"])

    assert len(b0.loc["gt"]) == len(objects) and b0.loc["gt", "detected"].all()
    assert b0.loc[("sam3_text_mask", person.obj_id), "iou"] == 1.0   # its GT mask, crowd detection ignored
    gd = b0.loc["grounding_dino"]
    assert gd.loc[person.obj_id, "detected"] and gd.loc[person.obj_id, "det_score"] == 0.8
    missed = gd[~gd.detected]
    assert len(missed) == len(objects) - 1 and (missed.iou == 0).all()
    assert df[(df.prompt_origin == "grounding_dino") & (df.method == "oracle_clicks")].obj_id.unique().tolist() == [person.obj_id]

    oc = df[(df.prompt_origin == "gt") & (df.method == "oracle_clicks")]
    first = oc[oc.obj_id == person.obj_id].sort_values("step")
    assert first.step.tolist() == list(range(len(first))) and first.n_points.tolist() == first.step.tolist()
    assert first.iou.max() > first.iou.iloc[0]
    assert first.boundary_iou.iloc[1:].isna().all() and pd.notna(first.boundary_iou.iloc[0])
    assert np.isclose(first.iou.iloc[0], b0.loc[("gt", person.obj_id), "iou"])


def test_rq0_baselines_share_step0_and_budget(seg, objects, tmp_path):
    annotations, dets, person = _setup(objects)
    methods = ("oracle_clicks", "b1_random_k", "b2_multimask", "b3_box_refit", "b3_unc_click")
    rows = run_rq0(seg, objects, annotations, dets, run_id="demo_b", sources=("gt",), methods=methods,
                   max_clicks=2, use_prev_mask=False, log_every=0)
    df = read_results(write_results(rows, tmp_path / "demo_b").parent)
    assert set(df.method) == {"b0_none", *methods}
    s0 = df[df.step == 0].groupby("obj_id").iou.nunique()
    assert (s0 == 1).all()                                            # every method starts from the same mask
    assert df.step.max() == 2 and df[df.method == "b2_multimask"].step.max() == 1
    again = run_rq0(seg, objects, annotations, dets, run_id="demo_b", sources=("gt",), methods=("b1_random_k",),
                    max_clicks=2, log_every=0)
    r1 = [(r.obj_id, r.step, r.iou) for r in rows if r.method == "b1_random_k"]
    assert r1 == [(r.obj_id, r.step, r.iou) for r in again if r.method == "b1_random_k"]  # seeded per object
