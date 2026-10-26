"""RQ0 study: how often do real prompts give poor masks, and how much can ideal clicks fix? Owner: P3.

Every GT object gets rows for each requested prompt source (`prompt_origin`), SAM 3 frozen:

  gt              GT box -> interactive head. The ceiling for a perfect box.
  grounding_dino  Grounding DINO boxes (independent detector) -> interactive head.
  sam3_text       SAM 3's own text-detection boxes -> interactive head (isolates box quality).
  sam3_text_mask  SAM 3's text-detection masks as they are (no decoder call).

Detections are assigned to GT objects with `match` per (image, phrase). A GT object no detection
matched gets one `b0_none` step-0 row with iou 0 and `detected=False`. Box sources also get an
`oracle_clicks` trace: step 0 = the box alone, step k = k ideal clicks on top of it.
Boundary IoU is computed at step 0 only (NaN at click steps).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from promptcorrect.data.records import annotation_mask, decode_rle
from promptcorrect.eval.matching import match
from promptcorrect.eval.metrics import boundary_iou, iou
from promptcorrect.eval.oracle_clicks import run_oracle
from promptcorrect.types import ObjectRecord, Prompt, ResultRow

SOURCES = ("gt", "grounding_dino", "sam3_text", "sam3_text_mask")
DETECTION_FILE = {"grounding_dino": "grounding_dino", "sam3_text": "sam3_text", "sam3_text_mask": "sam3_text"}
NAN = float("nan")


def phrase_of(rec: ObjectRecord) -> str:
    return rec.category_name.replace("_", " ")  # same rule as detectors/export.py


def load_detections(path: str | Path) -> dict[tuple[int, str], list[dict]]:
    """Detection JSONL -> {(image_id, phrase): rows, highest score first}."""
    out: dict[tuple[int, str], list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[(row["image_id"], row["phrase"])].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: -r["score"])
    return dict(out)


def ignore_masks(anns: list[dict], record_ids: set[str], category_id: int, hw: tuple[int, int]) -> list[np.ndarray]:
    """Same-category annotations that are not object records (crowd or size-filtered), as in COCO eval."""
    return [annotation_mask(a["segmentation"], *hw) for a in anns
            if a["category_id"] == category_id and f"coco:{a['id']}" not in record_ids]


def _predict_boxes(segmenter, boxes, origin):
    """One batched decoder call -> (outputs, ms per prompt)."""
    t = time.perf_counter()
    outs = segmenter.predict([Prompt(box=list(b), origin=origin) for b in boxes]) if boxes else []
    return outs, (time.perf_counter() - t) * 1000 / max(len(boxes), 1)


def run_rq0(segmenter, records, annotations: dict, detections: dict[str, dict], *, run_id: str,
            sources=SOURCES, max_clicks: int = 5, use_prev_mask: bool = True, max_images: int | None = None,
            log_every: int = 25) -> list[ResultRow]:
    """`detections` maps a DETECTION_FILE name to a `load_detections` dict (only the ones `sources` need)."""
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; choose from {SOURCES}")
    by_image: dict[int, list[ObjectRecord]] = defaultdict(list)
    for r in records:
        by_image[r.image_id].append(r)
    record_ids = {r.obj_id for recs in by_image.values() for r in recs}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in annotations["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    rows: list[ResultRow] = []

    def add(rec, origin, method, step, i, b, **kw):
        rows.append(ResultRow(run_id=run_id, obj_id=rec.obj_id, image_id=rec.image_id, method=method, step=step,
                              iou=i, boundary_iou=b, prompt_origin=origin, novelty=rec.novelty,
                              size_bin=rec.size_bin, thinness=rec.thinness, **kw))

    def evaluate(rec, origin, gt, box=None, out=None, ms=NAN, mask=None, det_score=None):
        """Rows for one prompt on one GT object: b0_none, plus the oracle trace for box prompts."""
        if box is None:  # mask used as it is
            add(rec, origin, "b0_none", 0, iou(mask, gt), boundary_iou(mask, gt), det_score=det_score)
            return
        tr = run_oracle(segmenter, Prompt(box=list(box), origin=origin), gt, max_clicks,
                        start=out, start_ms=ms, use_prev_mask=use_prev_mask, keep_masks=True)
        b0 = boundary_iou(tr.masks[0], gt)
        add(rec, origin, "b0_none", 0, tr.ious[0], b0, n_decoder_calls=1, latency_ms=ms,
            det_score=det_score, pred_iou=tr.pred_ious[0])
        for k, (i, pi, t) in enumerate(zip(tr.ious, tr.pred_ious, tr.latency_ms)):
            add(rec, origin, "oracle_clicks", k, i, b0 if k == 0 else NAN, n_points=k, n_decoder_calls=1 + k,
                latency_ms=t, det_score=det_score, pred_iou=pi)

    image_ids = sorted(by_image)[:max_images]
    t0 = time.time()
    for n, image_id in enumerate(image_ids, 1):
        objs = by_image[image_id]
        gts = {r.obj_id: decode_rle(r.gt_mask) for r in objs}
        hw = next(iter(gts.values())).shape
        segmenter.set_image(image_id)

        if "gt" in sources:
            outs, ms = _predict_boxes(segmenter, [r.gt_box for r in objs], "gt")
            for rec, out in zip(objs, outs):
                evaluate(rec, "gt", gts[rec.obj_id], box=rec.gt_box, out=out, ms=ms)

        groups: dict[str, list[ObjectRecord]] = defaultdict(list)
        for r in objs:
            groups[phrase_of(r)].append(r)
        for phrase, recs in groups.items():
            ignore = None
            for origin in (s for s in sources if s != "gt"):
                dets = detections[DETECTION_FILE[origin]].get((image_id, phrase), [])
                if origin == "sam3_text_mask":
                    outs, ms = [], NAN
                    pred_masks = [decode_rle(d["mask_rle"]) for d in dets]
                else:
                    outs, ms = _predict_boxes(segmenter, [d["box"] for d in dets], origin)
                    pred_masks = [segmenter.full_res_mask(o) for o in outs]
                if ignore is None:
                    ignore = ignore_masks(anns_by_image[image_id], record_ids, recs[0].category_id, hw)
                res = match(pred_masks, [gts[r.obj_id] for r in recs], ignore_masks=ignore)
                pred_for = {g: p for p, g, _ in res.pairs}
                for g, rec in enumerate(recs):
                    gt = gts[rec.obj_id]
                    if g not in pred_for:
                        add(rec, origin, "b0_none", 0, 0.0, 0.0, detected=False)
                        continue
                    p = pred_for[g]
                    if origin == "sam3_text_mask":
                        evaluate(rec, origin, gt, mask=pred_masks[p], det_score=dets[p]["score"])
                    else:
                        evaluate(rec, origin, gt, box=dets[p]["box"], out=outs[p], ms=ms, det_score=dets[p]["score"])
        if log_every and n % log_every == 0:
            print(f"[rq0] {n}/{len(image_ids)} images, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return rows
