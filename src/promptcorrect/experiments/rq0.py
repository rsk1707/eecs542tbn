"""RQ0 preliminary study: how often do real prompts give poor masks, and how much can ideal clicks fix?

For every GT object (P2's records) four prompt sources are evaluated, all with SAM 3 frozen:

  gt_box              the GT box -> interactive head. The ceiling for a perfect prompt.
  grounding_dino_box  Grounding DINO boxes (independent detector) -> interactive head.
  sam3_text_box       SAM 3's own text-detection boxes -> interactive head (isolates box quality).
  sam3_text_mask      SAM 3's own text-detection masks, used as they are.

Detector outputs are attributed to GT objects with P3's `match` (one-to-one Hungarian on mask IoU, per
image and phrase, crowd/filtered annotations ignored). A GT object nobody detects is *missed*
(`detected=False`, iou 0): reported separately from *detected but poor mask*. For the box-based sources,
ideal clicks (largest error region, point farthest from its boundary) are then simulated step by step.

Output is one row per (object, source, method, step), long format (docs/interfaces.md section 5) plus
the extra columns `source`, `detected`, `det_score`, `pred_iou`, `n_gt_same_class`, `n_pred`, `n_fp`.
`boundary_iou` is computed at step 0 only unless `boundary_all` (it is slow and not needed for RQ0).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from promptcorrect.data.geometry import rle_decode
from promptcorrect.data.records import annotation_mask
from promptcorrect.eval import matching
from promptcorrect.eval.methods import check_method
from promptcorrect.eval.metrics import boundary_iou, iou
from promptcorrect.experiments import match_fallback
from promptcorrect.types import ObjectRecord, Prompt

SOURCES = ("gt_box", "grounding_dino_box", "sam3_text_box", "sam3_text_mask")
ORIGIN = {"gt_box": "gt", "grounding_dino_box": "grounding_dino", "sam3_text_box": "sam3_text",
          "sam3_text_mask": "sam3_text"}
ORACLE_SOURCES = ("gt_box", "grounding_dino_box", "sam3_text_box")
NAN = float("nan")


def oracle_click(pred: np.ndarray, gt: np.ndarray):
    """Ideal click: in the larger error region (false negative -> positive click, false positive ->
    negative click), the point farthest from that region's boundary. None if the mask is already exact."""
    fn, fp = gt & ~pred, pred & ~gt
    if not fn.any() and not fp.any():
        return None
    region, label = (fn, 1) if fn.sum() >= fp.sum() else (fp, 0)
    comp, n = ndimage.label(region)
    sizes = ndimage.sum(region, comp, range(1, n + 1))
    biggest = comp == (1 + int(np.argmax(sizes)))
    dist = ndimage.distance_transform_edt(np.pad(biggest, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return float(x), float(y), label


def resolve_match():
    """P3's `match` if implemented, else the P1 stand-in with the same contract. Returns (function, name)."""
    try:
        matching.match([], [])
    except NotImplementedError:
        return match_fallback.match, "experiments.match_fallback (P1 stand-in, same contract)"
    except Exception:  # P3's version exists but dislikes empty input: still use it
        pass
    return matching.match, "eval.matching.match (P3)"


def load_detections(path: str | Path) -> dict[tuple[int, str], list[dict]]:
    """JSONL detections -> {(image_id, phrase): [rows sorted by score, highest first]}."""
    out: dict[tuple[int, str], list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[(row["image_id"], row["phrase"])].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: -r["score"])
    return dict(out)


def _ignore_masks(anns, record_ids, h, w):
    """Masks of annotations that are not in the object records (crowd or size-filtered)."""
    out = []
    for ann in anns:
        if f"coco:{ann['id']}" not in record_ids:
            m = annotation_mask(ann["segmentation"], h, w)
            if m.any():
                out.append(m)
    return out


class _Rows:
    """Builds result rows with the shared columns filled in."""

    def __init__(self, run_id: str, boundary_all: bool):
        self.run_id, self.boundary_all, self.rows = run_id, boundary_all, []

    def add(self, rec: ObjectRecord, source: str, method: str, step: int, mask, gt, **extra) -> dict:
        detected = extra.pop("detected", True)
        b = NAN
        if mask is None:
            i, b = 0.0, 0.0
        else:
            i = iou(mask, gt)
            if step == 0 or self.boundary_all:
                b = boundary_iou(mask, gt)
        row = dict(
            run_id=self.run_id, obj_id=rec.obj_id, image_id=rec.image_id, method=check_method(method), step=step,
            source=source, prompt_origin=ORIGIN[source], iou=i, boundary_iou=b, detected=detected,
            category=rec.category_name, novelty=rec.novelty, size_bin=rec.size_bin, thinness=rec.thinness,
            n_points=0, n_decoder_calls=0, latency_ms=NAN, det_score=NAN, pred_iou=NAN,
            n_gt_same_class=NAN, n_pred=NAN, n_fp=NAN)
        row.update(extra)
        self.rows.append(row)
        return row


def _predict_boxes(segmenter, boxes, origin):
    """Boxes -> (bool masks, predicted IoUs, ms per prompt) in ONE batched decoder call."""
    if not boxes:
        return [], [], NAN
    t = time.perf_counter()
    outs = segmenter.predict([Prompt(box=list(b), origin=origin) for b in boxes])
    masks = [segmenter.full_res_mask(o) for o in outs]
    return masks, [o.iou_score for o in outs], (time.perf_counter() - t) * 1000 / len(boxes)


def _oracle_rows(segmenter, rows: _Rows, rec, source, box, mask0, gt, steps, common):
    """Oracle clicks for one object: step 0 = the box-only mask, then up to `steps` ideal clicks."""
    rows.add(rec, source, "oracle_clicks", 0, mask0, gt, n_decoder_calls=1, **common)
    mask, points = mask0, []
    for s in range(1, steps + 1):
        click = oracle_click(mask, gt)
        if click is None:
            break
        points.append(click)
        t = time.perf_counter()
        out = segmenter.predict([Prompt(box=list(box), points=list(points), origin="oracle")])[0]
        mask = segmenter.full_res_mask(out)
        rows.add(rec, source, "oracle_clicks", s, mask, gt, n_points=s, n_decoder_calls=1 + s,
                 latency_ms=(time.perf_counter() - t) * 1000, **common)


def run_rq0(segmenter, records, annotations, grounding_dino, sam3_text, *, run_id, sources=SOURCES,
            oracle_steps=5, boundary_all=False, max_images=None, log_every=25):
    """Run the study. `grounding_dino` / `sam3_text` are `load_detections` dicts (may be empty if the
    sources that need them are not requested). Returns (results rows, per-query rows) as DataFrames."""
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; choose from {SOURCES}")
    records = list(records)
    by_image: dict[int, list[ObjectRecord]] = defaultdict(list)
    for r in records:
        by_image[r.image_id].append(r)
    image_ids = sorted(by_image)[:max_images]
    record_ids = {r.obj_id for r in records}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in annotations["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    rows, queries, t0 = _Rows(run_id, boundary_all), [], time.time()
    match, match_impl = resolve_match()
    for n, image_id in enumerate(image_ids, 1):
        objs = by_image[image_id]
        h, w = objs[0].gt_mask["size"]
        gt = {r.obj_id: rle_decode(r.gt_mask) for r in objs}
        segmenter.set_image(image_id)

        if "gt_box" in sources:
            masks, pious, ms = _predict_boxes(segmenter, [r.gt_box for r in objs], "gt")
            for rec, mask, pi in zip(objs, masks, pious):
                common = dict(pred_iou=pi, det_score=NAN)
                rows.add(rec, "gt_box", "b0_none", 0, mask, gt[rec.obj_id], n_decoder_calls=1, latency_ms=ms, **common)
                if oracle_steps and "gt_box" in ORACLE_SOURCES:
                    _oracle_rows(segmenter, rows, rec, "gt_box", rec.gt_box, mask, gt[rec.obj_id], oracle_steps, common)

        wanted = [s for s in sources if s != "gt_box"]
        if not wanted:
            continue
        ignore = _ignore_masks(anns_by_image.get(image_id, []), record_ids, h, w)
        groups: dict[str, list[ObjectRecord]] = defaultdict(list)
        for r in objs:
            groups[r.category_name].append(r)
        for phrase, recs in groups.items():
            gts = [gt[r.obj_id] for r in recs]
            for source in wanted:
                dets = (grounding_dino if source == "grounding_dino_box" else sam3_text).get((image_id, phrase), [])
                if source == "sam3_text_mask":
                    pred_masks, pious, ms = [rle_decode(d["mask_rle"]) for d in dets], [NAN] * len(dets), NAN
                else:
                    pred_masks, pious, ms = _predict_boxes(segmenter, [d["box"] for d in dets], ORIGIN[source])
                if pred_masks:
                    res = match(pred_masks, gts, ignore_masks=ignore)
                    pairs = {g: p for p, g, _ in res.pairs}
                    n_fp, n_ign = len(res.unmatched_preds), len(res.ignored_preds)
                else:
                    pairs, n_fp, n_ign = {}, 0, 0
                queries.append(dict(run_id=run_id, image_id=image_id, phrase=phrase, source=source, n_gt=len(recs),
                                    n_pred=len(dets), n_matched=len(pairs), n_fp=n_fp, n_ignored=n_ign))
                for gi, rec in enumerate(recs):
                    common = dict(n_gt_same_class=len(recs), n_pred=len(dets), n_fp=n_fp)
                    if gi not in pairs:
                        rows.add(rec, source, "b0_none", 0, None, gts[gi], detected=False, **common)
                        continue
                    p = pairs[gi]
                    common.update(det_score=dets[p]["score"], pred_iou=pious[p])
                    n_calls = 0 if source == "sam3_text_mask" else 1
                    rows.add(rec, source, "b0_none", 0, pred_masks[p], gts[gi], n_decoder_calls=n_calls,
                             latency_ms=ms, **common)
                    if oracle_steps and source in ORACLE_SOURCES:
                        _oracle_rows(segmenter, rows, rec, source, dets[p]["box"], pred_masks[p], gts[gi],
                                     oracle_steps, common)
        if log_every and n % log_every == 0:
            print(f"[rq0] {n}/{len(image_ids)} images, {len(rows.rows)} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows.rows)
    df.attrs["match_impl"] = match_impl
    return df, pd.DataFrame(queries)
