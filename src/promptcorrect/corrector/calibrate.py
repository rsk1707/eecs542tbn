"""Choose the v1 click-or-abstain thresholds on held-out TRAINING images with the real segmenter. Owner: P5.

For every held-out training sample (the image-level holdout of `split_by_image`, so the same images the
checkpoint was selected on and never the val objects) the sample's own prompt box goes through the segmenter
(step 0), the corrector gives its predicted IoU `q` and a candidate click for each error threshold, and the
segmenter is called again with that click, with and without logit feedback. That gives the *measured* IoU
gain of every candidate click.

`choose_thresholds` then sweeps the quality threshold tau (click only if q < tau) and keeps the (error
threshold, tau) pair with the largest mean ΔIoU over all held-out samples subject to harm on good masks
(IoU at step 0 >= `good_iou`, harmed = final < start - 0.01, as in docs/interfaces.md section 5) at most
`max_harm`. tau = 0 (never click) is always feasible, so a corrector that cannot beat its own harm is told to
abstain everywhere rather than being tuned on val.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from promptcorrect.corrector.policy import CorrectorPolicy
from promptcorrect.data.geometry import rle_decode
from promptcorrect.eval.metrics import iou
from promptcorrect.types import ObjectRecord, Prompt

TAUS = tuple(np.round(np.linspace(0.0, 1.0, 101), 2).tolist()) + (float("inf"),)


def run_calibration(segmenter, policy: CorrectorPolicy, metas: list[dict], records: dict[str, ObjectRecord],
                    load_image, *, err_thresholds=(0.3, 0.5, 0.7), feedback_modes=(True, False),
                    max_images: int | None = None, log_every: int = 25) -> pd.DataFrame:
    """`metas`: sample meta rows (meta.jsonl) to replay; `records`: obj_id -> ObjectRecord (the samples run's
    records.jsonl). Returns one row per (sample, err_threshold, feedback): iou0, q, has_click, iou1."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for m in metas:
        by_image[m["image_id"]].append(m)
    image_ids = sorted(by_image)[:max_images]
    rows, t0 = [], time.time()
    for n, image_id in enumerate(image_ids, 1):
        ms = by_image[image_id]
        image_t = torch.from_numpy(np.array(load_image(image_id))).permute(2, 0, 1).contiguous()
        segmenter.set_image(image_id)
        prompts = [Prompt(box=list(m["box"]), origin=m.get("origin", "gt")) for m in ms]
        outs = segmenter.predict(prompts)
        gts: dict[str, np.ndarray] = {}
        for m, prompt, out in zip(ms, prompts, outs):
            rec = records[m["obj_id"]]
            if rec.obj_id not in gts:
                gts[rec.obj_id] = rle_decode(rec.gt_mask)
            gt = gts[rec.obj_id]
            logits = segmenter.to_full_res(out.mask_logits)
            iou0 = iou((logits > 0).numpy(), gt)
            view = policy.view(image_t, logits, prompt.box)
            cache: dict[tuple, float] = {}  # (click, feedback) -> iou1: thresholds often give the same click
            for thr in err_thresholds:
                click = policy.candidate(view, thr)
                for fb in feedback_modes:
                    iou1 = float("nan")
                    if click is not None:
                        key = (click, fb)
                        if key not in cache:
                            p = Prompt(box=list(prompt.box), points=[click], origin="corrector",
                                       mask_logits=out.mask_logits if fb else None)
                            o1 = segmenter.predict([p])[0]
                            cache[key] = iou(segmenter.full_res_mask(o1), gt)
                        iou1 = cache[key]
                    rows.append(dict(sample=m.get("sample"), obj_id=rec.obj_id, image_id=image_id,
                                     origin=m.get("origin"), pert_type=m.get("pert_type"), size_bin=rec.size_bin,
                                     iou0=iou0, q=view.q, err_threshold=thr, feedback=fb,
                                     has_click=click is not None, click_label=-1 if click is None else click[2],
                                     iou1=iou1))
        if log_every and n % log_every == 0:
            print(f"[calib] {n}/{len(image_ids)} images, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def threshold_curve(df: pd.DataFrame, feedback: bool, *, good_iou: float = 0.75, harm_eps: float = 0.01,
                    taus=TAUS) -> pd.DataFrame:
    """Outcome of every (err_threshold, tau) on the calibration rows for one feedback setting."""
    out = []
    for thr, sub in df[df.feedback == feedback].groupby("err_threshold"):
        iou0, q = sub.iou0.to_numpy(), sub.q.to_numpy()
        has, iou1 = sub.has_click.to_numpy(bool), sub.iou1.to_numpy()
        good, fail = iou0 >= good_iou, iou0 < good_iou
        for tau in taus:
            clicked = has & (np.isnan(q) | (q < tau))
            final = np.where(clicked, iou1, iou0)
            harmed, improved = final < iou0 - harm_eps, final > iou0 + harm_eps
            d = final - iou0
            out.append(dict(
                err_threshold=thr, tau=tau, n=len(sub), n_fail=int(fail.sum()), n_good=int(good.sum()),
                clicked_pct=100 * clicked.mean(), delta_all=d.mean(),
                delta_fail=d[fail].mean() if fail.any() else np.nan,
                delta_good=d[good].mean() if good.any() else np.nan,
                harm_good_pct=100 * harmed[good].mean() if good.any() else np.nan,
                harm_all_pct=100 * harmed.mean(), improved_fail_pct=100 * improved[fail].mean() if fail.any() else np.nan,
                clicked_fail_pct=100 * clicked[fail].mean() if fail.any() else np.nan,
                clicked_good_pct=100 * clicked[good].mean() if good.any() else np.nan))
    return pd.DataFrame(out)


def choose_thresholds(curve: pd.DataFrame, max_harm: float = 0.05) -> dict:
    """Largest mean ΔIoU with harm on good masks <= max_harm (fraction); ties -> fewer clicks."""
    ok = curve[curve.harm_good_pct.fillna(0) <= 100 * max_harm]
    if ok.empty:  # cannot happen with tau = 0 in the grid, but be explicit
        raise ValueError("no feasible threshold; include tau = 0")
    best = ok.sort_values(["delta_all", "clicked_pct"], ascending=[False, True]).iloc[0]
    res = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (np.integer,)) else v)
           for k, v in best.items()}
    res["tau"] = None if not np.isfinite(res["tau"]) else res["tau"]  # None = no quality gate
    res["max_harm"] = max_harm
    return res
