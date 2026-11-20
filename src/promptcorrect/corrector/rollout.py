"""End-to-end check of corrector v0 with the frozen segmenter: does clicking where it predicts an error help,
and does it leave good masks alone? Owner: P5.

Every object's clean GT box goes through the segmenter (step 0, the `b0_none` row), then `corrector_v<n>`
gets up to `steps` corrections: crop -> predicted error map -> one click (clicks.py) -> segmenter again with
the box and all clicks so far. If the corrector abstains it stops, and its final mask is the step-0 mask.

Only code that exists in the repo is used. Baselines (P4's `baselines/`), ideal clicks (P3's simulator) and
perturbed prompts (P4's `perturb/`) are added here by importing the owners' code once it lands.
v0 is trained on step-0 states only (empty click channels), so `steps=1` is the in-distribution setting.

Rows are long format (docs/interfaces.md section 5), one per (obj_id, method, step). A method that stops
early has no rows after its last step: final IoU = the row with the highest step.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from promptcorrect.corrector.clicks import propose_click
from promptcorrect.corrector.crops import box_mask, clicks_to_crop, crop_object, crop_to_image, crop_window
from promptcorrect.corrector.model import build_inputs
from promptcorrect.corrector.samples import MAX_CLICKS
from promptcorrect.data.geometry import rle_decode
from promptcorrect.data.records import group_by_image
from promptcorrect.eval.methods import check_method
from promptcorrect.eval.metrics import boundary_iou, iou
from promptcorrect.types import Prompt

NAN = float("nan")


class _Corrector:
    def __init__(self, model, device, crop_size: int, context: float, threshold: float):
        self.model, self.device = model, device
        self.crop_size, self.context, self.threshold = crop_size, context, threshold

    @torch.no_grad()
    def click(self, image_t, logits_full, box, points):
        S = self.crop_size
        window = crop_window(box, self.context)
        c = crop_object(image_t, logits_full, None, window, S)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None].to(self.device)  # noqa: E731
        x = build_inputs(t(c["rgb"]), t(c["logits"]), t(box_mask(box, window, S)), t(c["valid"]),
                         t(clicks_to_crop(points, window, S, MAX_CLICKS)))
        probs = torch.sigmoid(self.model(x).float())[0].cpu().numpy()
        click = propose_click(probs, self.threshold)
        if click is None:
            return None
        u, v, label = click
        x_img, y_img = crop_to_image(u, v, window, S)
        H, W = image_t.shape[-2:]
        return float(np.clip(x_img, 0, W - 1e-3)), float(np.clip(y_img, 0, H - 1e-3)), label


def run_rollouts(segmenter, model, records, load_image, *, run_id: str, method: str = "corrector_v0",
                 steps: int = 1, crop_size: int = 128, context: float = 0.5, threshold: float = 0.5,
                 device="cpu", max_images: int | None = None, boundary_all: bool = False,
                 log_every: int = 25) -> pd.DataFrame:
    check_method(method)
    corr = _Corrector(model, device, crop_size, context, threshold)
    by_image = group_by_image(records)
    image_ids = sorted(by_image)[:max_images]
    rows, t0 = [], time.time()

    def add(rec, meth, step, scores, **extra):
        rows.append(dict(
            run_id=run_id, obj_id=rec.obj_id, image_id=rec.image_id, method=meth, step=step,
            iou=scores[0], boundary_iou=scores[1], prompt_origin="gt", novelty=rec.novelty,
            size_bin=rec.size_bin, thinness=rec.thinness, category=rec.category_name, **extra))

    for n, image_id in enumerate(image_ids, 1):
        objs = by_image[image_id]
        image_t = torch.from_numpy(np.array(load_image(image_id))).permute(2, 0, 1).contiguous()
        segmenter.set_image(image_id)
        t = time.perf_counter()
        outs0 = segmenter.predict([Prompt(box=list(r.gt_box), origin="gt") for r in objs])
        ms0 = (time.perf_counter() - t) * 1000 / max(1, len(objs))
        for rec, out0 in zip(objs, outs0):
            gt = rle_decode(rec.gt_mask)
            logits = segmenter.to_full_res(out0.mask_logits)
            mask = (logits > 0).numpy()
            common = dict(n_points=0, n_decoder_calls=1, latency_ms=ms0, pred_iou=out0.iou_score, click_label=-1)
            scores0 = (iou(mask, gt), boundary_iou(mask, gt))  # step 0 is shared by both methods
            add(rec, "b0_none", 0, scores0, **common)
            add(rec, method, 0, scores0, **common)
            points = []
            for s in range(1, steps + 1):
                t = time.perf_counter()
                click = corr.click(image_t, logits, rec.gt_box, points)
                if click is None:
                    break
                points.append(click)
                out = segmenter.predict([Prompt(box=list(rec.gt_box), points=list(points), origin="corrector")])[0]
                logits = segmenter.to_full_res(out.mask_logits)
                mask = (logits > 0).numpy()
                add(rec, method, s, (iou(mask, gt), boundary_iou(mask, gt) if boundary_all else NAN),
                    n_points=len(points), n_decoder_calls=1 + s, latency_ms=(time.perf_counter() - t) * 1000,
                    pred_iou=out.iou_score, click_label=click[2])
        if log_every and n % log_every == 0:
            print(f"[rollout] {n}/{len(image_ids)} images, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (obj, method): step-0 IoU, final IoU, steps used, first click's label."""
    key = ["obj_id", "method"]
    first = df[df.step == 0].set_index(key)
    last = df.sort_values("step").groupby(key).tail(1).set_index(key)
    out = first[["size_bin", "iou"]].rename(columns={"iou": "iou0"})
    out["iou_final"] = last["iou"]
    out["steps_used"] = last["step"]
    out["first_click"] = df[df.step == 1].set_index(key)["click_label"].reindex(out.index)
    out["delta"] = out.iou_final - out.iou0
    return out.reset_index()


def summarize(df: pd.DataFrame, steps: int) -> dict[str, pd.DataFrame]:
    f = final_rows(df)
    f["harmed"] = f.iou_final < f.iou0 - 0.01        # docs/interfaces.md section 5
    f["improved"] = f.iou_final > f.iou0 + 0.01
    f["group"] = "all"
    groups = pd.concat([f, f[f.iou0 < 0.75].assign(group="start<0.75"), f[f.iou0 >= 0.75].assign(group="start>=0.75"),
                        f.assign(group="size:" + f.size_bin)])

    def agg(g: pd.DataFrame) -> pd.DataFrame:
        return g.groupby(["group", "method"]).agg(
            n=("iou0", "size"), iou0=("iou0", "mean"), iou_final=("iou_final", "mean"), delta=("delta", "mean"),
            harm_pct=("harmed", lambda s: 100 * s.mean()), improved_pct=("improved", lambda s: 100 * s.mean()),
            clicked_pct=("steps_used", lambda s: 100 * (s > 0).mean()),
            pos_click_pct=("first_click", lambda s: 100 * (s == 1).mean()),
            succ50_0_pct=("iou0", lambda s: 100 * (s >= 0.5).mean()),
            succ50_final_pct=("iou_final", lambda s: 100 * (s >= 0.5).mean()),
            succ75_0_pct=("iou0", lambda s: 100 * (s >= 0.75).mean()),
            succ75_final_pct=("iou_final", lambda s: 100 * (s >= 0.75).mean())).round(4)

    wide = df.pivot_table(index=["obj_id", "method"], columns="step", values="iou")
    wide = wide.reindex(columns=range(steps + 1)).ffill(axis=1)
    return {"final": agg(groups), "iou_by_step": wide.groupby(level="method").mean().round(4)}


def format_report(tables: dict[str, pd.DataFrame]) -> str:
    return "\n\n".join(f"== {k}\n{v.to_string()}" for k, v in tables.items())
