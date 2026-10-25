"""End-to-end check of corrector v0 against the simple baselines and the ideal-click upper bound. Owner: P5.

For each object, eval prompts are the clean GT box (RQ4: does the corrector leave it alone?) plus one
perturbed box per perturbation type at a level drawn from `levels` (stand-in perturbations, standins.py).
Every prompt gets the same step-0 mask, then up to `steps` corrections per method:

  corrector_v<n>  click where the corrector's predicted error is (clicks.py); abstain -> stop
  b1_random_k     step k = k random positive clicks inside the prompt box     (stand-in for P4's)
  b3_box_refit    refit the box to the current mask and rerun; stop when the box no longer changes (stand-in)
  oracle_clicks   ideal clicks from GT (experiments/rq0.py `oracle_click`), upper bound

Clicks always go to the segmenter together with the prompt box, as in the RQ0 oracle. Rows are long format
(docs/interfaces.md section 5) with an extra `prompt_id` column, because one object has several prompts.
A method that stops early has no rows after its last step: final IoU = the row with the highest step.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from promptcorrect.corrector.clicks import propose_click
from promptcorrect.corrector.crops import box_mask, clicks_to_crop, crop_object, crop_to_image, crop_window
from promptcorrect.corrector.model import build_inputs
from promptcorrect.corrector.samples import MAX_CLICKS
from promptcorrect.corrector.standins import PERTURBATIONS, box_iou, perturb_box, random_click
from promptcorrect.data.geometry import mask_to_box, rle_decode
from promptcorrect.data.records import group_by_image
from promptcorrect.eval.methods import check_method
from promptcorrect.eval.metrics import boundary_iou, iou
from promptcorrect.experiments.rq0 import oracle_click
from promptcorrect.types import Prompt

NAN = float("nan")
BASELINES = ("b1_random_k", "b3_box_refit", "oracle_clicks")


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


def eval_prompts(records, rng: np.random.Generator, levels, hw_of) -> list[tuple]:
    """-> [(record, prompt_id, box, perturbation or None)]; deterministic given rng."""
    out = []
    for rec in records:
        out.append((rec, "clean", list(rec.gt_box), None))
        for kind in PERTURBATIONS:
            level = float(levels[int(rng.integers(len(levels)))])
            box, pert = perturb_box(rec.gt_box, kind, level, rng, hw_of(rec))
            out.append((rec, f"{kind}@{level:g}", box, pert))
    return out


def run_rollouts(segmenter, model, records, load_image, *, run_id: str, method: str = "corrector_v0",
                 baselines=BASELINES, steps: int = 3, levels=(0.1, 0.2, 0.3, 0.4), crop_size: int = 128,
                 context: float = 0.5, threshold: float = 0.5, device="cpu", seed: int = 0,
                 max_images: int | None = None, boundary_all: bool = False, log_every: int = 25) -> pd.DataFrame:
    check_method(method)
    for b in baselines:
        check_method(b)
    rng = np.random.default_rng(seed)               # eval prompts: identical whichever methods run
    click_rng = np.random.default_rng(seed + 1)     # b1_random_k
    corr = _Corrector(model, device, crop_size, context, threshold) if model is not None else None
    by_image = group_by_image(records)
    image_ids = sorted(by_image)[:max_images]
    rows, t0 = [], time.time()

    def add(rec, pid, box_in, pert, meth, step, mask, gt, scores=None, **extra):
        if scores is None:
            scores = (iou(mask, gt), boundary_iou(mask, gt) if (step == 0 or boundary_all) else NAN)
        rows.append(dict(
            run_id=run_id, obj_id=rec.obj_id, image_id=rec.image_id, method=meth, step=step, prompt_id=pid,
            iou=scores[0], boundary_iou=scores[1],
            prompt_origin="perturbed" if pert else "gt", pert_type=pert["type"] if pert else None,
            pert_level=pert["params"]["level"] if pert else 0.0, box_iou_in=box_iou(box_in, rec.gt_box),
            novelty=rec.novelty, size_bin=rec.size_bin, thinness=rec.thinness, category=rec.category_name,
            **extra))

    for n, image_id in enumerate(image_ids, 1):
        objs = by_image[image_id]
        image = load_image(image_id)
        H, W = image.shape[:2]
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        gts = {r.obj_id: rle_decode(r.gt_mask) for r in objs}
        segmenter.set_image(image_id)
        prompts = eval_prompts(objs, rng, levels, lambda r: (H, W))

        t = time.perf_counter()
        outs0 = segmenter.predict([Prompt(box=b, origin="perturbed" if p else "gt", perturbation=p) for _, _, b, p in prompts])
        ms0 = (time.perf_counter() - t) * 1000 / max(1, len(prompts))
        for (rec, pid, box, pert), out0 in zip(prompts, outs0):
            gt = gts[rec.obj_id]
            logits0 = segmenter.to_full_res(out0.mask_logits)
            mask0 = (logits0 > 0).numpy()
            common = dict(n_points=0, n_decoder_calls=1, latency_ms=ms0, pred_iou=out0.iou_score,
                          scores=(iou(mask0, gt), boundary_iou(mask0, gt)))  # step 0 is shared by all methods
            add(rec, pid, box, pert, "b0_none", 0, mask0, gt, **common)

            for meth in ([method] if corr is not None else []) + list(baselines):
                add(rec, pid, box, pert, meth, 0, mask0, gt, **common)
                cur_box, points, logits, mask = list(box), [], logits0, mask0
                for s in range(1, steps + 1):
                    t = time.perf_counter()
                    if meth == method:
                        click = corr.click(image_t, logits, box, points)
                        if click is None:
                            break
                        points.append(click)
                    elif meth == "b1_random_k":
                        points = [random_click(box, click_rng) for _ in range(s)]
                    elif meth == "oracle_clicks":
                        click = oracle_click(mask, gt)
                        if click is None:
                            break
                        points.append(click)
                    elif meth == "b3_box_refit":
                        new_box = mask_to_box(mask) if mask.any() else cur_box
                        if max(abs(a - b) for a, b in zip(new_box, cur_box)) < 0.5:
                            break
                        cur_box = new_box
                    else:
                        raise ValueError(meth)
                    out = segmenter.predict([Prompt(box=cur_box, points=list(points), origin=meth)])[0]
                    logits = segmenter.to_full_res(out.mask_logits)
                    mask = (logits > 0).numpy()
                    add(rec, pid, box, pert, meth, s, mask, gt, n_points=len(points), n_decoder_calls=1 + s,
                        latency_ms=(time.perf_counter() - t) * 1000, pred_iou=out.iou_score)
        if log_every and n % log_every == 0:
            print(f"[rollout] {n}/{len(image_ids)} images, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (obj, prompt, method): step-0 IoU, final IoU, steps used."""
    key = ["obj_id", "prompt_id", "method"]
    first = df[df.step == 0].set_index(key)
    last = df.sort_values("step").groupby(key).tail(1).set_index(key)
    out = first[["prompt_origin", "pert_type", "pert_level", "box_iou_in", "size_bin", "iou"]].rename(columns={"iou": "iou0"})
    out["iou_final"] = last["iou"]
    out["steps_used"] = last["step"]
    out["delta"] = out.iou_final - out.iou0
    return out.reset_index()


def iou_by_step(df: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Mean IoU at each step 0..steps with the last value carried forward after a method stops."""
    wide = df.pivot_table(index=["obj_id", "prompt_id", "method"], columns="step", values="iou")
    wide = wide.reindex(columns=range(steps + 1)).ffill(axis=1)
    return wide


def summarize(df: pd.DataFrame, steps: int) -> dict[str, pd.DataFrame]:
    f = final_rows(df)
    f["harmed"] = f.iou_final < f.iou0 - 0.01        # docs/interfaces.md section 5
    f["improved"] = f.iou_final > f.iou0 + 0.01
    groups = {"all": f, "clean": f[f.prompt_origin == "gt"], "perturbed": f[f.prompt_origin == "perturbed"],
              "failures<0.75": f[f.iou0 < 0.75]}

    def agg(g: pd.DataFrame) -> pd.DataFrame:
        return g.groupby("method").agg(
            n=("iou0", "size"), iou0=("iou0", "mean"), iou_final=("iou_final", "mean"), delta=("delta", "mean"),
            harm_pct=("harmed", lambda s: 100 * s.mean()), improved_pct=("improved", lambda s: 100 * s.mean()),
            succ75_0_pct=("iou0", lambda s: 100 * (s >= 0.75).mean()),
            succ75_final_pct=("iou_final", lambda s: 100 * (s >= 0.75).mean()),
            succ50_final_pct=("iou_final", lambda s: 100 * (s >= 0.5).mean()),
            mean_steps=("steps_used", "mean")).round(4)

    tables = {f"final_{k}": agg(g) for k, g in groups.items()}
    wide = iou_by_step(df, steps)
    for name, sel in (("all", None), ("perturbed", "perturbed"), ("failures<0.75", "fail")):
        w = wide
        if sel == "perturbed":
            keep = set(map(tuple, f.loc[f.prompt_origin == "perturbed", ["obj_id", "prompt_id"]].values))
            w = wide[[(o, p) in keep for o, p, _ in wide.index]]
        elif sel == "fail":
            w = wide[wide[0] < 0.75]
        tables[f"iou_by_step_{name}"] = w.groupby(level="method").mean().round(4)
    pert = f[f.prompt_origin == "perturbed"]
    tables["delta_by_type_level"] = pert.pivot_table(index=["pert_type", "pert_level"], columns="method",
                                                     values="delta", aggfunc="mean").round(4)
    return tables


def format_report(tables: dict[str, pd.DataFrame]) -> str:
    return "\n\n".join(f"== {k}\n{v.to_string()}" for k, v in tables.items())
