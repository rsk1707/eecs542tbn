"""Sensitivity sweep and baselines-vs-oracle on perturbed GT boxes (demo items 3 and 4). Owner: P4.

For every object, the clean GT box plus one prompt per (type, level) of the grid (perturb/box.py) goes
through the segmenter (one batched call per image), giving `b0_none` step 0. Optionally every prompt is
then corrected for `steps` steps by each requested method: the rule baselines (baselines/rules.py) and
`oracle_clicks` (ideal clicks from the GT, the upper bound; same click rule as the RQ0 study).

One row per (object, prompt, method, step): ResultRow columns plus `pert_id` (index of the prompt within
the object), `pert_dir` (shift angle in degrees / truncated side), `clipped`, `n_points`, `pred_iou`,
`category`, `area`. `boundary_iou` is computed at step 0 only unless `boundary_all` (it is slow).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from promptcorrect.baselines.rules import METHODS as BASELINES
from promptcorrect.baselines.rules import Step, next_prompt, rollout
from promptcorrect.eval.methods import check_method
from promptcorrect.eval.metrics import boundary_iou, iou
from promptcorrect.eval.oracle_clicks import ideal_click as oracle_click
from promptcorrect.perturb.box import SWEEP_LEVELS, object_rng, sweep_prompts
from promptcorrect.perturb.severity import describe, severity_columns
from promptcorrect.types import ObjectRecord, Prompt

METHODS = BASELINES + ("oracle_clicks",)
NAN = float("nan")


def _oracle_rule(gt: np.ndarray):
    def rule(prompt: Prompt, logits, mask, rng):
        click = oracle_click(mask, gt)
        return None if click is None else next_prompt(prompt, "oracle_clicks", point=click)
    return rule


def _pert_fields(p: Prompt) -> dict:
    if p.perturbation is None:
        return dict(pert_type=None, pert_level=None, pert_dir=None, clipped=False)
    params = p.perturbation["params"]
    direction = (f"{math.degrees(params['angle']) % 360:.0f}" if "angle" in params else params.get("side"))
    return dict(pert_type=p.perturbation["type"], pert_level=p.perturbation["level"], pert_dir=direction,
                clipped=params["clipped"])


def run_sweep(segmenter, records, *, run_id: str, levels: dict = SWEEP_LEVELS, methods=("b0_none",), steps: int = 3,
              seed: int = 0, boundary_all: bool = False, boundary: bool = True, max_images: int | None = None,
              log_every: int = 10, time_budget_s: float | None = None) -> pd.DataFrame:
    """Run the sweep on the current records. Returns the long-format rows as a DataFrame.

    `time_budget_s`: before starting an image, stop if this many seconds have passed since the sweep began;
    the rows of the images done so far are returned (`df.attrs["stopped_early"]` is True), so a Slurm limit
    does not lose the whole run. `boundary=False` skips the step-0 Boundary IoU (no sweep summary uses it;
    it is about a third of the CPU work per row). `df.attrs["timing"]` holds the seconds spent per phase
    (SAM prediction, upsampling, metrics, correction rollouts), also printed with an ETA every `log_every` images.
    """
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}; choose from {METHODS}")
    by_image: dict[int, list[ObjectRecord]] = defaultdict(list)
    for r in records:
        by_image[r.image_id].append(r)
    image_ids = sorted(by_image)[:max_images]
    rows, t0 = [], time.time()
    timing = {"predict": 0.0, "to_full_res": 0.0, "metrics": 0.0, "rollout": 0.0}
    done, stopped = 0, False

    for n, image_id in enumerate(image_ids, 1):
        if time_budget_s is not None and time.time() - t0 >= time_budget_s:
            stopped = True
            print(f"[sweep] time budget of {time_budget_s:.0f}s reached after {done}/{len(image_ids)} images; "
                  "keeping what is done", flush=True)
            break
        objs = by_image[image_id]
        cache: dict[str, np.ndarray] = {}
        segmenter.set_image(image_id)
        prompts = [(rec, i, p) for rec in objs
                   for i, p in enumerate(sweep_prompts(rec.gt_box, tuple(rec.gt_mask["size"]),
                                                       object_rng(seed, rec.obj_id), levels))]
        t_a = time.perf_counter()
        outs = segmenter.predict([p for _, _, p in prompts])
        t_b = time.perf_counter()
        fulls = [segmenter.to_full_res(o.mask_logits).float().cpu().numpy() for o in outs]
        t_c = time.perf_counter()
        ms = (t_c - t_a) * 1000 / max(1, len(prompts))
        timing["predict"] += t_b - t_a
        timing["to_full_res"] += t_c - t_b

        t_m, roll_before = time.perf_counter(), timing["rollout"]
        for (rec, pid, prompt), out, full in zip(prompts, outs, fulls):
            sev = severity_columns(describe(prompt, rec, objs, mask_cache=cache))
            gt = cache[rec.obj_id]
            common = dict(run_id=run_id, obj_id=rec.obj_id, image_id=image_id, pert_id=pid,
                          prompt_origin=prompt.origin, **_pert_fields(prompt), **sev, category=rec.category_name,
                          novelty=rec.novelty, size_bin=rec.size_bin, thinness=rec.thinness, area=rec.area)
            s0 = Step(0, prompt, out, full > 0, ms, 1)
            b0 = boundary_iou(s0.mask, gt) if boundary else NAN
            for method in methods:
                if method == "b0_none":
                    traj = [s0]
                else:
                    rule = _oracle_rule(gt) if method == "oracle_clicks" else None
                    t_r = time.perf_counter()
                    traj = rollout(segmenter, method, prompt, steps, object_rng(seed, rec.obj_id, pid, method),
                                   first=(s0, full), rule=rule)
                    timing["rollout"] += time.perf_counter() - t_r
                for s in traj:
                    rows.append(dict(common, method=check_method(method), step=s.step, iou=iou(s.mask, gt),
                                     boundary_iou=b0 if s.step == 0 else (boundary_iou(s.mask, gt) if boundary_all else NAN),
                                     pred_iou=s.output.iou_score, n_points=len(s.prompt.points),
                                     n_decoder_calls=s.n_decoder_calls, latency_ms=s.latency_ms))
        timing["metrics"] += time.perf_counter() - t_m - (timing["rollout"] - roll_before)
        done = n
        if log_every and n % log_every == 0:
            el = time.time() - t0
            print(f"[sweep] {n}/{len(image_ids)} images, {len(rows)} rows, {el:.0f}s elapsed, "
                  f"ETA {el / n * (len(image_ids) - n):.0f}s | seconds: predict {timing['predict']:.0f}, "
                  f"upsample {timing['to_full_res']:.0f}, metrics {timing['metrics']:.0f}, "
                  f"rollout {timing['rollout']:.0f}", flush=True)
    df = pd.DataFrame(rows)
    df.attrs.update(timing={k: round(v, 1) for k, v in timing.items()}, images_done=done,
                    images_total=len(image_ids), stopped_early=stopped)
    return df


# ---- summaries (raw values in, tables out; docs/interfaces.md section 5) ---------------------------------

THRESHOLDS = (0.5, 0.75)
BOX_IOU_BINS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
HARM_EPS = 0.01  # harm = final IoU < step-0 IoU - 0.01 (docs/interfaces.md section 5)
MIN_BIN_N = 5  # by-box-IoU cells with fewer prompts than this are blanked in the summary and figure


def _step0(df: pd.DataFrame) -> pd.DataFrame:
    """One row per prompt before any correction (step 0 is the same mask for every method)."""
    g = df[df.step == 0].drop_duplicates(["obj_id", "pert_id"]).copy()
    g["pert_type"] = g.pert_type.fillna("none")
    g["pert_level"] = g.pert_level.astype(float)
    return g


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """No correction: mean IoU and failure rates per (type, level). The clean box is type "none"."""
    g = _step0(df)
    agg = {"n": ("iou", "size"), "mean_iou": ("iou", "mean"), "mean_box_iou": ("box_iou_in", "mean"),
           "mean_coverage": ("coverage", "mean")}
    for th in THRESHOLDS:
        g[f"fail<{th}_%"] = (g.iou < th) * 100.0
        agg[f"fail<{th}_%"] = (f"fail<{th}_%", "mean")
    return g.groupby(["pert_type", "pert_level"], dropna=False).agg(**agg).round(3)


def by_box_iou(df: pd.DataFrame, min_n: int = 1) -> pd.DataFrame:
    """No correction: mean mask IoU per bin of the prompt's box IoU with the GT box, per type (comparable
    across types, and with real detector boxes). Cells with fewer than `min_n` prompts are left empty
    (NaN): a mean over one or two prompts is noise, not a result."""
    g = _step0(df)
    g["box_iou_bin"] = pd.cut(g.box_iou_in, BOX_IOU_BINS, right=False)
    mean = g.pivot_table(index="box_iou_bin", columns="pert_type", values="iou", aggfunc="mean", observed=False)
    count = g.pivot_table(index="box_iou_bin", columns="pert_type", values="iou", aggfunc="count", observed=False)
    return mean.where(count >= min_n).round(3)


def trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Wide IoU per (method, obj_id, pert_id) x step; methods that stopped early keep their last IoU."""
    w = df.pivot_table(index=["method", "obj_id", "pert_id"], columns="step", values="iou", aggfunc="first")
    return w.ffill(axis=1)


def method_table(df: pd.DataFrame, which: str = "perturbed") -> pd.DataFrame:
    """Per method, over clean ("clean") or perturbed ("perturbed") prompts: mean IoU at each step, then at
    the final step the mean gain, harm rate, improved rate and success@0.75."""
    sel = df[df.pert_type.isna()] if which == "clean" else df[df.pert_type.notna()]
    w = trajectories(sel)
    if w.empty:
        return pd.DataFrame()
    m = w.groupby(level="method", sort=False)
    t = m.mean()
    t.columns = [f"iou@{c}" for c in t.columns]
    first, final = w[0], w.iloc[:, -1]
    stats = pd.DataFrame({"n": m.size(), "delta_iou": (final - first).groupby(level="method").mean(),
                          "harm_%": ((final < first - HARM_EPS) * 100.0).groupby(level="method").mean(),
                          "improved_%": ((final > first + HARM_EPS) * 100.0).groupby(level="method").mean(),
                          "success@0.75_%": ((final >= 0.75) * 100.0).groupby(level="method").mean()})
    order = [mm for mm in METHODS if mm in t.index]
    return stats.join(t).loc[order].round(3)


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {"sensitivity": sensitivity(df), "iou_by_box_iou": by_box_iou(df, min_n=MIN_BIN_N)}
    if df.method.nunique() > 1 or df.step.max() > 0:
        tables["methods_perturbed"] = method_table(df, "perturbed")
        tables["methods_clean"] = method_table(df, "clean")
    return {k: v for k, v in tables.items() if not v.empty}


def format_report(tables: dict[str, pd.DataFrame]) -> str:
    return "\n".join(f"== {name} ==\n{t.to_string()}\n" for name, t in tables.items())
