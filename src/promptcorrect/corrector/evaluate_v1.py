"""Corrector v1 end to end vs the baselines and ideal clicks, on the same objects and step-0 masks. Owner: P5.

Every object gets its clean GT box (prompt_origin "gt") and `n_perturbed` random shift / scale / truncate boxes
from P4's `sample_perturbation` (prompt_origin "perturbed", drawn from `object_rng(seed, obj_id, "eval")` so they
are independent of the training draws and of image order). Each prompt goes through the segmenter once
(step 0, shared by all methods), then each method gets up to `steps` correction steps:

  b0_none                         step 0 only
  corrector_v1 (and v0, ...)      CorrectorPolicy: error-map click if the quality gate allows, else stop
  b1_random_k ... b3_unc_click    P4's rules (baselines/rules.py), no logit feedback (as P4 defines them)
  oracle_clicks                   P3's ideal clicks (eval/oracle_clicks.py), with `oracle_feedback`

Rows are P3's `ResultRow`s (one per obj_id, prompt_origin, pert_type, pert_level, method, step) with P4's
`describe()` severity, written with P3's `write_results`. A method that stops has no rows after its last step.
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd
import torch

from promptcorrect.baselines.rules import Step, rollout
from promptcorrect.corrector.policy import CorrectorPolicy
from promptcorrect.data.geometry import rle_decode
from promptcorrect.data.records import group_by_image
from promptcorrect.eval.methods import check_method
from promptcorrect.eval.metrics import boundary_iou, iou
from promptcorrect.eval.oracle_clicks import run_oracle
from promptcorrect.perturb.box import TYPES, object_rng, perturbation_fields, sample_perturbation
from promptcorrect.perturb.severity import describe, severity_columns
from promptcorrect.types import ObjectRecord, Prompt, ResultRow

NAN = float("nan")
BASELINES = ("b1_random_k", "b2_multimask", "b3_box_refit", "b3_unc_click")
RULE_BASELINES = BASELINES  # names used in the go/no-go comparison


def eval_prompts(rec: ObjectRecord, *, seed: int = 0, n_perturbed: int = 2, include_clean: bool = True,
                 types: tuple[str, ...] = TYPES) -> list[Prompt]:
    hw = tuple(rec.gt_mask["size"])
    rng = object_rng(seed, rec.obj_id, "eval")
    out = [Prompt(box=[float(v) for v in rec.gt_box], origin="gt")] if include_clean else []
    return out + [sample_perturbation(rec.gt_box, hw, rng, types) for _ in range(n_perturbed)]


def run_eval_v1(segmenter, records: list[ObjectRecord], load_image, *, run_id: str,
                policies: dict[str, CorrectorPolicy], baselines=BASELINES, oracle: bool = True,
                oracle_feedback: bool = True, steps: int = 1, n_perturbed: int = 2, include_clean: bool = True,
                seed: int = 0, max_images: int | None = None, boundary_all: bool = False,
                log_every: int = 25) -> list[ResultRow]:
    for m in [*policies, *baselines]:
        check_method(m)
    by_image = group_by_image(records)
    image_ids = sorted(by_image)[:max_images]
    rows: list[ResultRow] = []
    t0 = time.time()

    for n, image_id in enumerate(image_ids, 1):
        objs = by_image[image_id]
        image_t = torch.from_numpy(np.array(load_image(image_id))).permute(2, 0, 1).contiguous()
        segmenter.set_image(image_id)
        gts = {r.obj_id: rle_decode(r.gt_mask) for r in objs}
        jobs = [(rec, p) for rec in objs for p in eval_prompts(rec, seed=seed, n_perturbed=n_perturbed,
                                                               include_clean=include_clean)]
        t = time.perf_counter()
        outs0 = segmenter.predict([p for _, p in jobs])
        ms0 = (time.perf_counter() - t) * 1000 / max(1, len(jobs))

        for (rec, prompt), out0 in zip(jobs, outs0):
            gt = gts[rec.obj_id]
            pf = perturbation_fields(prompt)
            sev = severity_columns(describe(prompt, rec, objs, mask_cache=gts))
            base = dict(run_id=run_id, obj_id=rec.obj_id, image_id=rec.image_id, prompt_origin=prompt.origin,
                        pert_type=pf["pert_type"], pert_level=pf["pert_level"], novelty=rec.novelty,
                        size_bin=rec.size_bin, thinness=rec.thinness, **sev)

            def add(method, step, i, b=NAN, **kw):
                rows.append(ResultRow(method=method, step=step, iou=i, boundary_iou=b, **base, **kw))

            logits0_t = segmenter.to_full_res(out0.mask_logits)
            logits0 = logits0_t.float().cpu().numpy()
            mask0 = logits0 > 0
            i0, b0 = iou(mask0, gt), boundary_iou(mask0, gt)
            step0 = dict(n_points=0, n_decoder_calls=1, latency_ms=ms0, pred_iou=out0.iou_score)
            add("b0_none", 0, i0, b0, **step0)

            for method, pol in policies.items():
                add(method, 0, i0, b0, **step0)
                out, logits_t, points = out0, logits0_t, []
                for s in range(1, steps + 1):
                    t = time.perf_counter()
                    click = pol.decide(pol.view(image_t, logits_t, prompt.box, points))
                    if click is None:
                        break
                    points.append(click)
                    out = segmenter.predict([pol.next_prompt(prompt, points, out)])[0]
                    logits_t = segmenter.to_full_res(out.mask_logits)
                    mask = (logits_t > 0).numpy()
                    add(method, s, iou(mask, gt), boundary_iou(mask, gt) if boundary_all else NAN,
                        n_points=len(points), n_decoder_calls=1 + s,
                        latency_ms=(time.perf_counter() - t) * 1000, pred_iou=out.iou_score)

            key = f"{rec.obj_id}|{prompt.origin}|{pf['pert_type']}|{pf['pert_level']}"
            for method in baselines:
                rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
                trace = rollout(segmenter, method, prompt, steps, rng,
                                first=(Step(0, prompt, out0, mask0, ms0, 1), logits0))
                add(method, 0, i0, b0, **step0)
                for st in trace[1:]:
                    add(method, st.step, iou(st.mask, gt), boundary_iou(st.mask, gt) if boundary_all else NAN,
                        n_points=len(st.prompt.points), n_decoder_calls=st.n_decoder_calls,
                        latency_ms=st.latency_ms, pred_iou=st.output.iou_score)

            if oracle:
                tr = run_oracle(segmenter, prompt, gt, steps, start=out0, start_ms=ms0, use_prev_mask=oracle_feedback)
                for k, (i, pi, lat) in enumerate(zip(tr.ious, tr.pred_ious, tr.latency_ms)):
                    add("oracle_clicks", k, i, b0 if k == 0 else NAN, n_points=k, n_decoder_calls=1 + k,
                        latency_ms=lat, pred_iou=pi)
        if log_every and n % log_every == 0:
            el = time.time() - t0
            print(f"[eval_v1] {n}/{len(image_ids)} images, {len(rows)} rows, {el:.0f}s "
                  f"(ETA {el / n * (len(image_ids) - n):.0f}s)", flush=True)
    return rows


# -- summary ---------------------------------------------------------------------------------------------

_PROMPT_KEY = ["obj_id", "prompt_origin", "pert_type", "pert_level"]


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (prompt, method): iou0, iou_final (last step), steps_used."""
    key = _PROMPT_KEY + ["method"]
    d = df.copy()
    d["pert_type"] = d["pert_type"].astype(object).where(d["pert_type"].notna(), "none")
    d["pert_level"] = d["pert_level"].fillna(-1.0)
    first = d[d.step == 0].set_index(key)
    last = d.sort_values("step").groupby(key).tail(1).set_index(key)
    out = first[["size_bin", "iou"]].rename(columns={"iou": "iou0"})
    out["iou_final"] = last["iou"]
    out["steps_used"] = last["step"]
    out["delta"] = out.iou_final - out.iou0
    return out.reset_index()


def summary_table(df: pd.DataFrame, good_iou: float = 0.75, harm_eps: float = 0.01) -> pd.DataFrame:
    """The RQ1/RQ4 table (docs/next_steps.md section 2): per (source, start group, method)."""
    f = final_rows(df)
    f["harmed"] = f.iou_final < f.iou0 - harm_eps
    f["improved"] = f.iou_final > f.iou0 + harm_eps
    parts = []
    for src, fs in (("all", f), ("gt", f[f.prompt_origin == "gt"]), ("perturbed", f[f.prompt_origin == "perturbed"])):
        for grp, g in (("all", fs), (f"start<{good_iou}", fs[fs.iou0 < good_iou]),
                       (f"start>={good_iou}", fs[fs.iou0 >= good_iou])):
            if len(g):
                parts.append(g.assign(source=src, start=grp))
        for sb, g in fs.groupby("size_bin"):
            parts.append(g.assign(source=src, start=f"size:{sb}"))
    allg = pd.concat(parts)
    return allg.groupby(["source", "start", "method"]).agg(
        n=("iou0", "size"), iou0=("iou0", "mean"), iou_final=("iou_final", "mean"), delta=("delta", "mean"),
        below75_0_pct=("iou0", lambda s: 100 * (s < good_iou).mean()),
        below75_final_pct=("iou_final", lambda s: 100 * (s < good_iou).mean()),
        harm_pct=("harmed", lambda s: 100 * s.mean()), improved_pct=("improved", lambda s: 100 * s.mean()),
        clicked_pct=("steps_used", lambda s: 100 * (s > 0).mean())).round(4)


def go_no_go(table: pd.DataFrame, method: str, good_iou: float = 0.75, max_harm_pct: float = 5.0) -> pd.DataFrame:
    """docs/next_steps.md section 2: ΔIoU on starts < good_iou beats the best rule baseline, and harm on starts
    >= good_iou is below max_harm_pct. One row per prompt source."""
    out = []
    for src in table.index.get_level_values("source").unique():
        try:
            fail = table.loc[(src, f"start<{good_iou}")]
            good = table.loc[(src, f"start>={good_iou}")]
        except KeyError:
            continue
        if method not in fail.index:
            continue
        rules = [m for m in RULE_BASELINES if m in fail.index]
        best_rule = fail.loc[rules, "delta"].idxmax() if rules else None
        d, h = float(fail.loc[method, "delta"]), float(good.loc[method, "harm_pct"]) if method in good.index else 0.0
        out.append(dict(source=src, method=method, delta_fail=d,
                        best_rule=best_rule, best_rule_delta=float(fail.loc[best_rule, "delta"]) if best_rule else NAN,
                        oracle_delta=float(fail.loc["oracle_clicks", "delta"]) if "oracle_clicks" in fail.index else NAN,
                        harm_good_pct=h, beats_rules=bool(best_rule is None or d > fail.loc[best_rule, "delta"]),
                        harm_ok=h < max_harm_pct))
    res = pd.DataFrame(out)
    if len(res):
        res["go"] = res.beats_rules & res.harm_ok
    return res


def iou_by_step(df: pd.DataFrame, steps: int) -> pd.DataFrame:
    d = df.copy()
    d["pert_level"] = d["pert_level"].fillna(-1.0)
    d["pert_type"] = d["pert_type"].astype(object).where(d["pert_type"].notna(), "none")
    wide = d.pivot_table(index=_PROMPT_KEY + ["method"], columns="step", values="iou")
    wide = wide.reindex(columns=range(steps + 1)).ffill(axis=1)
    return wide.groupby(level=["prompt_origin", "method"]).mean().round(4)


def load_eval_records(paths: str, records: str, annotations: str, split: str, out_dir) -> tuple[list[ObjectRecord], object, str]:
    """The demo val objects. If the shared records file is unreadable (it was mode 600 once), rebuild it with
    P2's builder from the same inputs into `out_dir` (the sha256 in the manifest shows whether it matches).
    Returns (records, file used, source description)."""
    import json

    from promptcorrect.data.prepare import rooted_path
    from promptcorrect.data.records import build_coco_records, read_records, write_records

    f = rooted_path(paths, "data_root", records)
    try:
        return list(read_records(f)), f, str(f)
    except PermissionError:
        print(f"[eval_v1] cannot read {f}; rebuilding from {annotations} + {split}", flush=True)
    ann = json.loads(rooted_path(paths, "data_root", annotations).read_text(encoding="utf-8"))
    ids = json.loads(rooted_path(paths, "data_root", split).read_text(encoding="utf-8"))["val"]
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "objects_val.jsonl"
    if not f.exists():
        write_records(build_coco_records(ann, ids), f)
    return list(read_records(f)), f, f"rebuilt from {annotations} + {split}"
