"""Headline RQ0 tables from the per-object rows written by `run_rq0` (counts and rates, no plots).

All rates are computed from raw columns only (docs/interfaces.md section 5): a GT object that no
detection matched counts as a failure with IoU 0 in the "all" columns and is excluded from the
"detected" columns, so *missed* and *detected but poor mask* stay separate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = (0.5, 0.75, 0.9)


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """Per source, no correction: missed rate, mean IoU, and failure rates at IoU < 0.5 / 0.75 / 0.9."""
    b0 = df[(df.method == "b0_none") & (df.step == 0)]
    out = []
    for src, g in b0.groupby("source", sort=False):
        det = g[g.detected]
        row = {"source": src, "n_gt": len(g), "missed_%": 100 * (1 - g.detected.mean()),
               "mean_iou_all": g.iou.mean(), "mean_iou_detected": det.iou.mean() if len(det) else np.nan}
        for t in THRESHOLDS:
            row[f"fail<{t}_all_%"] = 100 * (g.iou < t).mean()
        for t in THRESHOLDS:
            row[f"fail<{t}_detected_%"] = 100 * (det.iou < t).mean() if len(det) else np.nan
        out.append(row)
    return pd.DataFrame(out).round(3)


def by_size(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Failure rate (IoU < threshold, missed counted) per source and COCO size bin."""
    b0 = df[(df.method == "b0_none") & (df.step == 0)].copy()
    b0["fail"] = (b0.iou < threshold) * 100.0
    t = b0.pivot_table(index="source", columns="size_bin", values="fail", aggfunc="mean", sort=False)
    t["n_gt"] = b0.groupby("source", sort=False).size()
    return t.round(2)


def _oracle_wide(df: pd.DataFrame) -> pd.DataFrame:
    o = df[df.method == "oracle_clicks"]
    wide = o.pivot_table(index=["source", "obj_id"], columns="step", values="iou", aggfunc="first")
    return wide.ffill(axis=1)  # an object that stopped early (exact mask) keeps its last IoU


def oracle_curves(df: pd.DataFrame, targets=(0.85, 0.9), failure_below: float = 0.75) -> dict[str, pd.DataFrame]:
    """Headroom from ideal clicks, over detected objects: mean IoU after k clicks, the fraction of objects
    that has reached a target IoU within k clicks, and the same mean IoU restricted to the failures
    (step-0 IoU below `failure_below`)."""
    wide = _oracle_wide(df)
    if wide.empty:
        return {}
    by_src = wide.groupby(level="source", sort=False)
    out = {"oracle_mean_iou_by_clicks": by_src.mean().round(3)}
    for t in targets:
        reached = (wide >= t).cummax(axis=1).groupby(level="source", sort=False).mean() * 100
        out[f"oracle_reached_{t}_%_by_clicks"] = reached.round(2)
    fails = wide[wide[0] < failure_below]
    if len(fails):
        t = fails.groupby(level="source", sort=False).mean().round(3)
        t.insert(0, "n_failures", fails.groupby(level="source", sort=False).size())
        out[f"oracle_mean_iou_by_clicks_failures<{failure_below}"] = t
    return out


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """SAM 3's predicted IoU vs the real IoU for box prompts, detected objects, no correction."""
    g = df[(df.method == "b0_none") & (df.step == 0) & df.detected & df.pred_iou.notna()]
    rows = []
    for src, s in g.groupby("source", sort=False):
        rows.append({"source": src, "n": len(s), "mean_pred_iou": s.pred_iou.mean(), "mean_true_iou": s.iou.mean(),
                     "mean_overconfidence": (s.pred_iou - s.iou).mean(),
                     "corr": s.pred_iou.corr(s.iou) if len(s) > 2 else np.nan})
    return pd.DataFrame(rows).round(3)


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {"headline": headline(df), "failure_rate_by_size_bin_iou<0.75": by_size(df), "calibration": calibration(df)}
    tables.update(oracle_curves(df))
    return tables


def format_report(tables: dict[str, pd.DataFrame]) -> str:
    parts = []
    for name, t in tables.items():
        parts.append(f"== {name} ==\n{t.to_string()}\n")
    return "\n".join(parts)
