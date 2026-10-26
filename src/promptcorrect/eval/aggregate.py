"""Summaries from a results DataFrame (docs/interfaces.md section 5). Owner: P3.

Derived metrics are computed here, never stored: failure rates, harm rate, NoC, click curves.
Tables take `by` (default: prompt source) so the same numbers can be split by size_bin, novelty, etc.
"""

from __future__ import annotations

import pandas as pd

from promptcorrect.eval.oracle_clicks import noc

THRESHOLDS = (0.5, 0.75, 0.9)
HARM_MARGIN = 0.01  # harm: iou_final < iou_step0 - 0.01 (defined once, here)
TRACE = ["obj_id", "prompt_origin", "pert_type", "pert_level", "method"]  # one trace = one prompt x method
ATTRS = ["size_bin", "novelty"]


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    """Unperturbed rows have null pert_type/pert_level; fill them so groupby keeps them."""
    return df.assign(pert_type=df.pert_type.fillna("none"), pert_level=df.pert_level.fillna(-1.0))


def initial(df: pd.DataFrame) -> pd.DataFrame:
    """The uncorrected mask: one row per object and prompt."""
    return df[(df.method == "b0_none") & (df.step == 0)]


def failure_table(df: pd.DataFrame, by=("prompt_origin",)) -> pd.DataFrame:
    """How often the initial mask is poor. Missed objects count as IoU 0 in `mean_iou` and `fail@t`;
    `mean_iou_detected` and `pred_iou` cover detected objects only."""
    s0 = initial(df)
    g, gd = s0.groupby(list(by)), s0[s0.detected].groupby(list(by))
    t = pd.DataFrame({
        "n": g.size(), "detected": g.detected.mean(), "mean_iou": g.iou.mean(),
        "mean_iou_detected": gd.iou.mean(), "mean_boundary_iou": g.boundary_iou.mean(),
        **{f"fail@{t}": g.iou.apply(lambda x, t=t: (x < t).mean()) for t in THRESHOLDS},
        "pred_iou": gd.pred_iou.mean(),
    })
    t["overconfidence"] = t.pred_iou - t.mean_iou_detected
    return t


def traces(df: pd.DataFrame, method: str | None = "oracle_clicks", only_below: float | None = None) -> pd.DataFrame:
    """[trace, step] IoU for one method (None: every correction method). Traces that stopped early keep
    their last mask (forward-filled). `only_below`: keep traces whose step-0 IoU is below this."""
    d = _keyed(df[df.method == method] if method else df[df.method != "b0_none"])
    w = d.set_index(TRACE + ATTRS + ["step"]).iou.unstack("step").ffill(axis=1)
    return w if only_below is None else w[w[0] < only_below]


def step_curve(df: pd.DataFrame, by=("prompt_origin", "method"), **kw) -> pd.DataFrame:
    """Mean IoU after k steps of each correction method. Raw, not best-so-far: a deployed method cannot
    use the GT to go back to its best step, so this is the fair comparison across methods."""
    w = traces(df, method=None, **kw)
    g = w.groupby(level=list(by))
    return g.mean().assign(n=g.size())


def click_curve(df: pd.DataFrame, by=("prompt_origin",), **kw) -> pd.DataFrame:
    """Mean best-so-far IoU after k clicks: groups as rows, k as columns, plus n."""
    w = traces(df, **kw)
    g = w.cummax(axis=1).groupby(level=list(by))
    return g.mean().assign(n=g.size())


def noc_table(df: pd.DataFrame, by=("prompt_origin",), thresholds=(0.85, 0.9), **kw) -> pd.DataFrame:
    """Mean clicks to reach each IoU threshold, and the share that never does within the click budget
    (those count as budget + 1, so the mean is a lower bound)."""
    w = traces(df, **kw)
    budget = int(w.columns.max())
    out = {}
    for t in thresholds:
        n = pd.Series([noc(r, t, budget) for r in w.to_numpy().tolist()], index=w.index)
        g = n.groupby(level=list(by))
        out[f"noc@{round(t * 100)}"] = g.mean()
        out[f"not_reached@{round(t * 100)}"] = g.apply(lambda x: (x > budget).mean())
    out["n"] = w.groupby(level=list(by)).size()
    return pd.DataFrame(out)


def harm_table(df: pd.DataFrame, by=("prompt_origin", "method")) -> pd.DataFrame:
    """Per correction method: mean IoU gain from step 0 to the method's last step, and the harm rate."""
    d = _keyed(df[df.method != "b0_none"]).sort_values("step")
    g = d.groupby(TRACE + ATTRS)
    first, last = g.iou.first(), g.iou.last()
    t = pd.DataFrame({"gain": last - first, "harmed": last < first - HARM_MARGIN}).reset_index()
    out = t.groupby(list(by))
    return pd.DataFrame({"n": out.size(), "mean_gain": out.gain.mean(), "harm_rate": out.harmed.mean()})


def headline(df: pd.DataFrame) -> dict[str, float]:
    """A few numbers per prompt source for results/runs_index.csv."""
    f = failure_table(df)
    return {f"{src}_{col}": round(float(f.loc[src, col]), 4) for src in f.index for col in ("mean_iou", "fail@0.75")}


def all_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {
        "failure": failure_table(df),
        "failure_by_size": failure_table(df, by=("prompt_origin", "size_bin")),
    }
    if (df.method != "b0_none").any():
        tables |= {
            "methods": step_curve(df),
            "methods_failures": step_curve(df, only_below=0.75),
            "harm": harm_table(df),
        }
    if (df.method == "oracle_clicks").any():
        tables |= {
            "click_curve": click_curve(df),
            "click_curve_failures": click_curve(df, only_below=0.75),
            "noc": noc_table(df),
            "noc_failures": noc_table(df, only_below=0.75),
            "noc_by_size": noc_table(df, by=("prompt_origin", "size_bin")),
        }
    return {k: v.round(4) for k, v in tables.items()}


def as_text(tables: dict[str, pd.DataFrame]) -> str:
    return "\n\n".join(f"== {k} ==\n{v.to_string()}" for k, v in tables.items())
