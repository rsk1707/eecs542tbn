"""Slide / poster figures for corrector v1 (matplotlib, light surface). Owner: P5.

Every number is computed from the eval run's ResultRows (results.parquet, P3's schema) or the calibration
curve, never typed in. Colours follow the team's validated categorical order (figures.py): corrector_v1 blue,
corrector_v0 orange, oracle_clicks aqua, fixed per method in every figure; the four rule baselines are
context in neutral gray. Every mark carries a direct text label (aqua is below 3:1 on white).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from promptcorrect.corrector.evaluate_v1 import final_rows
from promptcorrect.corrector.figures import AQUA, BLUE, GRID, INK, INK2, INK3, ORANGE, _spread, _style
from promptcorrect.corrector.figures import _grid as _grid1

GRAY = "#a3a29c"
METHODS = ["corrector_v1", "corrector_v0", "oracle_clicks", "b2_multimask", "b3_box_refit", "b3_unc_click",
           "b1_random_k"]
LABEL = {"corrector_v1": "Corrector v1", "corrector_v0": "Corrector v0", "oracle_clicks": "Ideal click",
         "b2_multimask": "Multimask", "b3_box_refit": "Box refit", "b3_unc_click": "Uncertainty click",
         "b1_random_k": "Random click", "b0_none": "No correction"}
COLOR = {"corrector_v1": BLUE, "corrector_v0": ORANGE, "oracle_clicks": AQUA}
SOURCES = [("gt", "Clean GT box"), ("perturbed", "Perturbed box (shift / scale / truncate)")]


def _grid(ax, axis: str = "y") -> None:
    for a in ("x", "y") if axis == "both" else (axis,):
        _grid1(ax, a)


def _color(m: str) -> str:
    return COLOR.get(m, GRAY)


def _finals(df: pd.DataFrame, good_iou: float = 0.75, harm_eps: float = 0.01) -> pd.DataFrame:
    f = final_rows(df)
    f["harmed"] = f.iou_final < f.iou0 - harm_eps
    f["clicked"] = f.steps_used > 0
    f["fail"] = f.iou0 < good_iou
    return f


def _stats(f: pd.DataFrame) -> pd.DataFrame:
    """Per (source, method): mean ΔIoU on failing starts, harm % on good starts, clicked %."""
    rows = []
    for (src, m), g in f.groupby(["prompt_origin", "method"]):
        rows.append(dict(source=src, method=m, delta_fail=g.loc[g.fail, "delta"].mean(),
                         harm_good=100 * g.loc[~g.fail, "harmed"].mean(), clicked=100 * g.clicked.mean(),
                         n_fail=int(g.fail.sum()), n_good=int((~g.fail).sum())))
    return pd.DataFrame(rows).set_index(["source", "method"])


def tradeoff(df: pd.DataFrame, out: Path, max_harm: float = 5.0) -> Path:
    """Gain on failing masks vs harm on good masks, one point per method, per prompt source."""
    _style()
    st = _stats(_finals(df))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
    for ax, (src, title) in zip(axes, SOURCES):
        s = st.loc[src]
        xmax = max(50.0, float(np.nan_to_num(s.harm_good.max())) * 1.15)
        ax.axvspan(0, max_harm, color=BLUE, alpha=0.07, linewidth=0)
        ax.axvline(max_harm, color=BLUE, linewidth=1, linestyle=(0, (4, 3)), alpha=0.6)
        ax.text(max_harm + 0.6, 0.02, f"harm limit {max_harm:.0f}%", transform=ax.get_xaxis_transform(),
                color=INK2, fontsize=10, va="bottom")
        ax.axhline(0, color=INK3, linewidth=1)
        pts = [(m, float(s.loc[m, "harm_good"]), float(s.loc[m, "delta_fail"])) for m in METHODS if m in s.index]
        pts = [p for p in pts if np.isfinite(p[1]) and np.isfinite(p[2])]
        if not pts:
            ax.set_title(f"{title}: no failing and good masks to compare")
            continue
        yspan = (max(p[2] for p in pts) - min(min(p[2] for p in pts), 0)) or 1.0
        for m, x, y in pts:
            hero = m == "corrector_v1"
            ax.scatter(x, y, s=170 if hero else 90, color=_color(m), edgecolor="white", linewidth=2, zorder=3)
        ys = _spread([p[2] for p in pts], 0.055 * yspan)
        for (m, x, y), yl in zip(pts, ys):
            ax.text(x + 0.012 * xmax, yl, f"{LABEL[m]}  {round(y, 3) + 0.0:+.3f}", color=INK if m in COLOR else INK2, va="center",
                    fontsize=11, fontweight="bold" if m == "corrector_v1" else "normal")
        ax.set_xlim(0, xmax * 1.25)
        ax.set_title(title)
        ax.set_xlabel("harm on good masks (start IoU ≥ 0.75), % of objects made worse")
        _grid(ax, "both")
    axes[0].set_ylabel("mean ΔIoU on failing masks (start IoU < 0.75)")
    fig.suptitle("One correction step: gain where SAM 3 fails vs harm where it was already right",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def bars(df: pd.DataFrame, out: Path, max_harm: float = 5.0) -> Path:
    """2 x 2: rows = prompt source, columns = ΔIoU on failing masks | harm on good masks."""
    _style()
    st = _stats(_finals(df))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    order = METHODS[::-1]
    for r, (src, title) in enumerate(SOURCES):
        s = st.loc[src]
        ms = [m for m in order if m in s.index]
        y = np.arange(len(ms))
        for c, (col, xlabel, fmt) in enumerate((("delta_fail", "mean ΔIoU, failing masks", "{:+.3f}"),
                                                 ("harm_good", "harm on good masks, %  (dashed = 5% limit)", "{:.1f}%"))):
            ax = axes[r, c]
            raw = s.loc[ms, col].to_numpy(dtype=float)
            v = np.nan_to_num(raw)  # an empty group (e.g. no failing masks) draws no bar and says "n/a"
            ax.barh(y, v, height=0.62, color=[_color(m) for m in ms], edgecolor="white", linewidth=2)
            span = max(abs(v).max(), 1e-6)
            for yi, vi, ri in zip(y, v, raw):  # labels right of the bar end, or right of zero for negative bars
                ax.text(max(vi, 0.0) + 0.015 * span, yi, "n/a" if np.isnan(ri) else fmt.format(round(vi, 3) + 0.0), va="center", ha="left", color=INK2,
                        fontsize=10.5)
            ax.axvline(0, color=INK3, linewidth=1)
            if col == "harm_good":
                ax.axvline(max_harm, color=BLUE, linewidth=1, linestyle=(0, (4, 3)), alpha=0.7)
            lo = min(0.0, float(v.min())) * 1.35
            ax.set_xlim(lo, float(v.max()) * 1.22 + 1e-6)
            ax.set_yticks(y, [LABEL[m] for m in ms] if c == 0 else [""] * len(ms))
            for t, m in zip(ax.get_yticklabels(), ms):
                t.set_fontweight("bold" if m == "corrector_v1" else "normal")
            ax.set_xlabel(xlabel)
            _grid(ax, "x")
        short = title.split(" (")[0]
        axes[r, 0].set_title(f"{short}: gain on failing masks (n = {int(s.n_fail.iloc[0])})")
        axes[r, 1].set_title(f"{short}: harm on good masks (n = {int(s.n_good.iloc[0])})")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def severity(df: pd.DataFrame, out: Path, edges=(0.2, 0.4, 0.6, 0.8, 0.95, 1.0001),
             methods=("corrector_v1", "corrector_v0", "oracle_clicks", "b2_multimask", "b3_box_refit")) -> Path:
    """Mean ΔIoU and v1's click rate vs the prompt box's IoU with the GT box (all prompts, clean included)."""
    _style()
    f = _finals(df)
    s0 = df[(df.step == 0) & (df.method == "b0_none")][["obj_id", "prompt_origin", "pert_type", "pert_level",
                                                          "box_iou_in"]].copy()
    s0["pert_type"] = s0["pert_type"].astype(object).where(s0["pert_type"].notna(), "none")
    s0["pert_level"] = s0["pert_level"].fillna(-1.0)
    f = f.merge(s0, on=["obj_id", "prompt_origin", "pert_type", "pert_level"], how="left")
    f["bin"] = pd.cut(f.box_iou_in.clip(lower=edges[0]), list(edges), right=False, labels=False)
    centers = list(range(len(edges) - 1))  # evenly spaced categories
    names = [f"{'<' if i == 0 else ''}{a:.2f}–{min(b, 1.0):.2f}" if i else f"< {b:.2f}"
             for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.axhline(0, color=INK3, linewidth=1)
    ends = []
    for m in methods:
        g = f[f.method == m].groupby("bin").delta.mean().reindex(range(len(centers)))
        hero = m == "corrector_v1"
        ax.plot(centers, g.values, color=_color(m), linewidth=3 if hero else 2,
                linestyle="-" if m in COLOR else (0, (4, 2)), marker="o", markersize=8 if hero else 6,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3 if hero else 2)
        ends.append((LABEL[m], float(g.values[0])))
    span = float(np.nanmax([e[1] for e in ends]) - min(0, np.nanmin([e[1] for e in ends]))) or 1.0
    for (text, _), yl in zip(ends, _spread([e[1] for e in ends], 0.075 * span)):
        ax.text(centers[0] - 0.15, yl, text, ha="right", va="center", color=INK2, fontsize=11)
    ax.set_xlim(-1.6, centers[-1] + 0.25)
    ax.set_xticks(centers, names, fontsize=10)
    ax.set_xlabel("prompt box IoU with the GT box  (right = clean box, left = worse box)")
    ax.set_ylabel("mean ΔIoU after one step (all objects)")
    ax.set_title("Gain by box quality")
    _grid(ax, "y")

    ax = axes[1]
    v1 = f[f.method == "corrector_v1"]
    click = (100 * v1.groupby("bin").clicked.mean()).reindex(range(len(centers)))
    n = v1.groupby("bin").size().reindex(range(len(centers))).fillna(0).astype(int)
    ax.bar(range(len(centers)), click.values, width=0.66, color=BLUE, edgecolor="white", linewidth=2)
    for i, (c, k) in enumerate(zip(click.values, n.values)):
        ax.text(i, c + 1.5, f"{c:.0f}%", ha="center", color=INK2, fontsize=10.5)
    ax.set_xticks(range(len(centers)), [f"{nm}\nn={k}" for nm, k in zip(names, n.values)], fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of objects corrector v1 clicks on")
    ax.set_xlabel("prompt box IoU with the GT box")
    ax.set_title("How often v1 clicks, by box quality")
    _grid(ax, "y")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def calibration(curve: pd.DataFrame, chosen: dict, out: Path) -> Path:
    """The abstain threshold sweep on held-out TRAINING images at the chosen error threshold."""
    _style()
    c = curve[(curve.err_threshold == chosen["err_threshold"]) & np.isfinite(curve.tau)].sort_values("tau")
    tau = chosen["tau"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, col, ylabel, title in (
            (axes[0], "delta_all", "mean ΔIoU, all held-out samples", "Gain vs abstain threshold"),
            (axes[1], "harm_good_pct", "harm on good masks, %", "Harm vs abstain threshold")):
        ax.plot(c.tau, c[col], color=BLUE, linewidth=2)
        if tau is not None:
            ax.axvline(tau, color=INK3, linewidth=1, linestyle=(0, (4, 3)))
            yv = float(c.loc[(c.tau - tau).abs().idxmin(), col])
            ax.scatter([tau], [yv], s=80, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(f"chosen τ = {tau:.2f}: {yv:+.3f}" if col == "delta_all" else f"chosen τ = {tau:.2f}: {yv:.1f}%",
                        (tau, yv), xytext=(-12, 12), textcoords="offset points", ha="right", color=INK, fontsize=11)
        if col == "harm_good_pct":
            ax.axhline(100 * chosen.get("max_harm", 0.05), color=BLUE, linewidth=1, linestyle=(0, (4, 3)), alpha=0.6)
            ax.text(0.01, 100 * chosen.get("max_harm", 0.05) + 0.6, "5% limit", color=INK2, fontsize=10)
        ax.set_xlabel("τ: click only if predicted IoU < τ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        _grid(ax, "both")
    fig.suptitle(f"Choosing the abstain threshold on {int(c.n.iloc[0])} held-out training samples "
                 f"(error threshold {chosen['err_threshold']}, logit feedback)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
