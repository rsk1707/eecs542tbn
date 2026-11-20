"""Slide-ready figures for a sensitivity sweep (perturb/sweep.py), in the shared slide style.

Chart choices: IoU against perturbation level and against box IoU are lines (change over an ordered
amount); correction methods against step are lines; harm rates are bars. Each perturbation type and each
method keeps ONE categorical color in every figure, in fixed palette order (slots 1-6 validated on the
light surface: adjacent-pair CVD dE 9.1, normal-vision dE 19.6). Aqua, yellow and magenta sit below 3:1
contrast, so every line carries a direct end label and all numbers are also in the summary CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from promptcorrect.perturb.box import IDENTITY
from promptcorrect.perturb.sweep import MIN_BIN_N, by_box_iou, method_table, sensitivity, trajectories

# Shared plot style, defined here so this module is self-contained.
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#85847f", "#e4e3df"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "axes.grid": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


TYPE_COLORS = {"shift": "#2a78d6", "scale": "#eb6834", "truncate": "#1baf7a"}
TYPE_LABELS = {"shift": "shift (fraction of box size)", "scale": "scale factor", "truncate": "fraction cut off one side"}
METHOD_COLORS = {"oracle_clicks": "#2a78d6", "b1_random_k": "#eb6834", "b2_multimask": "#1baf7a",
                 "b3_box_refit": "#eda100", "b3_unc_click": "#e87ba4"}
METHOD_LABELS = {"b0_none": "no correction", "oracle_clicks": "ideal clicks (upper bound)",
                 "b1_random_k": "random clicks", "b2_multimask": "SAM multimask", "b3_box_refit": "box refit",
                 "b3_unc_click": "uncertainty click"}


def _end_labels(ax, items, min_gap: float):
    """Labels at line ends, nudged apart vertically so they never overlap. items: [(x, y, text)]."""
    placed = []
    for x, y, text in sorted(items, key=lambda it: it[1]):
        y_text = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_text)
        ax.annotate(text, (x, y), xytext=(x, y_text), textcoords="data", va="center", ha="left", color=INK,
                    fontsize=12, annotation_clip=False)


def _axes(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def sensitivity_curves(df: pd.DataFrame, out: Path) -> Path:
    """One panel per perturbation type: mean mask IoU against level, the clean box drawn at the identity level."""
    _style()
    t = sensitivity(df)
    clean = t.loc["none", "mean_iou"].iloc[0] if "none" in t.index.get_level_values(0) else np.nan
    types = [p for p in TYPE_COLORS if p in t.index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(types), figsize=(4.6 * len(types), 4.6), sharey=True, squeeze=False)
    lo = min(t.mean_iou.min(), clean if np.isfinite(clean) else 1) - 0.06
    for ax, p in zip(axes[0], types):
        s = t.loc[p, "mean_iou"].copy()
        if np.isfinite(clean):
            s.loc[IDENTITY[p]] = clean
        s = s.sort_index()
        ax.plot(s.index, s.values, color=TYPE_COLORS[p], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        for x in (s.index[0], s.index[-1]):
            ax.annotate(f"{s.loc[x]:.2f}", (x, s.loc[x]), xytext=(0, 10), textcoords="offset points", ha="center",
                        color=INK, fontsize=12)
        if np.isfinite(clean):
            ax.axhline(clean, color=INK3, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(p)
        ax.set_xlabel(TYPE_LABELS[p])
        if p == "scale":
            ax.set_xscale("log")
            ax.set_xticks(list(s.index), [f"{v:g}" for v in s.index])
            ax.minorticks_off()
        ax.set_ylim(max(0, lo), 1.0)
        _axes(ax)
    axes[0][0].set_ylabel("mean mask IoU")
    if np.isfinite(clean):
        axes[0][-1].text(0.98, clean - 0.01, "clean GT box", transform=axes[0][-1].get_yaxis_transform(), va="top",
                         ha="right", color=INK3, fontsize=11)
    fig.suptitle("How fast does SAM 3's mask degrade as the box gets worse? (no correction)", x=0.01, y=0.99,
                 ha="left", fontsize=15, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def iou_vs_box_iou(df: pd.DataFrame, out: Path) -> Path:
    """Mean mask IoU per box-IoU bin, one line per type: puts all perturbation types on one severity axis."""
    _style()
    t = by_box_iou(df, min_n=MIN_BIN_N)
    x = np.array([iv.left + (min(iv.right, 1.0) - iv.left) / 2 for iv in t.index])
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ends = []
    for p in [p for p in TYPE_COLORS if p in t.columns]:
        v = t[p].to_numpy()
        ok = np.isfinite(v)
        if not ok.any():
            continue
        ax.plot(x[ok], v[ok], color=TYPE_COLORS[p], linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=p)
        ends.append((x[ok][-1] + 0.015, v[ok][-1], p))
    _end_labels(ax, ends, min_gap=0.045)
    ax.set_xlabel("box IoU of the prompt with the GT box (binned)")
    ax.set_ylabel("mean mask IoU")
    ax.set_xlim(0, 1.1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, 1.0)
    ax.set_title("Mask quality against prompt-box quality, all perturbation types")
    ax.legend(frameon=False, loc="lower right", fontsize=11, labelcolor=INK2)
    _axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def method_curves(df: pd.DataFrame, out: Path) -> Path | None:
    """Left: mean IoU against correction step on perturbed boxes. Right: harm rate on clean GT boxes."""
    w = trajectories(df[df.pert_type.notna()])
    methods = [m for m in METHOD_COLORS if m in w.index.get_level_values(0)]
    if not methods:
        return None
    _style()
    mean = w.groupby(level="method").mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    base = mean.loc[methods, 0].mean()  # every method starts from the same step-0 mask
    ax.axhline(base, color=INK3, linewidth=1, linestyle=(0, (4, 3)), label=METHOD_LABELS["b0_none"])
    last = mean.columns[-1]
    for m in methods:
        v = mean.loc[m].to_numpy()
        ax.plot(mean.columns, v, color=METHOD_COLORS[m], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=METHOD_LABELS[m])
    lo, hi = min(base, mean.loc[methods].min().min()), max(base, mean.loc[methods].max().max())
    _end_labels(ax, [(last + 0.12, mean.loc[m, last], f"{mean.loc[m, last]:.3f}") for m in methods],
                min_gap=0.06 * max(hi - lo, 0.02))
    ax.set_xticks(mean.columns)
    ax.set_xlim(-0.2, mean.columns.max() + 0.7)
    ax.set_xlabel("correction step (clicks or box updates)")
    ax.set_ylabel("mean mask IoU")
    ax.set_title("Perturbed boxes: what each correction recovers")
    ax.legend(frameon=False, fontsize=11, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    _axes(ax)

    clean = method_table(df, "clean")
    ms = [m for m in methods if m in clean.index]
    y = np.arange(len(ms))[::-1]
    vals = [clean.loc[m, "harm_%"] for m in ms]
    bx.barh(y, vals, height=0.5, color=[METHOD_COLORS[m] for m in ms], edgecolor=SURFACE, linewidth=1.5)
    top = max(vals + [1.0])
    for yi, v in zip(y, vals):
        bx.text(v + top * 0.02, yi, f"{v:.1f}%", va="center", color=INK, fontsize=12)
    bx.set_yticks(y, [METHOD_LABELS[m] for m in ms])
    bx.set_xlim(0, top * 1.25)
    bx.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    bx.set_title("Clean GT boxes: objects made worse")
    bx.xaxis.grid(True, color=GRID, linewidth=1)
    bx.set_axisbelow(True)
    bx.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [sensitivity_curves(df, out_dir / "sensitivity_curves.png"),
            iou_vs_box_iou(df, out_dir / "sensitivity_by_box_iou.png"),
            method_curves(df, out_dir / "baselines_vs_oracle.png")]
    return [p for p in made if p is not None]
