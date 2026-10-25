"""Slide-ready RQ0 figures from the per-object rows (matplotlib, light surface).

Chart choices (dataviz method): failure rates and by-size rates are bars (magnitude by category); the
click curves are lines (change over a count); predicted-vs-true IoU is a density (one series, so one
sequential blue ramp). Each prompt source keeps ONE categorical color in every figure (fixed order, validated:
adjacent-pair CVD dE 9.1, normal-vision dE 22.9). Aqua and yellow sit below 3:1 contrast on the white
surface, so every mark carries a visible value or direct label and the numbers are also in the summary CSVs.
Text uses ink colors, never the series colors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from promptcorrect.experiments.rq0_summary import by_size, headline, oracle_curves

SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#85847f", "#e4e3df"
# categorical slots 1-4, fixed order: blue, orange, aqua, yellow
COLORS = {"gt_box": "#2a78d6", "grounding_dino_box": "#eb6834", "sam3_text_box": "#1baf7a", "sam3_text_mask": "#eda100"}
LABELS = {"gt_box": "GT box (ceiling)", "grounding_dino_box": "Grounding DINO box",
          "sam3_text_box": "SAM 3 text → box", "sam3_text_mask": "SAM 3 text mask"}
ORDER = list(COLORS)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"]  # sequential blue, steps 100 -> 650


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "axes.grid": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _present(sources) -> list[str]:
    return [s for s in ORDER if s in set(sources)]


def failure_rates(df: pd.DataFrame, out: Path) -> Path:
    """Three panels: missed objects, IoU < 0.5 and IoU < 0.75 (missed objects count as failures)."""
    _style()
    h = headline(df).set_index("source")
    srcs = _present(h.index)
    panels = [("missed_%", "Object missed (no detection)"), ("fail<0.5_all_%", "Mask IoU below 0.5"),
              ("fail<0.75_all_%", "Mask IoU below 0.75")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.9 * len(srcs) + 2.3), sharey=True)
    y = np.arange(len(srcs))[::-1]
    top = max(h.loc[srcs, c].max() for c, _ in panels)
    for ax, (col, title) in zip(axes, panels):
        vals = [h.loc[s, col] for s in srcs]
        ax.barh(y, vals, height=0.5, color=[COLORS[s] for s in srcs], edgecolor=SURFACE, linewidth=1.5)
        for yi, v in zip(y, vals):
            ax.text(v + top * 0.015, yi, f"{v:.1f}%", va="center", ha="left", color=INK, fontsize=12)
        ax.set_title(title)
        ax.set_xlim(0, top * 1.22)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_yticks(y, [LABELS[s] for s in srcs])
    fig.suptitle("Share of ground-truth objects with a poor mask, per prompt source", x=0.01, y=0.99, ha="left",
                 fontsize=15, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def click_curves(df: pd.DataFrame, out: Path) -> Path | None:
    """Mean IoU and share of objects reaching IoU >= 0.85, against the number of ideal clicks."""
    tables = oracle_curves(df)
    if not tables:
        return None
    _style()
    mean = tables["oracle_mean_iou_by_clicks"]
    reach = tables["oracle_reached_0.85_%_by_clicks"]
    srcs = _present(mean.index)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, tab, title, fmt, ylab in [
            (axes[0], mean, "Mean mask IoU", lambda v: f"{v:.3f}", "mean IoU"),
            (axes[1], reach, "Objects with IoU ≥ 0.85", lambda v: f"{v:.0f}%", "% of objects")]:
        ymin, ymax = tab.loc[srcs].min().min(), tab.loc[srcs].max().max()
        pad = (ymax - ymin) * 0.18 or 0.05
        for s in srcs:
            x, v = tab.columns.to_numpy(), tab.loc[s].to_numpy()
            ax.plot(x, v, color=COLORS[s], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=LABELS[s])
            ax.annotate(fmt(v[-1]), (x[-1], v[-1]), xytext=(9, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=12)
        ax.set_title(title)
        ax.set_xlabel("ideal clicks added to the box prompt")
        ax.set_ylabel(ylab)
        ax.set_xticks(tab.columns)
        ax.set_xlim(-0.2, tab.columns.max() + 0.9)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    if len(srcs) > 1:
        axes[0].legend(frameon=False, loc="lower right", fontsize=11, labelcolor=INK2)
    fig.suptitle("How much do ideal clicks recover? (detected objects, ground-truth-guided clicks)", x=0.01,
                 y=0.99, ha="left", fontsize=15, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def overconfidence(df: pd.DataFrame, out: Path, source: str = "gt_box") -> Path | None:
    """SAM 3's own predicted IoU against the real IoU, one box source, no correction."""
    g = df[(df.method == "b0_none") & (df.step == 0) & df.detected & (df.source == source) & df.pred_iou.notna()]
    if len(g) < 3:
        return None
    _style()
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    hb = ax.hexbin(g.iou, g.pred_iou, gridsize=28, extent=(0, 1, 0, 1), mincnt=1, cmap=cmap, linewidths=0.4,
                   edgecolors=SURFACE)
    ax.plot([0, 1], [0, 1], color=INK3, linewidth=1.5)
    ax.text(0.50, 0.38, "calibrated", color=INK3, fontsize=11, rotation=42, ha="left", va="bottom")
    ax.set_xlabel("real mask IoU with the ground truth")
    ax.set_ylabel("IoU predicted by SAM 3")
    ax.set_title("SAM 3 over-estimates its own quality")
    ax.text(0.97, 0.04, f"{LABELS[source]}, n = {len(g)}\nmean predicted IoU  {g.pred_iou.mean():.2f}\n"
            f"mean real IoU          {g.iou.mean():.2f}", ha="right", va="bottom", color=INK, fontsize=12,
            linespacing=1.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("objects", color=INK2)
    cb.outline.set_edgecolor(GRID)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def failures_by_size(df: pd.DataFrame, out: Path, threshold: float = 0.75) -> Path:
    """Grouped bars: share with IoU below the threshold, per COCO size bin and source."""
    _style()
    t = by_size(df, threshold)
    srcs = _present(t.index)
    bins = [b for b in ("small", "medium", "large") if b in t.columns]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.78 / len(srcs)
    for i, s in enumerate(srcs):
        x = np.arange(len(bins)) + (i - (len(srcs) - 1) / 2) * width
        vals = [t.loc[s, b] for b in bins]
        ax.bar(x, vals, width=width, color=COLORS[s], edgecolor=SURFACE, linewidth=1.5, label=LABELS[s])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", color=INK, fontsize=11)
    ax.set_xticks(np.arange(len(bins)), [f"{b} objects" for b in bins])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylim(0, max(10, t.loc[srcs, bins].max().max() * 1.18))
    ax.set_title(f"Poor masks (IoU below {threshold}) by object size")
    if len(srcs) > 1:
        ax.legend(frameon=False, fontsize=11, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                  ncol=len(srcs))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [failure_rates(df, out_dir / "rq0_failure_rates.png"), click_curves(df, out_dir / "rq0_clicks.png"),
            overconfidence(df, out_dir / "rq0_overconfidence.png"), failures_by_size(df, out_dir / "rq0_by_size.png")]
    return [p for p in made if p is not None]
