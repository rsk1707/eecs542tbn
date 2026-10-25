"""Slide figures for corrector v0 (matplotlib, light surface). Owner: P5.

Same style and validated categorical order as viz/rq0_figures.py (blue, orange, aqua, yellow), assigned by
method and kept fixed in every figure; "no correction" is a neutral gray reference line. Every bar carries
its value as text, and all numbers are also in the summary CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from promptcorrect.viz.rq0_figures import GRID, INK, INK2, INK3, SURFACE, _style

COLORS = {"corrector_v0": "#2a78d6", "b3_box_refit": "#eb6834", "oracle_clicks": "#1baf7a",
          "b1_random_k": "#eda100", "b0_none": INK3}
LABELS = {"corrector_v0": "Corrector v0", "b3_box_refit": "Box refit", "oracle_clicks": "Ideal clicks (upper bound)",
          "b1_random_k": "Random clicks", "b0_none": "No correction"}


def _color(m: str) -> str:
    return COLORS.get(m, COLORS["corrector_v0"] if m.startswith("corrector_v") else INK2)


def _label(m: str) -> str:
    return LABELS.get(m, m.replace("_", " "))


def _grid(ax, axis="y"):
    getattr(ax, f"{axis}axis").grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def training_curves(history: list[dict], out: Path) -> Path:
    _style()
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for col, label, c in (("train_loss", "Train", "#2a78d6"), ("val_loss", "Held-out train images", "#eb6834")):
        ax.plot(h.epoch, h[col], color=c, linewidth=2)
        ax.text(h.epoch.iloc[-1] + 0.3, h[col].iloc[-1], label, color=INK2, va="center", fontsize=11)
    ax.set_title("Loss (BCE + Dice)")
    ax.set_xlabel("epoch")
    ax.set_xlim(h.epoch.min(), h.epoch.max() * 1.35)
    _grid(ax)
    ax = axes[1]
    for col, label, c in (("fn_auroc", "AUROC, missed pixels", "#2a78d6"), ("fp_auroc", "AUROC, extra pixels", "#eb6834"),
                          ("click_hit_rate", "Click hits a real error", "#1baf7a")):
        ax.plot(h.epoch, h[col], color=c, linewidth=2)
        ax.text(h.epoch.iloc[-1] + 0.3, h[col].iloc[-1], f"{label} {h[col].iloc[-1]:.2f}", color=INK2, va="center", fontsize=11)
    ax.set_title("Validation (held-out train images)")
    ax.set_xlabel("epoch")
    ax.set_xlim(h.epoch.min(), h.epoch.max() * 1.6)
    ax.set_ylim(0.5, 1.0)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def iou_by_step(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    _style()
    panels = [("iou_by_step_perturbed", "Perturbed boxes"), ("iou_by_step_failures<0.75", "Objects starting below IoU 0.75")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, (key, title) in zip(axes, panels):
        t = tables[key]
        steps = list(t.columns)
        for m in t.index:
            y = t.loc[m].values
            ax.plot(steps, y, color=_color(m), linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2,
                    linestyle="--" if m == "oracle_clicks" else "-")
            ax.text(steps[-1] + 0.08, y[-1], f"{_label(m)} {y[-1]:.3f}", color=INK2, va="center", fontsize=11)
        ax.set_title(title)
        ax.set_xticks(steps)
        ax.set_xlabel("correction step")
        ax.set_ylabel("mean mask IoU")
        ax.set_xlim(steps[0] - 0.1, steps[-1] + 1.6)
        _grid(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def gain_and_harm(tables: dict[str, pd.DataFrame], out: Path) -> Path:
    """Left: mean IoU change on perturbed boxes. Right: share of clean GT-box prompts made worse (RQ4)."""
    _style()
    methods = [m for m in tables["final_perturbed"].index if m != "b0_none"]
    panels = [("final_perturbed", "delta", "Mean IoU change, perturbed boxes", "{:+.3f}"),
              ("final_clean", "harm_pct", "Clean GT boxes made worse (IoU −0.01)", "{:.1f}%")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.7 * len(methods) + 2.2), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (key, col, title, fmt) in zip(axes, panels):
        vals = [float(tables[key].loc[m, col]) for m in methods]
        ax.barh(y, vals, height=0.5, color=[_color(m) for m in methods], edgecolor=SURFACE, linewidth=1.5)
        span = max(abs(v) for v in vals) or 1.0
        for yi, v in zip(y, vals):
            ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), yi, fmt.format(v), va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=12)
        ax.axvline(0, color=INK3, linewidth=1)
        lo = min(0.0, min(vals))
        ax.set_xlim(lo * 1.4 - (span * 0.05 if lo < 0 else 0), max(0.0, max(vals)) * 1.35 + span * 0.05)
        ax.set_title(title)
        _grid(ax, "x")
    axes[0].set_yticks(y, [_label(m) for m in methods])
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
