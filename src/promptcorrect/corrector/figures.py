"""Slide figures for corrector v0 (matplotlib, light surface). Owner: P5.

Colours are the team's validated categorical order from the RQ0 figures (blue, orange, aqua; adjacent-pair
CVD dE 9.1), fixed per series in every figure; "no correction" is a neutral gray. Every bar carries its
value as text (aqua sits below 3:1 contrast on white), and all numbers are also in the summary CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#85847f", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GROUPS = [("all", "All\nobjects"), ("start<0.75", "Start below\nIoU 0.75"), ("size:small", "Small"),
          ("size:medium", "Medium"), ("size:large", "Large")]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "axes.grid": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _grid(ax, axis="y"):
    getattr(ax, f"{axis}axis").grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _spread(ys: list[float], min_gap: float) -> list[float]:
    """Label positions for line ends: same order as `ys`, at least `min_gap` apart, centred on the data."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_gap)
    shift = (sum(ys) - sum(pos)) / len(pos) if pos else 0.0
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k] + shift
    return out


def _end_labels(ax, x, series: list[tuple[str, float]], gap: float):
    for (text, _), y in zip(series, _spread([v for _, v in series], gap)):
        ax.text(x, y, text, color=INK2, va="center", fontsize=11)


def training_curves(history: list[dict], out: Path) -> Path:
    _style()
    h = pd.DataFrame(history)
    x_end = h.epoch.iloc[-1] + 0.3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    loss = (("train_loss", "Train", BLUE), ("val_loss", "Held-out train images", ORANGE))
    for col, _, c in loss:
        ax.plot(h.epoch, h[col], color=c, linewidth=2)
    span = float(h[[c for c, _, _ in loss]].max().max() - h[[c for c, _, _ in loss]].min().min()) or 1.0
    _end_labels(ax, x_end, [(label, h[col].iloc[-1]) for col, label, _ in loss], 0.08 * span)
    ax.set_title("Loss (BCE + Dice)")
    ax.set_xlabel("epoch")
    ax.set_xlim(h.epoch.min(), h.epoch.max() * 1.35)
    _grid(ax)
    ax = axes[1]
    val = (("fn_auroc", "AUROC, missed pixels", BLUE), ("fp_auroc", "AUROC, extra pixels", ORANGE),
           ("click_hit_rate", "Click hits a real error", AQUA))
    for col, _, c in val:
        ax.plot(h.epoch, h[col], color=c, linewidth=2)
    _end_labels(ax, x_end, [(f"{label} {h[col].iloc[-1]:.2f}", h[col].iloc[-1]) for col, label, _ in val], 0.035)
    ax.set_title("Validation (held-out train images)")
    ax.set_xlabel("epoch")
    ax.set_xlim(h.epoch.min(), h.epoch.max() * 1.6)
    ax.set_ylim(0.0, 1.0)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def corrector_vs_none(tables: dict[str, pd.DataFrame], out: Path, method: str = "corrector_v0") -> Path:
    """Left: mean IoU without and with the corrector, per object group. Right: share of objects the
    corrector improved / harmed (IoU change beyond +-0.01), per group."""
    _style()
    final = tables["final"]
    groups = [(g, label) for g, label in GROUPS if (g, method) in final.index]
    x = np.arange(len(groups))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    def bars(ax, xs, vals, color, fmt):
        ax.bar(xs, vals, width=w, color=color, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=10.5)

    ax = axes[0]
    bars(ax, x - w / 2, [final.loc[(g, "b0_none"), "iou_final"] for g, _ in groups], INK3, "{:.3f}")
    bars(ax, x + w / 2, [final.loc[(g, method), "iou_final"] for g, _ in groups], BLUE, "{:.3f}")
    ax.set_title("Mean mask IoU (clean GT box → SAM 3)")
    ax.set_ylim(0, 1.15)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK3), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["No correction", "Corrector v0 (1 click)"], frameon=False, loc="upper left", ncols=2)
    ax = axes[1]
    imp = [final.loc[(g, method), "improved_pct"] for g, _ in groups]
    harm = [final.loc[(g, method), "harm_pct"] for g, _ in groups]
    bars(ax, x - w / 2, imp, BLUE, "{:.1f}%")
    bars(ax, x + w / 2, harm, ORANGE, "{:.1f}%")
    ax.set_title("Objects changed by the corrector")
    ax.set_ylim(0, max(max(imp), max(harm), 10.0) * 1.3)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Improved (> +0.01)", "Harmed (< −0.01)"], frameon=False, loc="upper left", ncols=2)
    for ax in axes:
        ax.set_xticks(x, [label for _, label in groups])
        _grid(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
