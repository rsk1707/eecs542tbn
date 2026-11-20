"""RQ0 figures from a results DataFrame. Owner: P3. Needs the `plot` extra (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from promptcorrect.eval.aggregate import click_curve, initial, step_curve, traces  # noqa: E402
from promptcorrect.eval.oracle_clicks import noc  # noqa: E402

# One fixed colour per prompt source in every figure (Okabe-Ito, colour-blind safe).
COLOR = {"gt": "#000000", "grounding_dino": "#E69F00", "sam3_text": "#0072B2", "sam3_text_mask": "#56B4E9"}
METHOD_STYLE = {"oracle_clicks": ("#000000", "ideal clicks (upper bound)"), "b1_random_k": ("#999999", "random click"),
                "b2_multimask": ("#CC79A7", "multimask"), "b3_box_refit": ("#009E73", "box refit"),
                "b3_unc_click": ("#D55E00", "uncertainty click")}
LABEL = {"gt": "GT box", "grounding_dino": "Grounding DINO box", "sam3_text": "SAM 3 text box",
         "sam3_text_mask": "SAM 3 text mask"}


def _order(sources) -> list[str]:
    return [s for s in COLOR if s in set(sources)]


def failure_bars(df: pd.DataFrame, ax=None):
    """Share of GT objects per outcome of the initial prompt: missed, IoU < 0.5, 0.5-0.75, >= 0.75."""
    s0 = initial(df)
    src = _order(s0.prompt_origin.unique())
    g = s0.groupby("prompt_origin")
    parts = {
        "missed": g.apply(lambda x: (~x.detected).mean()),
        "IoU < 0.5": g.apply(lambda x: (x.detected & (x.iou < 0.5)).mean()),
        "0.5-0.75": g.apply(lambda x: ((x.iou >= 0.5) & (x.iou < 0.75)).mean()),
    }
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    bottom = pd.Series(0.0, index=src)
    for (name, v), color in zip(parts.items(), ["#999999", "#D55E00", "#F0E442"]):
        v = v.reindex(src)
        ax.bar([LABEL[s] for s in src], v, bottom=bottom, label=name, color=color, edgecolor="white")
        bottom += v
    for i, total in enumerate(bottom):
        ax.text(i, total + 0.01, f"{total:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("share of GT objects with IoU < 0.75")
    ax.set_ylim(0, min(1, bottom.max() + 0.12))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Initial prompts that fail (n = {s0.obj_id.nunique()} objects)", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    return ax.figure


def click_curves(df: pd.DataFrame, ax=None):
    """Best-so-far IoU vs ideal clicks; solid = all detected objects, dashed = initial IoU < 0.75."""
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    for kw, style in (({}, "-"), ({"only_below": 0.75}, "--")):
        c = click_curve(df, **kw)
        for s in _order(c.index):
            row = c.loc[s].drop("n")
            ax.plot(row.index, row.values, style, marker="o", ms=3, color=COLOR[s],
                    label=f"{LABEL[s]}{' (failures)' if kw else ''}")
    ax.set_xlabel("ideal clicks added to the box")
    ax.set_ylabel("mean IoU (best so far)")
    ax.set_title("Headroom: how much ideal clicks fix", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    return ax.figure


def noc_hist(df: pd.DataFrame, thr: float = 0.85, ax=None):
    """Distribution of clicks needed to reach `thr` IoU; the last bin is 'not reached'."""
    w = traces(df)
    budget = int(w.columns.max())
    n = pd.Series([noc(r, thr, budget) for r in w.to_numpy().tolist()], index=w.index)
    src = _order(n.index.get_level_values("prompt_origin").unique())
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    width = 0.8 / len(src)
    for i, s in enumerate(src):
        share = n.xs(s, level="prompt_origin").value_counts(normalize=True).reindex(range(budget + 2), fill_value=0)
        ax.bar(share.index + (i - (len(src) - 1) / 2) * width, share.values, width, color=COLOR[s], label=LABEL[s])
    ax.set_xticks(range(budget + 2), [str(k) for k in range(budget + 1)] + ["not\nreached"])
    ax.set_xlabel(f"ideal clicks to reach IoU {thr:.2f}")
    ax.set_ylabel("share of detected objects")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"NoC@{round(thr * 100)}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return ax.figure


def overconfidence(df: pd.DataFrame, source: str = "gt", ax=None):
    """SAM 3's predicted IoU against the real IoU of the initial mask."""
    s0 = initial(df)
    s0 = s0[(s0.prompt_origin == source) & s0.detected]
    ax = ax or plt.subplots(figsize=(4, 4))[1]
    ax.scatter(s0.iou, s0.pred_iou, s=6, alpha=0.4, color=COLOR[source], linewidths=0)
    ax.plot([0, 1], [0, 1], color="#999999", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("real IoU"); ax.set_ylabel("SAM 3 predicted IoU")
    ax.set_title(f"{LABEL[source]}: predicted {s0.pred_iou.mean():.2f} vs real {s0.iou.mean():.2f}", fontsize=10)
    return ax.figure


NEEDS_CLICKS = {"click_curves", "noc85"}
def method_curves(df: pd.DataFrame, only_below: float | None = 0.75):
    """Mean IoU (raw, not best so far) after k steps of each correction method, one panel per box source."""
    c = step_curve(df, only_below=only_below)
    if c.empty:  # no failures at all: show every object
        c, only_below = step_curve(df), None
    src = _order(c.index.get_level_values("prompt_origin").unique())
    fig, axes = plt.subplots(1, len(src), figsize=(3.6 * len(src), 3.4), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], src):
        for m, (color, label) in METHOD_STYLE.items():
            if (s, m) in c.index:
                row = c.loc[(s, m)].drop("n").dropna()
                ax.plot(row.index, row.values, marker="o", ms=3, color=color, label=label)
        ax.set_title(f"{LABEL[s]} (n = {int(c.loc[s].n.iloc[0])})", fontsize=10)
        ax.set_xlabel("correction steps")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0][0].set_ylabel("mean IoU" + (f" (initial IoU < {only_below})" if only_below else ""))
    axes[0][-1].legend(frameon=False, fontsize=7)
    return fig


FIGURES = {"failure_bars": failure_bars, "click_curves": click_curves, "noc85": noc_hist, "overconfidence": overconfidence,
           "method_curves": method_curves}


def save_all(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        if name in NEEDS_CLICKS and not (df.method == "oracle_clicks").any():
            continue
        if name == "method_curves" and df.method[df.method != "b0_none"].nunique() < 2:
            continue
        fig = fn(df)
        fig.tight_layout()
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    return paths
