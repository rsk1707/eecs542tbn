"""TEMPORARY stand-ins for P4's code, so corrector v0 is not blocked. Owner: P5 until P4's versions land.

P4 owns perturbations (`perturb/`), `describe()` and the rule-based baselines (`baselines/`); all three were
stubs when corrector v0 was written. These follow the descriptions in docs/demo_sprint.md and
docs/interfaces.md. Every run manifest that uses them says so ("standins": ...). Once P4's code exists, switch
the callers to it and delete this file; numbers for the report must come from P4's implementations.

Perturbations are relative to the GT box (CLAUDE.md section 4.4); `level` is the severity knob:
  shift     move the centre by level * (w, h) in a random direction
  scale     multiply w and h by (1 + level) or 1 / (1 + level), coin flip
  truncate  cut level * (side length) off one random side
"""

from __future__ import annotations

import math

import numpy as np

PERTURBATIONS = ("shift", "scale", "truncate")
STANDINS_NOTE = "P5 stand-ins (corrector/standins.py) for P4's perturbations, box_iou and b1/b3 baselines"


def _clip(box: list[float], hw: tuple[int, int], min_side: float = 2.0) -> list[float]:
    H, W = hw
    x0, y0, x1, y1 = (float(np.clip(box[0], 0, W)), float(np.clip(box[1], 0, H)),
                      float(np.clip(box[2], 0, W)), float(np.clip(box[3], 0, H)))
    if x1 - x0 < min_side:
        c = min(max((x0 + x1) / 2, min_side / 2), W - min_side / 2)
        x0, x1 = c - min_side / 2, c + min_side / 2
    if y1 - y0 < min_side:
        c = min(max((y0 + y1) / 2, min_side / 2), H - min_side / 2)
        y0, y1 = c - min_side / 2, c + min_side / 2
    return [x0, y0, x1, y1]


def perturb_box(box: list[float], kind: str, level: float, rng: np.random.Generator,
                hw: tuple[int, int]) -> tuple[list[float], dict]:
    """GT box -> (perturbed box clipped to the image, {"type", "params"}) ."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if kind == "shift":
        a = float(rng.uniform(0, 2 * math.pi))
        dx, dy = level * w * math.cos(a), level * h * math.sin(a)
        out, params = [x0 + dx, y0 + dy, x1 + dx, y1 + dy], {"level": level, "angle": a}
    elif kind == "scale":
        s = (1 + level) if rng.random() < 0.5 else 1 / (1 + level)
        out, params = [cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2], {"level": level, "s": s}
    elif kind == "truncate":
        side = ("left", "top", "right", "bottom")[int(rng.integers(4))]
        out = list(box)
        if side == "left":
            out[0] += level * w
        elif side == "right":
            out[2] -= level * w
        elif side == "top":
            out[1] += level * h
        else:
            out[3] -= level * h
        params = {"level": level, "side": side}
    else:
        raise ValueError(f"unknown perturbation {kind!r}; choose from {PERTURBATIONS}")
    return _clip(out, hw), {"type": kind, "params": params}


def box_iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def random_click(box: list[float], rng: np.random.Generator) -> tuple[float, float, int]:
    """b1_random_k: one positive click uniformly inside the prompt box."""
    return float(rng.uniform(box[0], box[2])), float(rng.uniform(box[1], box[3])), 1
