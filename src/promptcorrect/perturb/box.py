"""Box perturbations relative to the GT box. Owner: P4.

Every generator takes a box (xyxy, continuous pixel edges) and the image size `(H, W)` and returns a
`Prompt(origin="perturbed", perturbation={"type", "level", "params"})`. `level` is the one number a
sensitivity curve is plotted against; it means the same thing for every object:

  shift     displacement as a fraction of the box size: offset = level * (w cos a, h sin a)
  scale     side-length factor around the box centre (0.5 = half the width and height, 2.0 = double)
  truncate  fraction of the box removed from one side (0.3 = 30% of the width or height cut off)

Boxes are clipped to the image (and kept at least `MIN_SIZE` px wide and high); `params` records what was
actually applied. To compare *across* types, use `describe()` (perturb/severity.py), never `level`.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from promptcorrect.types import Prompt

TYPES = ("shift", "scale", "truncate")
SIDES = ("left", "right", "top", "bottom")
IDENTITY = {"shift": 0.0, "scale": 1.0, "truncate": 0.0}  # the level that leaves the box unchanged
MIN_SIZE = 1.0

# Sensitivity-sweep grid (~18 boxes per object with the clean box). Shift direction and truncated side
# are drawn per object and level from `object_rng`, so the curves average over directions.
SWEEP_LEVELS = {
    "shift": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    "scale": (0.5, 0.67, 0.8, 1.25, 1.5, 2.0),
    "truncate": (0.1, 0.2, 0.3, 0.4, 0.5),
}
# Ranges for random training perturbations (`sample_perturbation`); scale is drawn log-uniformly.
TRAIN_RANGES = {"shift": (0.0, 0.5), "scale": (0.5, 2.0), "truncate": (0.0, 0.5)}


def object_rng(seed: int, *keys: str | int) -> np.random.Generator:
    """Generator that depends only on (seed, keys), not on iteration order (e.g. keys = obj_id, pert_id)."""
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng([seed, *ints])


def clip_box(box: list[float], image_hw: tuple[int, int], min_size: float = MIN_SIZE) -> list[float]:
    """Clip to [0, W] x [0, H] and keep at least `min_size` px per side (a box pushed off the image
    collapses onto the border instead of becoming empty)."""
    H, W = image_hw
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(max(x0, 0.0), W - min_size), min(max(x1, min_size), W)
    y0, y1 = min(max(y0, 0.0), H - min_size), min(max(y1, min_size), H)
    if x1 - x0 < min_size:
        x1 = x0 + min_size
    if y1 - y0 < min_size:
        y1 = y0 + min_size
    return [x0, y0, x1, y1]


def _prompt(raw: list[float], image_hw, ptype: str, level: float, params: dict) -> Prompt:
    box = clip_box(raw, image_hw)
    params["clipped"] = bool(np.any(np.abs(np.subtract(box, raw)) > 1e-6))
    return Prompt(box=box, origin="perturbed", perturbation={"type": ptype, "level": float(level), "params": params})


def shift_box(box: list[float], level: float, angle: float, image_hw: tuple[int, int]) -> Prompt:
    """Move the box by `level` box sizes in direction `angle` (radians; 0 = right, pi/2 = down)."""
    x0, y0, x1, y1 = box
    dx, dy = level * (x1 - x0) * math.cos(angle), level * (y1 - y0) * math.sin(angle)
    return _prompt([x0 + dx, y0 + dy, x1 + dx, y1 + dy], image_hw, "shift", level,
                   {"angle": float(angle), "dx": float(dx), "dy": float(dy)})


def scale_box(box: list[float], factor: float, image_hw: tuple[int, int]) -> Prompt:
    """Scale width and height by `factor` around the box centre."""
    if factor <= 0:
        raise ValueError(f"scale factor must be positive, got {factor}")
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, factor * (x1 - x0) / 2, factor * (y1 - y0) / 2
    return _prompt([cx - hw, cy - hh, cx + hw, cy + hh], image_hw, "scale", factor, {"factor": float(factor)})


def truncate_box(box: list[float], level: float, side: str, image_hw: tuple[int, int]) -> Prompt:
    """Cut `level` of the box off one side ("left" | "right" | "top" | "bottom")."""
    if not 0 <= level < 1:
        raise ValueError(f"truncation level must be in [0, 1), got {level}")
    x0, y0, x1, y1 = box
    cut_w, cut_h = level * (x1 - x0), level * (y1 - y0)
    new = {"left": [x0 + cut_w, y0, x1, y1], "right": [x0, y0, x1 - cut_w, y1],
           "top": [x0, y0 + cut_h, x1, y1], "bottom": [x0, y0, x1, y1 - cut_h]}
    if side not in new:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return _prompt(new[side], image_hw, "truncate", level, {"side": side})


def perturb(box: list[float], ptype: str, level: float, image_hw: tuple[int, int],
            rng: np.random.Generator) -> Prompt:
    """One perturbation of `ptype` at `level`; the free direction (shift angle, truncated side) comes from `rng`."""
    if ptype == "shift":
        return shift_box(box, level, float(rng.uniform(0, 2 * math.pi)), image_hw)
    if ptype == "scale":
        return scale_box(box, level, image_hw)
    if ptype == "truncate":
        return truncate_box(box, level, SIDES[int(rng.integers(len(SIDES)))], image_hw)
    raise ValueError(f"unknown perturbation {ptype!r} (expected one of {TYPES})")


def sample_perturbation(box: list[float], image_hw: tuple[int, int], rng: np.random.Generator,
                        types: tuple[str, ...] = TYPES, ranges: dict | None = None) -> Prompt:
    """Random perturbation for training data: type uniform over `types`, level uniform in its range
    (log-uniform for scale)."""
    ranges = {**TRAIN_RANGES, **(ranges or {})}
    ptype = types[int(rng.integers(len(types)))]
    lo, hi = ranges[ptype]
    level = math.exp(rng.uniform(math.log(lo), math.log(hi))) if ptype == "scale" else float(rng.uniform(lo, hi))
    return perturb(box, ptype, level, image_hw, rng)


def sweep_prompts(box: list[float], image_hw: tuple[int, int], rng: np.random.Generator,
                  levels: dict[str, tuple[float, ...]] = SWEEP_LEVELS) -> list[Prompt]:
    """The clean box (origin "gt", no perturbation) followed by one prompt per (type, level) in `levels`."""
    prompts = [Prompt(box=[float(v) for v in box], origin="gt")]
    for ptype, lv in levels.items():
        prompts.extend(perturb(box, ptype, level, image_hw, rng) for level in lv)
    return prompts
