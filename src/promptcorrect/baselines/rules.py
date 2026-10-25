"""Rule-based correction baselines (method names from P3's registry). Owner: P4.

A rule sees only what a deployed system has: the current prompt and the segmenter's current output (never
the GT). It returns the next prompt, or None to stop (nothing left to do). `rollout` runs a rule for up
to k steps through the Segmenter interface; step 0 is the initial prompt's own mask (= `b0_none`).

  b1_random_k   add one positive click, uniform inside the prompt box (or the current mask's box)
  b2_multimask  one step: rerun the initial prompt with multimask=True, keep SAM's top-scored candidate
  b3_box_refit  replace the box by the tight box of the current mask (grown by `margin` of its size);
                stop once the box no longer changes
  b3_unc_click  click at the centre of the largest region with |logit| < `tau` (full resolution). Label:
                positive if the point is inside the prompt box, negative outside (the rule trusts the box).
                Regions that already hold a click are skipped; the rule stops when none is left.

Clicks accumulate across steps (like the oracle); `Prompt.mask_logits` is never fed back.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from promptcorrect.data.geometry import mask_to_box
from promptcorrect.segmenters.base import Segmenter
from promptcorrect.types import Prompt, SegOutput

# rule(prompt, full-res logits [H, W], current bool mask [H, W], rng) -> next prompt or None
Rule = Callable[[Prompt, np.ndarray, np.ndarray, np.random.Generator], "Prompt | None"]


def next_prompt(prompt: Prompt, method: str, box=None, point=None) -> Prompt:
    """Copy of `prompt` with the box replaced (if `box` is given) and `point` appended (if given)."""
    return Prompt(box=list(prompt.box) if box is None and prompt.box is not None else box,
                  points=list(prompt.points) + ([point] if point else []),
                  origin=f"baseline:{method}", perturbation=prompt.perturbation)


def random_click(prompt: Prompt, logits: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Prompt | None:
    box = prompt.box if prompt.box is not None else (mask_to_box(mask) if mask.any() else None)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return next_prompt(prompt, "b1_random_k", point=(float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)), 1))


def box_refit(prompt: Prompt, logits: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
              margin: float = 0.0) -> Prompt | None:
    if not mask.any():
        return None
    x0, y0, x1, y1 = mask_to_box(mask)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    H, W = mask.shape
    box = [max(0.0, x0 - mx), max(0.0, y0 - my), min(float(W), x1 + mx), min(float(H), y1 + my)]
    if prompt.box is not None and np.allclose(box, prompt.box, atol=0.5):
        return None
    return next_prompt(prompt, "b3_box_refit", box=box)


def uncertainty_click(prompt: Prompt, logits: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                      tau: float = 1.0) -> Prompt | None:
    region = np.abs(logits) < tau
    comp, n = ndimage.label(region)
    if n == 0:
        return None
    sizes = ndimage.sum(region, comp, range(1, n + 1))
    H, W = region.shape
    for x, y, _ in prompt.points:  # an uncertain region that already holds a click is not clicked again
        k = comp[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)]
        if k:
            sizes[k - 1] = 0
    if not sizes.any():
        return None
    biggest = comp == 1 + int(np.argmax(sizes))
    dist = ndimage.distance_transform_edt(np.pad(biggest, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    px, py = float(x), float(y)  # same pixel-index convention as the oracle click
    b = prompt.box
    label = 1 if b is None or (b[0] <= px <= b[2] and b[1] <= py <= b[3]) else 0
    return next_prompt(prompt, "b3_unc_click", point=(px, py, label))


RULES: dict[str, Rule] = {"b1_random_k": random_click, "b3_box_refit": box_refit, "b3_unc_click": uncertainty_click}
METHODS = ("b0_none", "b1_random_k", "b2_multimask", "b3_box_refit", "b3_unc_click")


@dataclass
class Step:
    step: int
    prompt: Prompt
    output: SegOutput
    mask: np.ndarray          # bool [H, W], original resolution
    latency_ms: float         # this step: decoder call + upsampling
    n_decoder_calls: int      # cumulative, including this step


def evaluate(segmenter: Segmenter, prompt: Prompt, step: int = 0, calls_before: int = 0,
             multimask: bool = False) -> tuple[Step, np.ndarray]:
    """One decoder call -> (Step, full-res logits as a float numpy array)."""
    t = time.perf_counter()
    out = segmenter.predict([prompt], multimask=multimask)[0]
    full = segmenter.to_full_res(out.mask_logits).float().cpu().numpy()
    ms = (time.perf_counter() - t) * 1000
    return Step(step, prompt, out, full > 0, ms, calls_before + 1), full


def rollout(segmenter: Segmenter, method: str, prompt: Prompt, steps: int, rng: np.random.Generator | None = None,
            first: tuple[Step, np.ndarray] | None = None, rule: Rule | None = None, **rule_kwargs) -> list[Step]:
    """Run `method` from `prompt` for up to `steps` correction steps on the current image.

    Returns [step 0, step 1, ...]; fewer than steps + 1 entries when the rule stops early. `first` reuses an
    already computed step 0 (e.g. from a batched call). `rule` overrides the registry (used for the oracle).
    """
    if method not in METHODS and rule is None:
        raise ValueError(f"unknown baseline {method!r}; expected one of {METHODS}")
    rng = rng if rng is not None else np.random.default_rng(0)
    s0, logits = first if first is not None else evaluate(segmenter, prompt)
    out = [s0]
    if method == "b0_none" or steps <= 0:
        return out
    if method == "b2_multimask":
        s1, _ = evaluate(segmenter, prompt, 1, s0.n_decoder_calls, multimask=True)
        s1.prompt = next_prompt(prompt, method)
        return out + [s1]
    rule = rule or RULES[method]
    cur = s0
    for k in range(1, steps + 1):
        nxt = rule(cur.prompt, logits, cur.mask, rng, **rule_kwargs)
        if nxt is None:
            break
        cur, logits = evaluate(segmenter, nxt, k, cur.n_decoder_calls)
        out.append(cur)
    return out
