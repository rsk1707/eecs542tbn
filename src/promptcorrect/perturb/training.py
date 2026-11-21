"""Training-data variants for the corrector (`RunConfig.train_data`). Owner: P4.

The corrector's samples (P5's `corrector/samples.py`) are frozen-segmenter states for one prompt each. This
module decides *which prompts* an object contributes, so P5's generator stays variant-agnostic:

  synthetic  the clean GT box (optional) plus `n_perturbed` random shift / scale / truncate boxes from
             `sample_perturbation` (levels drawn from TRAIN_RANGES; one random type per box)
  real       real detector / SAM 3 text failures mined by P2 on COCO train         (not available yet)
  mixed      real + synthetic                                                      (not available yet)
  matched    the objects and count of `real`, with synthetic perturbations instead (not available yet)

Prompts are drawn from `object_rng(seed, obj_id, "train")`, so an object's prompts do not depend on which
other objects or images were processed. Every sample's metadata records the perturbation and its severity
via `describe()`, the same function the sweep and RQ2 use.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from promptcorrect.perturb.box import TYPES, object_rng, perturbation_fields, sample_perturbation
from promptcorrect.perturb.severity import describe, severity_columns
from promptcorrect.types import ObjectRecord, Prompt

TRAIN_DATA = ("synthetic", "real", "mixed", "matched")

# prompt_fn(record, all records of its image) -> prompts for that object
PromptFn = Callable[[ObjectRecord, list[ObjectRecord]], list[Prompt]]


def synthetic_prompts(rec: ObjectRecord, seed: int = 0, n_perturbed: int = 1, include_clean: bool = True,
                      types: tuple[str, ...] = TYPES, ranges: dict | None = None) -> list[Prompt]:
    """The clean GT box (if `include_clean`) followed by `n_perturbed` random perturbations of it."""
    hw = tuple(rec.gt_mask["size"])
    rng = object_rng(seed, rec.obj_id, "train")
    prompts = [Prompt(box=[float(v) for v in rec.gt_box], origin="gt")] if include_clean else []
    prompts += [sample_perturbation(rec.gt_box, hw, rng, types, ranges) for _ in range(n_perturbed)]
    return prompts


def make_prompt_fn(train_data: str = "synthetic", **kwargs) -> PromptFn:
    """Prompt generator for one `train_data` variant. kwargs go to `synthetic_prompts`."""
    if train_data not in TRAIN_DATA:
        raise ValueError(f"unknown train_data {train_data!r}; expected one of {TRAIN_DATA}")
    if train_data != "synthetic":
        raise NotImplementedError(f"train_data={train_data!r} needs P2's mined real failures on COCO train "
                                  "(docs/next_steps.md, P2 task 3); only 'synthetic' is available now")
    return lambda rec, objs: synthetic_prompts(rec, **kwargs)


def sample_meta(prompt: Prompt, rec: ObjectRecord, neighbors: list[ObjectRecord],
                mask_cache: dict[str, np.ndarray] | None = None) -> dict:
    """Per-sample metadata: the perturbation (pert_type, pert_level, pert_dir, clipped) and its severity
    (box_iou_in, coverage, contamination, delta_*), JSON-safe (NaN only for point-only prompts)."""
    return {**perturbation_fields(prompt),
            **severity_columns(describe(prompt, rec, neighbors, mask_cache=mask_cache))}


class SeverityMeta:
    """`extra_meta` for P5's `generate_samples`: `sample_meta` with a GT-mask cache that is emptied whenever
    the image changes, so memory stays at one image's masks."""

    def __init__(self):
        self._image_id, self._cache = None, {}

    def __call__(self, prompt: Prompt, rec: ObjectRecord, neighbors: list[ObjectRecord]) -> dict:
        if rec.image_id != self._image_id:
            self._image_id, self._cache = rec.image_id, {}
        return sample_meta(prompt, rec, neighbors, self._cache)
