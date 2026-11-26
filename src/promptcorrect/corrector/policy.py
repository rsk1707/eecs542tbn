"""Corrector v1 decision: where to click, and whether to click at all. Owner: P5.

For one object state (image, current full-res logits, prompt box, clicks so far) the model gives an error map
and, with the quality head, a predicted IoU `q` of the current mask. The policy:

  1. candidate click = the v0 rule (clicks.py) on the error map at `err_threshold` (None if no error region);
  2. click only if `q < quality_threshold` (the learned abstain decision; skipped when the threshold is None
     or the model has no quality head, which is v0 behaviour).

Both thresholds are chosen on held-out *training* images by calibrate.py, never on the val objects.
`feedback=True` sends the previous low-res logits back to the segmenter as `Prompt.mask_logits` together with
the box and all clicks, as SAM's interactive mode (and the ideal-click oracle) does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from promptcorrect.corrector.clicks import propose_click
from promptcorrect.corrector.crops import box_mask, clicks_to_crop, crop_object, crop_to_image, crop_window
from promptcorrect.corrector.model import CorrectorNet, build_inputs
from promptcorrect.corrector.samples import MAX_CLICKS
from promptcorrect.types import Prompt, SegOutput


@dataclass
class View:
    """The model's output for one state: error probabilities in the crop, predicted IoU, crop window."""
    probs: np.ndarray     # float [2, S, S] (FP, FN)
    q: float              # predicted IoU of the current mask (NaN without the quality head)
    window: list[float]
    image_hw: tuple[int, int]


class CorrectorPolicy:
    def __init__(self, model: CorrectorNet, device, crop_size: int = 128, context: float = 0.5,
                 err_threshold: float = 0.5, quality_threshold: float | None = None, feedback: bool = True):
        self.model, self.device = model.eval(), device
        self.crop_size, self.context = crop_size, context
        self.err_threshold, self.quality_threshold, self.feedback = err_threshold, quality_threshold, feedback

    @torch.no_grad()
    def view(self, image_t: torch.Tensor, logits_full: torch.Tensor, box: list[float],
             points: list[tuple[float, float, int]] = ()) -> View:
        """`image_t` uint8 [3, H, W]; `logits_full` [H, W] from `segmenter.to_full_res`."""
        S = self.crop_size
        window = crop_window(box, self.context)
        c = crop_object(image_t, logits_full, None, window, S)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None].to(self.device)  # noqa: E731
        x = build_inputs(t(c["rgb"]), t(c["logits"]), t(box_mask(box, window, S)), t(c["valid"]),
                         t(clicks_to_crop(list(points), window, S, MAX_CLICKS)))
        maps, q = self.model.forward_all(x)
        probs = torch.sigmoid(maps.float())[0].cpu().numpy()
        qv = float(torch.sigmoid(q.float())[0]) if q is not None else float("nan")
        return View(probs, qv, window, tuple(image_t.shape[-2:]))

    def candidate(self, v: View, err_threshold: float | None = None) -> tuple[float, float, int] | None:
        """Error-map click in image coordinates (x, y, label), ignoring the quality gate."""
        click = propose_click(v.probs, self.err_threshold if err_threshold is None else err_threshold)
        if click is None:
            return None
        u, vv, label = click
        x, y = crop_to_image(u, vv, v.window, self.crop_size)
        H, W = v.image_hw
        return float(np.clip(x, 0, W - 1e-3)), float(np.clip(y, 0, H - 1e-3)), int(label)

    def gate(self, v: View) -> bool:
        """True = the quality head allows a click."""
        if self.quality_threshold is None or np.isnan(v.q):
            return True
        return v.q < self.quality_threshold

    def decide(self, v: View) -> tuple[float, float, int] | None:
        return self.candidate(v) if self.gate(v) else None

    def next_prompt(self, prompt: Prompt, points: list[tuple[float, float, int]], prev: SegOutput) -> Prompt:
        return Prompt(box=list(prompt.box), points=list(points), origin="corrector",
                      mask_logits=prev.mask_logits if self.feedback else None, perturbation=prompt.perturbation)
