"""The Segmenter interface (docs/interfaces.md section 3). Owner: P1.

Everything else in the repo talks to a segmenter only through this class. Nobody outside `segmenters/`
resizes, pads or rescales logits or coordinates: use `to_full_res`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from promptcorrect.types import Prompt, SegOutput, TextDetection

if TYPE_CHECKING:  # annotations only: importing torch is slow on a cold filesystem and plot/eval scripts don't need it
    from torch import Tensor


class Segmenter(ABC):
    backend: str = "base"

    @abstractmethod
    def set_image(self, image_id: int) -> None:
        """Select the image for subsequent calls (loads cached features, or runs the encoder live)."""

    @abstractmethod
    def predict(self, prompts: list[Prompt], multimask: bool = False) -> list[SegOutput]:
        """One decoder run per prompt for the current image. Run under torch.no_grad() by default.

        `mask_logits` is the decoder's native low-res logits, shape [h, w]. With multimask=True,
        `alt_logits` / `alt_ious` carry all candidates and `mask_logits` is the best one (issue O5).
        """

    @abstractmethod
    def detect_text(self, phrase: str) -> list[TextDetection]:
        """Text prompt -> [(box xyxy, mask RLE, score), ...] for the current image. SAM 3 only."""

    @abstractmethod
    def to_full_res(self, logits: Tensor) -> Tensor:
        """Upsample low-res logits to the original image size [H, W]. Binary mask = result > 0."""

    def full_res_mask(self, output: SegOutput) -> np.ndarray:
        """Convenience: boolean mask [H, W] at original resolution."""
        return (self.to_full_res(output.mask_logits) > 0).cpu().numpy()
