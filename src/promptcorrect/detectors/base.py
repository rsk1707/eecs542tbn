"""Detector interface. TODO(P2): Grounding DINO wrapper (transformers) first; LLMDet time-boxed."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, image_rgb: np.ndarray, phrase: str) -> list[tuple[list[float], float]]:
        """image_rgb [H, W, 3] uint8, phrase -> [(box xyxy absolute pixels, score), ...]."""
