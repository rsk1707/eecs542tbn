"""SAM 2.1 backend (fallback segmenter). TODO(P1): implement against the Segmenter interface.

`presence_score` is None and `detect_text` is unsupported on this backend.
"""

from __future__ import annotations

from promptcorrect.segmenters.base import Segmenter


class Sam21Segmenter(Segmenter):
    backend = "sam21"

    def __init__(self, ckpt_root=None, device: str = "cuda"):
        raise NotImplementedError("SAM 2.1 wrapper not implemented yet (P1)")

    def set_image(self, image_id):
        raise NotImplementedError

    def predict(self, prompts, multimask=False):
        raise NotImplementedError

    def detect_text(self, phrase):
        raise NotImplementedError("SAM 2.1 has no text prompting; use a detector (Grounding DINO / LLMDet)")

    def to_full_res(self, logits):
        raise NotImplementedError
