"""SAM 3 backend. TODO(P1, Srikrishnan): implement against the Segmenter interface.

To verify with the smoke test (record results in CLAUDE.md section 12):
* interactive point/box predictor API (SAM 2-style) on an A40;
* box conventions: the text/geometric processor may use normalized cxcywh while the interactive
  predictor uses xyxy pixels; hide that here (issue O13);
* where `presence_score` comes from (probably the detector/text path only, issue O6);
* demo mode runs the encoder live in `set_image` (no feature cache yet, issue O7).
"""

from __future__ import annotations

from promptcorrect.segmenters.base import Segmenter


class Sam3Segmenter(Segmenter):
    backend = "sam3"

    def __init__(self, ckpt_root=None, device: str = "cuda"):
        raise NotImplementedError("SAM 3 wrapper not implemented yet (P1)")

    def set_image(self, image_id):
        raise NotImplementedError

    def predict(self, prompts, multimask=False):
        raise NotImplementedError

    def detect_text(self, phrase):
        raise NotImplementedError

    def to_full_res(self, logits):
        raise NotImplementedError
