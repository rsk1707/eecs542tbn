"""Box/RLE helpers. Convert once at load time, never again.

The implementations live in `records.py` (P2); these are the stable names the rest of the repo uses.
"""

from __future__ import annotations

import numpy as np

from promptcorrect.data.records import decode_rle as rle_decode  # noqa: F401  (re-export)
from promptcorrect.data.records import encode_rle as rle_encode  # noqa: F401  (re-export)
from promptcorrect.data.records import xywh_to_xyxy  # noqa: F401  (re-export)


def mask_to_box(mask: np.ndarray) -> list[float]:
    """Tight xyxy box around a boolean mask (continuous edges: max index + 1)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
