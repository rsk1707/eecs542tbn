"""SAM 3 text-prompt detections through P2's exporter. Owner: P1.

`export_detections` (P2) raises on any detection with a non-positive box or a score outside [0, 1], which
would abort a whole export. SAM 3 boxes are clipped to the image by `Sam3Segmenter.detect_text`, so a box
that lies entirely outside the image can collapse to zero width. `RecordingSegmenter` drops such
detections (and counts them), and records per-query facts for the manifest: how many detections each
phrase returned and SAM 3's raw presence probability for that phrase.
"""

from __future__ import annotations

import math


def is_valid_detection(box, score) -> bool:
    """Same validity rule as detectors.export.export_detections."""
    box = list(map(float, box))
    return (len(box) == 4 and all(map(math.isfinite, box)) and box[2] > box[0] and box[3] > box[1]
            and math.isfinite(float(score)) and 0 <= float(score) <= 1)


class RecordingSegmenter:
    """Wraps a Segmenter for export_detections: filters invalid detections, records per-query facts."""

    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.image_id: int | None = None
        self.queries: list[dict] = []

    def set_image(self, image_id: int) -> None:
        self.image_id = image_id
        self.segmenter.set_image(image_id)

    def detect_text(self, phrase: str):
        raw = self.segmenter.detect_text(phrase)
        kept = [d for d in raw if is_valid_detection(d[0], d[2])]
        self.queries.append({
            "image_id": self.image_id, "phrase": phrase, "n_detections": len(kept),
            "n_dropped_invalid": len(raw) - len(kept),
            "presence": getattr(self.segmenter, "last_presence", None),
        })
        return kept
