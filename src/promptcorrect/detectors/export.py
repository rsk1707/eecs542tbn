"""GT-present phrase export through P1's Segmenter or an independent detector."""

from collections import defaultdict
from collections.abc import Callable, Iterable
import json
import math
from pathlib import Path
from typing import Any

from promptcorrect.data.io import new_text_file
from promptcorrect.data.records import ObjectRecord


def export_detections(
    records: Iterable[ObjectRecord], output: str | Path, *,
    detector: Any = None, load_image: Callable[[int], Any] | None = None,
    segmenter: Any = None,
) -> int:
    """Write the documented detection JSONL; empty detections produce no rows.

    Supply either detector + image loader, or a Segmenter with detect_text.
    No matching, metrics, or failure thresholds are implemented here (P3 owns them).
    """
    if (detector is None) == (segmenter is None):
        raise ValueError("Supply exactly one detector or segmenter")
    if detector is not None and load_image is None:
        raise ValueError("Independent detectors require an image loader")
    phrases = defaultdict(set)
    for record in records:
        phrases[record.image_id].add(record.category_name.replace("_", " "))
    count = 0
    with new_text_file(output) as stream:
        for image_id in sorted(phrases):
            image = None
            if segmenter is not None:
                segmenter.set_image(image_id)
            else:
                image = load_image(image_id)
            try:
                for phrase in sorted(phrases[image_id]):
                    detections = (segmenter.detect_text(phrase) if segmenter is not None
                                  else detector.detect(image, phrase))
                    for detection in detections:
                        if segmenter is not None:
                            box, rle, score = detection
                        else:
                            box, score = detection
                        box, score = list(map(float, box)), float(score)
                        if (len(box) != 4 or not all(map(math.isfinite, box))
                                or box[2] <= box[0] or box[3] <= box[1]
                                or not math.isfinite(score) or not 0 <= score <= 1):
                            raise ValueError(f"Invalid detection for image {image_id}: {detection}")
                        row = {"image_id": image_id, "phrase": phrase, "box": box, "score": score}
                        if segmenter is not None:
                            rle = dict(rle)
                            if isinstance(rle["counts"], bytes):
                                rle["counts"] = rle["counts"].decode("ascii")
                            row["mask_rle"] = rle
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        count += 1
            finally:
                if image is not None and hasattr(image, "close"):
                    image.close()
    return count
