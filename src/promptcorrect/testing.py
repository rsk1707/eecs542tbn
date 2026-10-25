"""Synthetic objects for CPU tests and local development (no COCO needed)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from promptcorrect.data.geometry import mask_to_box, rle_encode
from promptcorrect.types import ObjectRecord

CATEGORIES = ["person", "cat", "chair"]


def _ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def make_synthetic_objects(
    n_images: int = 3, objects_per_image: int = 2, size: tuple[int, int] = (240, 320), seed: int = 0
) -> list[ObjectRecord]:
    """Ellipse objects laid out left-to-right so they do not overlap. image_ids start at 1."""
    rng = np.random.default_rng(seed)
    h, w = size
    records: list[ObjectRecord] = []
    slot = w / objects_per_image
    for image_id in range(1, n_images + 1):
        masks = []
        for k in range(objects_per_image):
            cx = slot * (k + 0.5)
            cy = h * rng.uniform(0.4, 0.6)
            rx = slot * rng.uniform(0.25, 0.4)
            ry = h * rng.uniform(0.2, 0.4)
            masks.append(_ellipse(h, w, cy, cx, ry, rx))
        for k, m in enumerate(masks):
            area = float(m.sum())
            box = mask_to_box(m)
            boundary = float((m & ~ndimage.binary_erosion(m)).sum())
            cat = CATEGORIES[k % len(CATEGORIES)]
            same = [mask_to_box(o) for j, o in enumerate(masks) if j != k and CATEGORIES[j % len(CATEGORIES)] == cat]
            bw = box[2] - box[0]
            cxk = (box[0] + box[2]) / 2
            n_nb = sum(1 for b in same if abs((b[0] + b[2]) / 2 - cxk) <= bw)
            records.append(
                ObjectRecord(
                    obj_id=f"coco:{image_id * 100 + k}",
                    image_id=image_id,
                    source="coco",
                    category_id=k % len(CATEGORIES) + 1,
                    category_name=cat,
                    novelty="seen",
                    gt_box=box,
                    gt_mask=rle_encode(m),
                    area=area,
                    size_bin="small" if area < 32**2 else ("medium" if area < 96**2 else "large"),
                    thinness=boundary**2 / area,
                    n_neighbors=n_nb,
                )
            )
    return records
