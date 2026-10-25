"""COCO object records using original-image coordinates and JSON-safe RLE."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from .io import new_text_file


@dataclass
class ObjectRecord:
    """P2-owned contract from docs/interfaces.md, section 1."""

    obj_id: str
    image_id: int
    source: str
    category_id: int
    category_name: str
    novelty: str
    gt_box: list[float]
    gt_mask: dict
    area: float
    size_bin: str
    thinness: float
    n_neighbors: int


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    """Convert continuous COCO pixel edges once, without rounding."""
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid COCO box: {[x, y, w, h]}")
    return [x, y, x + w, y + h]


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a 2D binary mask with ASCII counts suitable for JSONL."""
    if mask.ndim != 2:
        raise ValueError("Expected one 2D mask")
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def decode_rle(rle: dict) -> np.ndarray:
    """Decode compressed or uncompressed COCO RLE into a boolean mask."""
    rle = dict(rle)
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects(rle, *rle["size"])
    elif isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def annotation_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Support COCO polygons and both RLE variants, including multipart masks."""
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return decode_rle(mask_utils.merge(rles))
    mask = decode_rle(segmentation)
    if mask.shape != (height, width):
        raise ValueError("RLE dimensions disagree with image metadata")
    return mask


def mask_thinness(mask: np.ndarray) -> float:
    """Squared pixel-edge contour length / mask area, including holes."""
    area = int(mask.sum())
    if area == 0:
        raise ValueError("Thinness is undefined for an empty mask")
    padded = np.pad(mask.astype(np.int8), 1)
    perimeter = int(np.abs(np.diff(padded, axis=0)).sum())
    perimeter += int(np.abs(np.diff(padded, axis=1)).sum())
    return perimeter**2 / area


def build_coco_records(
    annotations: dict, image_ids: Iterable[int] | None = None
) -> Iterator[ObjectRecord]:
    """Yield records sorted by image/annotation ID, filtering once per object.

    Demo O1: drop crowds and boxes with BOTH dimensions below 32 pixels.
    Neighbor counts use retained same-class objects, Euclidean center distance
    <= the reference object's box width (therefore counts can be asymmetric).
    """
    images = {im["id"]: im for im in annotations["images"]}
    categories = {cat["id"]: cat["name"] for cat in annotations["categories"]}
    selected = set(images if image_ids is None else image_ids)
    if selected - images.keys():
        raise ValueError("Selected image IDs are absent from annotations")
    grouped = defaultdict(list)
    for ann in annotations["annotations"]:
        if ann["image_id"] not in selected or ann.get("iscrowd", 0):
            continue
        box = xywh_to_xyxy(ann["bbox"])
        if ann["bbox"][2] < 32 and ann["bbox"][3] < 32:
            continue
        grouped[ann["image_id"]].append((ann, box))
    for image_id in sorted(grouped):
        image = images[image_id]
        records = []
        for ann, box in sorted(grouped[image_id], key=lambda item: item[0]["id"]):
            mask = annotation_mask(ann["segmentation"], image["height"], image["width"])
            if not mask.any():
                raise ValueError(f"Empty mask for coco:{ann['id']}")
            area = float(ann["area"])
            if not math.isfinite(area) or area <= 0:
                raise ValueError(f"Invalid area for coco:{ann['id']}")
            records.append(ObjectRecord(
                obj_id=f"coco:{ann['id']}", image_id=image_id, source="coco",
                category_id=ann["category_id"],
                category_name=categories[ann["category_id"]].replace("_", " "),
                novelty="seen", gt_box=box, gt_mask=encode_rle(mask), area=area,
                size_bin="small" if area < 32**2 else "medium" if area < 96**2 else "large",
                thinness=mask_thinness(mask), n_neighbors=0,
            ))
        for record in records:
            x1, y1, x2, y2 = record.gt_box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            record.n_neighbors = sum(
                other.obj_id != record.obj_id
                and other.category_id == record.category_id
                and math.hypot((other.gt_box[0] + other.gt_box[2]) / 2 - cx,
                               (other.gt_box[1] + other.gt_box[3]) / 2 - cy) <= x2 - x1
                for other in records
            )
            yield record


def write_records(records: Iterable[ObjectRecord], path: str | Path) -> None:
    """Write records without partial files or accidental replacement."""
    with new_text_file(path) as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), allow_nan=False) + "\n")


def read_records(path: str | Path) -> Iterator[ObjectRecord]:
    """Read the shared JSONL schema without decoding masks eagerly."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield ObjectRecord(**json.loads(line))
