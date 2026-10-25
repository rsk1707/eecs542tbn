"""Fixed image-level splits, including the LVIS validation leakage guard."""

from collections.abc import Iterable
import json
from pathlib import Path
import random

from .io import new_text_file


def demo_split(image_ids: Iterable[int], count: int = 500, seed: int = 0) -> dict:
    """Sample from sorted unique IDs, independent of annotation ordering."""
    population = sorted(set(image_ids))
    if not 0 < count <= len(population):
        raise ValueError("Demo count must be positive and no larger than the image population")
    return {"train": [], "val": sorted(random.Random(seed).sample(population, count)),
            "lvis_val_excluded_from_train": []}


def training_split(
    coco_train_ids: Iterable[int], coco_val_ids: Iterable[int], lvis_val_ids: Iterable[int]
) -> dict:
    """Require LVIS IDs explicitly; never silently build a contaminated split."""
    train, val, lvis = set(coco_train_ids), set(coco_val_ids), set(lvis_val_ids)
    if train & val:
        raise ValueError("COCO train and validation images overlap")
    return {"train": sorted(train - lvis), "val": sorted(val),
            "lvis_val_excluded_from_train": sorted(train & lvis)}


def write_split(split: dict, path: str | Path) -> None:
    """Fixed split files are created once and never overwritten."""
    with new_text_file(path) as stream:
        json.dump(split, stream, indent=2, allow_nan=False)
        stream.write("\n")
