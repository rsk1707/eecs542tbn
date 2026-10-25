"""P2 data preparation orchestration, independent of the pending P1 scaffold."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .io import new_text_file
from .records import build_coco_records, write_records
from .splits import demo_split, write_split


def rooted_path(paths_file: str | Path, root: str, path: str | Path) -> Path:
    """Resolve artifact paths against a named root in the team's paths YAML."""
    paths_file = Path(paths_file).resolve()
    with paths_file.open(encoding="utf-8") as stream:
        roots = {str(k).lower(): v for k, v in yaml.safe_load(stream).items()}  # DATA_ROOT == data_root
    if root.lower() not in roots:
        raise KeyError(f"{root} not found in {paths_file} (keys are case-insensitive)")
    base = Path(roots[root.lower()]).expanduser()
    if not base.is_absolute():
        base = paths_file.parent / base
    return base / path


def file_sha256(path: str | Path) -> str:
    """Hash a source artifact for reproducible data/detection handoffs."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_demo(
    annotations_file: str | Path, split_file: str | Path, objects_file: str | Path,
    *, count: int = 500, seed: int = 0,
) -> dict[str, Any]:
    """Create a fixed demo subset and records, recording provenance separately."""
    split_file, objects_file = Path(split_file), Path(objects_file)
    manifest_file = objects_file.with_suffix(".manifest.json")
    if objects_file.exists() or manifest_file.exists():
        raise FileExistsError("Object output or manifest already exists; choose a new output")
    with Path(annotations_file).open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    split = demo_split((image["id"] for image in annotations["images"]), count, seed)
    if split_file.exists():
        if json.loads(split_file.read_text(encoding="utf-8")) != split:
            raise ValueError("Existing fixed split differs from requested seed/population/count")
    else:
        write_split(split, split_file)
    counts = Counter()

    def counted_records():
        for record in build_coco_records(annotations, split["val"]):
            counts[record.size_bin] += 1
            yield record

    write_records(counted_records(), objects_file)
    manifest = {
        "source": "coco", "split": "val2017", "seed": seed,
        "n_images": count, "n_objects": sum(counts.values()), "size_bins": dict(counts),
        "annotations_sha256": file_sha256(annotations_file),
        "splits_sha256": file_sha256(split_file), "records_sha256": file_sha256(objects_file),
        "filter": "iscrowd == 0 and not (box_w < 32 and box_h < 32)",
        "thinness": "squared pixel-edge perimeter (including holes) / decoded mask area",
        "neighbors": "retained same-class centers at Euclidean distance <= reference box width",
        "phrase_policy": "unique categories of retained objects per image",
    }
    with new_text_file(manifest_file) as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest
