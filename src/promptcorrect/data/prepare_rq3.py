"""Reproducible, leakage-safe COCO train/val and LVIS-val handoff. Owner: P2."""

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

from .io import new_text_file
from .lvis import build_lvis_records, draft_category_review, validate_category_review
from .prepare import file_sha256
from .records import build_coco_records, write_records
from .splits import training_split, write_split


def _read(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write(value: dict, path: str | Path) -> None:
    with new_text_file(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_category_review(coco_file: str | Path, lvis_file: str | Path,
                          synsets_file: str | Path, output: str | Path) -> dict:
    """Write an unreviewed candidate map; annotation files stay outside git."""
    draft = draft_category_review(_read(coco_file)["categories"],
                                  _read(lvis_file)["categories"], _read(synsets_file))
    draft["sources_sha256"] = {
        "coco": file_sha256(coco_file), "lvis": file_sha256(lvis_file),
        "coco_synsets": file_sha256(synsets_file),
    }
    _write(draft, output)
    return {"n_categories": len(draft["categories"]),
            "n_exact_synset_candidates": sum(r["novelty"] == "seen" for r in draft["categories"]),
            "review_complete": False}


def prepare_rq3(coco_train_file: str | Path, coco_val_file: str | Path,
                lvis_val_file: str | Path, review_file: str | Path,
                output_dir: str | Path, *, train_count: int | None = None,
                seed: int = 0) -> dict:
    """Build fixed splits and records; manifest.json marks a completed directory.

    Full COCO val and LVIS val are retained. Optional training subsampling occurs
    only AFTER excluding ALL LVIS-val images, including images with no retained
    objects. No images or checkpoints are needed. Existing directories are refused.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    inputs = {"coco_train": coco_train_file, "coco_val": coco_val_file,
              "lvis_val": lvis_val_file, "category_review": review_file}
    hashes = {key: file_sha256(path) for key, path in inputs.items()}
    review = _read(review_file)
    lvis = _read(lvis_val_file)
    val = _read(coco_val_file)
    train = _read(coco_train_file)
    if train["categories"] != val["categories"]:
        # Ordering is irrelevant, but category identity must agree.
        if sorted(train["categories"], key=lambda c: c["id"]) != sorted(val["categories"], key=lambda c: c["id"]):
            raise ValueError("COCO train/val categories disagree")
    if review.get("sources_sha256", {}).get("coco") != hashes["coco_val"] or review.get(
        "sources_sha256", {}
    ).get("lvis") != hashes["lvis_val"]:
        raise ValueError("Category review was drafted for different COCO-val/LVIS annotations")
    novelty = validate_category_review(review, val["categories"], lvis["categories"])
    split = training_split((im["id"] for im in train["images"]),
                           (im["id"] for im in val["images"]),
                           (im["id"] for im in lvis["images"]))
    n_eligible = len(split["train"])
    if train_count is not None:
        if not 0 < train_count <= n_eligible:
            raise ValueError("train_count must be positive and no larger than the leakage-safe pool")
        split["train"] = sorted(random.Random(seed).sample(split["train"], train_count))
    # Reserve the directory; a failed run never has a completion manifest.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_split(split, output_dir / "splits.json")
    _write(review, output_dir / "category_review.json")
    builders = {
        "objects_train.jsonl": (build_coco_records(train, split["train"]), len(split["train"])),
        "objects_val.jsonl": (build_coco_records(val, split["val"]), len(split["val"])),
        "objects_lvis_val.jsonl": (build_lvis_records(lvis, novelty), len(lvis["images"])),
    }
    inventory = {}
    for filename, (records, n_images) in builders.items():
        counts, sizes, categories, images = Counter(), Counter(), Counter(), set()

        def counted():
            for record in records:
                counts[record.novelty] += 1
                sizes[record.size_bin] += 1
                categories[str(record.category_id)] += 1
                images.add(record.image_id)
                yield record

        write_records(counted(), output_dir / filename)
        inventory[filename] = {"n_selected_images": n_images, "n_images_with_objects": len(images),
                               "n_objects": sum(counts.values()), "novelty": dict(counts),
                               "size_bins": dict(sizes), "categories": dict(categories)}
    manifest = {
        "schema_version": 1, "seed": seed, "train_count": train_count,
        "n_eligible_train_images": n_eligible,
        "n_lvis_val_excluded_from_train": len(split["lvis_val_excluded_from_train"]),
        "sources_sha256": hashes, "inventory": inventory,
        "artifacts_sha256": {name: file_sha256(output_dir / name) for name in
                             ["splits.json", "category_review.json", *builders]},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "filter": "iscrowd == 0 and not (box_w < 32 and box_h < 32)",
        "novelty_policy": "explicit reviewed category decisions; relative to COCO corrector training",
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pycocotools")},
        "thinness": "squared pixel-edge perimeter (including holes) / decoded mask area",
        "neighbors": "retained same-class centers at Euclidean distance <= reference box width",
    }
    _write(manifest, output_dir / "manifest.json")
    return manifest
