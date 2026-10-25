"""Grounding DINO export job; model inference belongs on a compute node."""

import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

from PIL import Image

from promptcorrect.data.io import new_text_file
from promptcorrect.data.prepare import file_sha256
from promptcorrect.data.records import read_records
from .export import export_detections
from .grounding_dino import GroundingDinoDetector


def run_grounding_dino(
    annotations_file: str | Path, records_file: str | Path, image_root: str | Path,
    output: str | Path, *, model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: str = "cpu", threshold: float = 0.35, text_threshold: float = 0.25,
    seed: int = 0, revision: str | None = None,
) -> dict:
    """Export detections and provenance; a manifest marks a completed export."""
    import numpy as np
    import torch

    output = Path(output)
    manifest_file = output.with_suffix(".manifest.json")
    if output.exists() or manifest_file.exists():
        raise FileExistsError("Detection output or manifest already exists")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with Path(annotations_file).open(encoding="utf-8") as stream:
        images = {image["id"]: image for image in json.load(stream)["images"]}
    records = list(read_records(records_file))
    for image_id in {record.image_id for record in records}:
        if image_id not in images:
            raise ValueError(f"Missing image metadata: {image_id}")
        if not (Path(image_root) / images[image_id]["file_name"]).is_file():
            raise FileNotFoundError(Path(image_root) / images[image_id]["file_name"])
    detector = GroundingDinoDetector(
        model_id, device=device, threshold=threshold, text_threshold=text_threshold,
        revision=revision,
    )

    def load_image(image_id: int) -> Image.Image:
        with Image.open(Path(image_root) / images[image_id]["file_name"]) as image:
            if image.size != (images[image_id]["width"], images[image_id]["height"]):
                raise ValueError(f"Image dimensions disagree with annotations: {image_id}")
            return image.convert("RGB")

    count = export_detections(records, output, detector=detector, load_image=load_image)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    manifest = {
        "model_id": model_id, "revision": revision,
        "resolved_revision": getattr(detector.model.config, "_commit_hash", None),
        "seed": seed, "device": device, "threshold": threshold,
        "text_threshold": text_threshold, "git_commit": git_commit, "dirty": dirty,
        "annotations_sha256": file_sha256(annotations_file),
        "records_sha256": file_sha256(records_file), "detections_sha256": file_sha256(output),
        "n_detections": count, "phrase_policy": "unique categories of retained objects per image",
        "queries": [{"image_id": image_id, "phrase": phrase} for image_id, phrase in
                    sorted({(r.image_id, r.category_name) for r in records})],
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "Pillow")},
    }
    with new_text_file(manifest_file) as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest
