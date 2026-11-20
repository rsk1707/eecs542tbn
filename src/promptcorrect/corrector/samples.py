"""Corrector v0 training samples: one frozen-segmenter state per sample, stored as object-centric crops.
Owner: P5. Format: docs/p5_corrector_v0.md.

v0 uses only code that exists in the repo: every object's clean GT box goes through the frozen segmenter
and the result is one sample, so the errors the corrector learns are SAM 3's own. Perturbed boxes (P4's
`perturb/`) and states after ideal clicks (P3's simulator) are added once those exist; the `clicks` array
is already in the format and is empty (label -1) for now.

Layout under `out_dir`: shard_XXXX.npz (rgb, logits, gt, valid, box, clicks), meta.jsonl (one row per sample,
`shard` / `row` point into the npz), manifest.json.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from promptcorrect.corrector.crops import box_mask, clicks_to_crop, crop_object, crop_window
from promptcorrect.data.geometry import rle_decode
from promptcorrect.data.records import build_coco_records, group_by_image
from promptcorrect.data.splits import training_split
from promptcorrect.eval.metrics import iou
from promptcorrect.types import ObjectRecord, Prompt

ARRAYS = ("rgb", "logits", "gt", "valid", "box", "clicks")
MAX_CLICKS = 4  # click slots stored per sample (most recent kept)


def train_subset_records(train_annotations: dict, val_annotations: dict, lvis_val: dict, n_images: int,
                         seed: int = 0) -> tuple[list[ObjectRecord], dict]:
    """Records for `n_images` COCO train images, sampled AFTER removing LVIS-val images (CLAUDE.md 4.5),
    using P2's split and record builder. Returns (records, split info for the manifest)."""
    split = training_split((im["id"] for im in train_annotations["images"]),
                           (im["id"] for im in val_annotations["images"]),
                           (im["id"] for im in lvis_val["images"]))
    with_objects = {a["image_id"] for a in train_annotations["annotations"] if not a.get("iscrowd", 0)}
    pool = [i for i in split["train"] if i in with_objects]
    chosen = sorted(random.Random(seed).sample(pool, n_images))
    records = list(build_coco_records(train_annotations, chosen))
    info = {"n_train_pool": len(pool), "n_lvis_val_excluded": len(split["lvis_val_excluded_from_train"]),
            "image_ids": sorted({r.image_id for r in records}), "seed": seed}
    return records, info


def load_coco_image(image_dirs: list[Path]) -> Callable[[int], np.ndarray]:
    from PIL import Image

    def load(image_id: int) -> np.ndarray:
        name = f"{image_id:012d}.jpg"
        for d in image_dirs:
            if (d / name).exists():
                return np.asarray(Image.open(d / name).convert("RGB"))
        raise FileNotFoundError(name)
    return load


class _ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir, self.shard_size = out_dir, shard_size
        self.buf: dict[str, list] = {k: [] for k in ARRAYS}
        self.meta: list[dict] = []
        self.shard, self.n = 0, 0
        out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = (out_dir / "meta.jsonl").open("w", encoding="utf-8")

    def add(self, arrays: dict, meta: dict) -> None:
        for k in ARRAYS:
            self.buf[k].append(arrays[k])
        self.meta.append({**meta, "shard": self.shard, "row": len(self.meta), "sample": self.n})
        self.n += 1
        if len(self.meta) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.meta:
            return
        path = self.out_dir / f"shard_{self.shard:04d}.npz"
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez_compressed(tmp, **{k: np.stack(v) for k, v in self.buf.items()})
        tmp.rename(path)
        for m in self.meta:
            self.meta_file.write(json.dumps(m) + "\n")
        self.meta_file.flush()
        self.buf = {k: [] for k in ARRAYS}
        self.meta, self.shard = [], self.shard + 1

    def close(self) -> None:
        self.flush()
        self.meta_file.close()


def generate_samples(segmenter, records: list[ObjectRecord], load_image: Callable[[int], np.ndarray],
                     out_dir: str | Path, *, crop_size: int = 128, context: float = 0.5,
                     max_objects_per_image: int | None = 8, seed: int = 0, shard_size: int = 2000,
                     max_images: int | None = None, log_every: int = 50) -> dict:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    by_image = group_by_image(records)
    image_ids = sorted(by_image)[:max_images]
    writer = _ShardWriter(out_dir, shard_size)
    t0 = time.time()
    try:
        for n, image_id in enumerate(image_ids, 1):
            objs = list(by_image[image_id])
            if max_objects_per_image and len(objs) > max_objects_per_image:
                objs = [objs[i] for i in sorted(rng.choice(len(objs), max_objects_per_image, replace=False))]
            image_t = torch.from_numpy(np.array(load_image(image_id))).permute(2, 0, 1).contiguous()
            segmenter.set_image(image_id)
            outs = segmenter.predict([Prompt(box=list(r.gt_box), origin="gt") for r in objs])
            for rec, out in zip(objs, outs):
                gt = rle_decode(rec.gt_mask)
                logits = segmenter.to_full_res(out.mask_logits)
                window = crop_window(rec.gt_box, context)
                arrays = crop_object(image_t, logits, gt, window, crop_size)
                arrays["box"] = box_mask(rec.gt_box, window, crop_size)
                arrays["clicks"] = clicks_to_crop([], window, crop_size, MAX_CLICKS)
                writer.add(arrays, dict(
                    obj_id=rec.obj_id, image_id=rec.image_id, category=rec.category_name, size_bin=rec.size_bin,
                    origin="gt", box=list(rec.gt_box), window=window, points=[], n_clicks=0,
                    iou=iou((logits > 0).numpy(), gt), pred_iou=out.iou_score,
                    crop_fp=float(((arrays["logits"] > 0) & ~arrays["gt"]).mean()),
                    crop_fn=float((arrays["gt"] & ~(arrays["logits"] > 0)).mean())))
            if log_every and n % log_every == 0:
                print(f"[samples] {n}/{len(image_ids)} images, {writer.n} samples, {time.time() - t0:.0f}s", flush=True)
    finally:
        writer.close()
    return {"n_samples": writer.n, "n_shards": writer.shard, "n_images": len(image_ids),
            "seconds": round(time.time() - t0, 1), "crop_size": crop_size, "context": context,
            "prompts": "clean GT box only", "max_objects_per_image": max_objects_per_image, "seed": seed}
