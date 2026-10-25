"""Corrector v0 training samples: one frozen-segmenter state per sample, stored as object-centric crops.
Owner: P5. Format: docs/p5_corrector_v0.md.

For every object: the clean GT box plus `n_perturbed` perturbed boxes (stand-in perturbations, see
standins.py) go through the segmenter. Each result is one sample ("step 0"). With probability
`click_state_prob` a second sample is added from the same prompt after 1..`max_clicks` ideal clicks, so the
click channels are not always empty (the eval rollout feeds the corrector its own earlier clicks).

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
from promptcorrect.corrector.standins import PERTURBATIONS, box_iou, perturb_box
from promptcorrect.data.geometry import rle_decode
from promptcorrect.data.records import build_coco_records, group_by_image
from promptcorrect.data.splits import training_split
from promptcorrect.eval.metrics import iou
from promptcorrect.experiments.rq0 import oracle_click
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
                     out_dir: str | Path, *, n_perturbed: int = 2, level_range: tuple[float, float] = (0.05, 0.5),
                     click_state_prob: float = 0.5, max_clicks: int = 2, crop_size: int = 128,
                     context: float = 0.5, max_objects_per_image: int | None = 8, seed: int = 0,
                     shard_size: int = 2000, max_images: int | None = None, log_every: int = 50) -> dict:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    by_image = group_by_image(records)
    image_ids = sorted(by_image)[:max_images]
    writer = _ShardWriter(out_dir, shard_size)
    t0, n_states = time.time(), {"step0": 0, "clicks": 0}

    def state(image_t, rec, gt, box, points, logits_low, pred_iou, pert, origin) -> None:
        logits = segmenter.to_full_res(logits_low)
        window = crop_window(box, context)
        arrays = crop_object(image_t, logits, gt, window, crop_size)
        arrays["box"] = box_mask(box, window, crop_size)
        arrays["clicks"] = clicks_to_crop(points, window, crop_size, MAX_CLICKS)
        writer.add(arrays, dict(
            obj_id=rec.obj_id, image_id=rec.image_id, category=rec.category_name, size_bin=rec.size_bin,
            origin=origin, pert_type=pert["type"] if pert else None,
            pert_level=pert["params"]["level"] if pert else 0.0, perturbation=pert, box=box, window=window,
            points=[list(p) for p in points], n_clicks=len(points), box_iou_in=box_iou(box, rec.gt_box),
            iou=iou((logits > 0).numpy(), gt), pred_iou=pred_iou,
            crop_fp=float(((arrays["logits"] > 0) & ~arrays["gt"]).mean()),
            crop_fn=float((arrays["gt"] & ~(arrays["logits"] > 0)).mean())))

    try:
        for n, image_id in enumerate(image_ids, 1):
            objs = list(by_image[image_id])
            if max_objects_per_image and len(objs) > max_objects_per_image:
                objs = [objs[i] for i in sorted(rng.choice(len(objs), max_objects_per_image, replace=False))]
            image = load_image(image_id)
            hw = image.shape[:2]
            image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            segmenter.set_image(image_id)

            prompts = []  # (rec, box, perturbation)
            for rec in objs:
                prompts.append((rec, list(rec.gt_box), None))
                for _ in range(n_perturbed):
                    kind = PERTURBATIONS[int(rng.integers(len(PERTURBATIONS)))]
                    level = float(rng.uniform(*level_range))
                    box, pert = perturb_box(rec.gt_box, kind, level, rng, hw)
                    prompts.append((rec, box, pert))
            outs = segmenter.predict([Prompt(box=b, origin="perturbed" if p else "gt", perturbation=p)
                                      for _, b, p in prompts])
            gts = {r.obj_id: rle_decode(r.gt_mask) for r in objs}
            for (rec, box, pert), out in zip(prompts, outs):
                gt = gts[rec.obj_id]
                origin = "perturbed" if pert else "gt"
                state(image_t, rec, gt, box, [], out.mask_logits, out.iou_score, pert, origin)
                n_states["step0"] += 1
                if rng.random() >= click_state_prob:
                    continue
                points, mask = [], segmenter.full_res_mask(out)
                for _ in range(int(rng.integers(1, max_clicks + 1))):
                    click = oracle_click(mask, gt)
                    if click is None:
                        break
                    points.append(click)
                    o = segmenter.predict([Prompt(box=box, points=list(points), origin="oracle")])[0]
                    mask = segmenter.full_res_mask(o)
                if points:
                    state(image_t, rec, gt, box, points, o.mask_logits, o.iou_score, pert, origin)
                    n_states["clicks"] += 1
            if log_every and n % log_every == 0:
                print(f"[samples] {n}/{len(image_ids)} images, {writer.n} samples, {time.time() - t0:.0f}s", flush=True)
    finally:
        writer.close()
    return {"n_samples": writer.n, "n_shards": writer.shard, "n_images": len(image_ids), "states": n_states,
            "seconds": round(time.time() - t0, 1), "crop_size": crop_size, "context": context,
            "n_perturbed": n_perturbed, "level_range": list(level_range), "click_state_prob": click_state_prob,
            "max_clicks": max_clicks, "max_objects_per_image": max_objects_per_image, "seed": seed}
