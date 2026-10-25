"""Corrector v0 training samples from SAM 3 on COCO train images (GPU). See promptcorrect/corrector/samples.py.

    sbatch --account=eecs542f26s001_class --time=03:00:00 --mem=64G --job-name=p5_samples \
        slurm/run_gpu.sbatch scripts/p5_make_samples.py --paths configs/paths/greatlakes.yaml \
        --run-id demo_p5_samples_0 --n-images 1500

Picks `--n-images` COCO train2017 images after removing LVIS-val images (leakage, CLAUDE.md 4.5), builds
their object records with P2's builder, and writes $RUNS_ROOT/<run-id>/{shard_*.npz, meta.jsonl,
records.jsonl, manifest.json, config.yaml}. Format: docs/p5_corrector_v0.md.
"""

import argparse
import json
import subprocess

from promptcorrect.config import RunConfig, save_run_config, set_seed
from promptcorrect.corrector.samples import generate_samples, load_coco_image, train_subset_records
from promptcorrect.corrector.standins import STANDINS_NOTE
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import write_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-images", type=int, default=1500)
    ap.add_argument("--train-annotations", default="coco/annotations/instances_train2017.json", help="relative to data_root")
    ap.add_argument("--val-annotations", default="coco/annotations/instances_val2017.json", help="relative to data_root")
    ap.add_argument("--lvis-val", default="lvis/lvis_v1_val.json", help="relative to data_root")
    ap.add_argument("--images", default="coco/train2017", help="relative to data_root")
    ap.add_argument("--n-perturbed", type=int, default=2, help="perturbed boxes per object (plus the clean GT box)")
    ap.add_argument("--level-min", type=float, default=0.05)
    ap.add_argument("--level-max", type=float, default=0.5)
    ap.add_argument("--click-state-prob", type=float, default=0.5)
    ap.add_argument("--max-clicks", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--context", type=float, default=0.5)
    ap.add_argument("--max-objects-per-image", type=int, default=8)
    ap.add_argument("--max-images", type=int, help="process only the first N chosen images (quick test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    if (out_dir / "meta.jsonl").exists():
        raise FileExistsError(f"{out_dir}/meta.jsonl exists; choose a new --run-id")
    set_seed(args.seed)
    files = {k: rooted_path(args.paths, "data_root", getattr(args, k))
             for k in ("train_annotations", "val_annotations", "lvis_val")}
    print("[samples] loading annotations ...", flush=True)
    loaded = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in files.items()}
    records, split_info = train_subset_records(loaded["train_annotations"], loaded["val_annotations"],
                                               loaded["lvis_val"], args.n_images, seed=args.seed)
    del loaded
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records(records, out_dir / "records.jsonl")
    print(f"[samples] {len(records)} objects in {len(split_info['image_ids'])} images", flush=True)

    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    images = rooted_path(args.paths, "data_root", args.images)
    segmenter = Sam3Segmenter(image_dirs=[images], track_presence=False)
    stats = generate_samples(
        segmenter, records, load_coco_image([images]), out_dir, n_perturbed=args.n_perturbed,
        level_range=(args.level_min, args.level_max), click_state_prob=args.click_state_prob,
        max_clicks=args.max_clicks, crop_size=args.crop_size, context=args.context,
        max_objects_per_image=args.max_objects_per_image, seed=args.seed, max_images=args.max_images)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_run_config(RunConfig(run_id=args.run_id, git_commit=commit, seed=args.seed, backend="sam3",
                              train_data="synthetic", max_steps=args.max_clicks, splits_file="none (see manifest)"), out_dir)
    manifest = {**stats, "n_objects": len(records), "split": {k: v for k, v in split_info.items() if k != "image_ids"},
                "image_ids": split_info["image_ids"], "standins": STANDINS_NOTE,
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                "inputs": {k: file_sha256(p) for k, p in files.items()},
                "notes": "SAM 3 hole filling off; LVIS-val images removed before sampling"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "image_ids"}, indent=2))


if __name__ == "__main__":
    main()
