"""Corrector training samples from perturbed boxes (GPU). Owner: P4. See promptcorrect/perturb/training.py.

    # quick check (20 images), then the full set
    sbatch --account=eecs542f26s001_class --time=00:30:00 --job-name=p4_samples slurm/run_gpu.sbatch \
        scripts/p4_make_samples.py --paths configs/paths/greatlakes.yaml --run-id p4_samples_synth_test --max-images 20
    sbatch --account=eecs542f26s001_class --time=04:00:00 --mem=64G --job-name=p4_samples slurm/run_gpu.sbatch \
        scripts/p4_make_samples.py --paths configs/paths/greatlakes.yaml --run-id p4_samples_synth_0

Same images, leakage handling and format as P5's scripts/p5_make_samples.py (docs/p5_corrector_v0.md): the
same `--n-images` COCO train2017 images (chosen after removing LVIS-val images, same seed), at most 8 objects
per image. Each object gives its clean GT box (unless --no-clean) plus --n-perturbed random shift / scale /
truncate boxes; meta.jsonl adds prompt_id, pert_type, pert_level, pert_dir, clipped and the describe()
severity columns. Writes $RUNS_ROOT/<run-id>/{shard_*.npz, meta.jsonl, records.jsonl, manifest.json, config.yaml}.
"""

import argparse
import json

from promptcorrect.config import RunConfig, resolve_git_commit, save_run_config, set_seed
from promptcorrect.corrector.samples import generate_samples, load_coco_image, train_subset_records
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import write_records
from promptcorrect.perturb.box import TRAIN_RANGES, TYPES
from promptcorrect.perturb.training import TRAIN_DATA, SeverityMeta, make_prompt_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-data", default="synthetic", choices=TRAIN_DATA)
    ap.add_argument("--n-perturbed", type=int, default=2, help="perturbed boxes per object")
    ap.add_argument("--no-clean", action="store_true", help="leave out the clean GT-box sample of each object")
    ap.add_argument("--types", nargs="+", default=list(TYPES), choices=TYPES)
    ap.add_argument("--n-images", type=int, default=3000)
    ap.add_argument("--train-annotations", default="coco/annotations/instances_train2017.json", help="relative to data_root")
    ap.add_argument("--val-annotations", default="coco/annotations/instances_val2017.json", help="relative to data_root")
    ap.add_argument("--lvis-val", default="lvis/lvis_v1_val.json", help="relative to data_root")
    ap.add_argument("--images", default="coco/train2017", help="relative to data_root")
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--context", type=float, default=0.5)
    ap.add_argument("--max-objects-per-image", type=int, default=8)
    ap.add_argument("--max-images", type=int, help="process only the first N chosen images (quick test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompt_fn = make_prompt_fn(args.train_data, seed=args.seed, n_perturbed=args.n_perturbed,
                               include_clean=not args.no_clean, types=tuple(args.types))
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
        segmenter, records, load_coco_image([images]), out_dir, crop_size=args.crop_size, context=args.context,
        max_objects_per_image=args.max_objects_per_image, seed=args.seed, max_images=args.max_images,
        prompt_fn=prompt_fn, extra_meta=SeverityMeta())

    commit = resolve_git_commit()
    save_run_config(RunConfig(run_id=args.run_id, git_commit=commit, seed=args.seed, backend="sam3",
                              train_data=args.train_data, perturbations=list(args.types), max_steps=0,
                              splits_file="none (see manifest)"), out_dir)
    manifest = {**stats, "n_objects": len(records), "split": {k: v for k, v in split_info.items() if k != "image_ids"},
                "image_ids": split_info["image_ids"], "git_commit": commit,
                "train_data": args.train_data, "prompts_per_object": {
                    "clean": not args.no_clean, "n_perturbed": args.n_perturbed, "types": args.types,
                    "level_ranges": {t: TRAIN_RANGES[t] for t in args.types},
                    "rng": "object_rng(seed, obj_id, 'train'): independent of image order"},
                "inputs": {k: file_sha256(p) for k, p in files.items()},
                "notes": "SAM 3 hole filling off; LVIS-val images removed before sampling; crop window and box "
                         "channel built from the prompt box, not the GT box"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "image_ids"}, indent=2))


if __name__ == "__main__":
    main()
