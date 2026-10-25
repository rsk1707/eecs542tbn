"""Corrector v0 vs no correction on the demo val objects (clean GT boxes), end to end with SAM 3 (GPU).
See promptcorrect/corrector/rollout.py.

    sbatch --account=eecs542f26s001_class --time=03:00:00 --job-name=p5_eval \
        slurm/run_gpu.sbatch scripts/p5_eval_corrector.py --paths configs/paths/greatlakes.yaml \
        --run-id demo_p5_eval_0 --train-run demo_p5_train_0

Writes $RUNS_ROOT/<run-id>/{results.parquet, summary_*.csv, corrector_vs_none.png, manifest.json,
config.yaml} and prints the summary tables.
"""

import argparse
import json
import subprocess
import time

import torch

from promptcorrect.config import RunConfig, save_run_config, set_seed
from promptcorrect.corrector.figures import corrector_vs_none
from promptcorrect.corrector.model import load_checkpoint
from promptcorrect.corrector.rollout import format_report, run_rollouts, summarize
from promptcorrect.corrector.samples import load_coco_image
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import build_coco_records, read_records, write_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-run", required=True, help="run id holding the checkpoint")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--method", default="corrector_v0")
    ap.add_argument("--records", default="objects_val.jsonl", help="relative to data_root (the demo val objects)")
    ap.add_argument("--annotations", default="coco/annotations/instances_val2017.json",
                    help="relative to data_root; only used if --records is unreadable")
    ap.add_argument("--split", default="splits_demo.json", help="relative to data_root; as --annotations")
    ap.add_argument("--images", default="coco/val2017", help="relative to data_root")
    ap.add_argument("--steps", type=int, default=1, help="corrector clicks per object (v0 is trained for 1)")
    ap.add_argument("--threshold", type=float, default=0.5, help="error-map threshold for clicking")
    ap.add_argument("--max-images", type=int, help="first N images only (quick test)")
    ap.add_argument("--boundary-all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    if (out_dir / "results.parquet").exists():
        raise FileExistsError(f"{out_dir}/results.parquet exists; choose a new --run-id")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = rooted_path(args.paths, "runs_root", args.train_run)
    model, ckpt = load_checkpoint(train_dir / args.ckpt, device)
    samples_run = json.loads((train_dir / "manifest.json").read_text())["samples"]
    samples_manifest = json.loads(rooted_path(args.paths, "runs_root", samples_run).joinpath("manifest.json").read_text())

    records_file = rooted_path(args.paths, "data_root", args.records)
    try:
        records = list(read_records(records_file))
        records_source = str(records_file)
    except PermissionError:
        # The shared file can be owner-only (mode 600). Rebuild it with P2's builder from the same inputs;
        # the sha256 in the manifest shows whether it is identical (demo: 86f4cb3d...).
        print(f"[eval] cannot read {records_file}; rebuilding from {args.annotations} + {args.split}", flush=True)
        ann = json.loads(rooted_path(args.paths, "data_root", args.annotations).read_text(encoding="utf-8"))
        ids = json.loads(rooted_path(args.paths, "data_root", args.split).read_text(encoding="utf-8"))["val"]
        out_dir.mkdir(parents=True, exist_ok=True)
        records_file = out_dir / "objects_val.jsonl"
        if not records_file.exists():
            write_records(build_coco_records(ann, ids), records_file)
        records = list(read_records(records_file))
        records_source = f"rebuilt from {args.annotations} + {args.split}"
    images = rooted_path(args.paths, "data_root", args.images)

    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    segmenter = Sam3Segmenter(image_dirs=[images], track_presence=False)
    t0 = time.time()
    df = run_rollouts(segmenter, model, records, load_coco_image([images]), run_id=args.run_id, method=args.method, steps=args.steps, crop_size=samples_manifest["crop_size"], context=samples_manifest["context"],
                      threshold=args.threshold, device=device, max_images=args.max_images,
                      boundary_all=args.boundary_all)
    seconds = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "results.parquet", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_run_config(RunConfig(run_id=args.run_id, git_commit=commit, seed=args.seed, backend="sam3",
                              train_data="real", max_steps=args.steps, splits_file="splits_demo.json"), out_dir)
    tables = summarize(df, args.steps)
    for name, t in tables.items():
        t.to_csv(out_dir / f"summary_{name}.csv")
    corrector_vs_none(tables, out_dir / "corrector_vs_none.png", args.method)
    manifest = {
        "train_run": args.train_run, "ckpt": args.ckpt, "ckpt_epoch": ckpt.get("epoch"), "samples_run": samples_run,
        "method": args.method, "compared_with": ["b0_none"], "prompts": "clean GT box", "steps": args.steps,
        "threshold": args.threshold, "max_images": args.max_images, "seconds": round(seconds, 1),
        "n_rows": len(df), "n_objects": int(df.obj_id.nunique()), "n_images": int(df.image_id.nunique()),
        "standins": "none", "git_commit": commit,
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "inputs": {"records_sha256": file_sha256(records_file), "records_source": records_source},
        "notes": "boundary_iou is step 0 only unless --boundary-all; SAM 3 hole filling off; clicks are sent with the prompt box",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(format_report(tables))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
