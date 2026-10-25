"""Corrector v0 vs baselines vs ideal clicks on the demo val objects, end to end with SAM 3 (GPU).
See promptcorrect/corrector/rollout.py.

    sbatch --account=eecs542f26s001_class --time=03:00:00 --job-name=p5_eval \
        slurm/run_gpu.sbatch scripts/p5_eval_corrector.py --paths configs/paths/greatlakes.yaml \
        --run-id demo_p5_eval_0 --train-run demo_p5_train_0

Writes $RUNS_ROOT/<run-id>/{results.parquet, summary_*.csv, iou_by_step.png, gain_and_harm.png,
manifest.json, config.yaml} and prints the summary tables.
"""

import argparse
import json
import subprocess
import time

import torch

from promptcorrect.config import RunConfig, save_run_config, set_seed
from promptcorrect.corrector.figures import gain_and_harm, iou_by_step
from promptcorrect.corrector.model import load_checkpoint
from promptcorrect.corrector.rollout import BASELINES, format_report, run_rollouts, summarize
from promptcorrect.corrector.samples import load_coco_image
from promptcorrect.corrector.standins import STANDINS_NOTE
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import read_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-run", required=True, help="run id holding the checkpoint")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--method", default="corrector_v0")
    ap.add_argument("--records", default="objects_val.jsonl", help="relative to data_root (the demo val objects)")
    ap.add_argument("--images", default="coco/val2017", help="relative to data_root")
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
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
    records = list(read_records(records_file))
    images = rooted_path(args.paths, "data_root", args.images)

    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    segmenter = Sam3Segmenter(image_dirs=[images], track_presence=False)
    t0 = time.time()
    df = run_rollouts(segmenter, model, records, load_coco_image([images]), run_id=args.run_id, method=args.method,
                      baselines=tuple(args.baselines), steps=args.steps, levels=tuple(args.levels),
                      crop_size=samples_manifest["crop_size"], context=samples_manifest["context"],
                      threshold=args.threshold, device=device, seed=args.seed, max_images=args.max_images,
                      boundary_all=args.boundary_all)
    seconds = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "results.parquet", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_run_config(RunConfig(run_id=args.run_id, git_commit=commit, seed=args.seed, backend="sam3",
                              train_data="synthetic", max_steps=args.steps, splits_file="splits_demo.json"), out_dir)
    tables = summarize(df, args.steps)
    for name, t in tables.items():
        t.to_csv(out_dir / f"summary_{name}.csv")
    iou_by_step(tables, out_dir / "iou_by_step.png")
    gain_and_harm(tables, out_dir / "gain_and_harm.png")
    manifest = {
        "train_run": args.train_run, "ckpt": args.ckpt, "ckpt_epoch": ckpt.get("epoch"), "samples_run": samples_run,
        "method": args.method, "baselines": args.baselines, "steps": args.steps, "levels": args.levels,
        "threshold": args.threshold, "max_images": args.max_images, "seconds": round(seconds, 1),
        "n_rows": len(df), "n_objects": int(df.obj_id.nunique()), "n_images": int(df.image_id.nunique()),
        "standins": STANDINS_NOTE, "git_commit": commit,
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "inputs": {"records_sha256": file_sha256(records_file)},
        "notes": "boundary_iou is step 0 only unless --boundary-all; SAM 3 hole filling off; clicks are sent with the prompt box",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(format_report(tables))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
