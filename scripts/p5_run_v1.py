"""Corrector v1 in ONE Slurm job: train (on P4's samples) -> calibrate (held-out training images) -> eval (demo val).

    sbatch --account=eecs542f26s001_class --time=06:00:00 --mem=96G --job-name=p5v1 \
        slurm/run_gpu.sbatch scripts/p5_run_v1.py --paths configs/paths/greatlakes.yaml --suffix 0 \
        --eval-args "--v0-train-run demo_p5_train_0"

Run ids: p5v1_{train,calib,eval}_<suffix>; samples default to P4's p4_samples_synth_0 (not regenerated).
A step whose output already exists is skipped, so a job that died can be resubmitted with the same command.
Extra per-step args: --train-args / --calib-args / --eval-args "<args>". Smoke test (a few minutes):

    sbatch --account=eecs542f26s001_class --time=00:45:00 --mem=64G --job-name=p5v1_smoke \
        slurm/run_gpu.sbatch scripts/p5_run_v1.py --paths configs/paths/greatlakes.yaml --suffix smoke \
        --train-args "--max-samples 3000 --epochs 2" --calib-args "--max-images 20" \
        --eval-args "--max-images 10 --no-index"
"""

import argparse
import shlex
import subprocess
import sys
import time

from promptcorrect.data.prepare import rooted_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--suffix", default="0")
    ap.add_argument("--prefix", default="p5v1")
    ap.add_argument("--samples", default="p4_samples_synth_0", help="samples run id (P4's synthetic variant)")
    ap.add_argument("--train-args", default="")
    ap.add_argument("--calib-args", default="")
    ap.add_argument("--eval-args", default="")
    args = ap.parse_args()

    if not rooted_path(args.paths, "runs_root", args.samples).joinpath("meta.jsonl").exists():
        sys.exit(f"[p5_v1] samples run {args.samples} has no meta.jsonl under runs_root")
    ids = {k: f"{args.prefix}_{k}_{args.suffix}" for k in ("train", "calib", "eval")}
    steps = [
        ("train", "manifest.json", ["scripts/p5_train_corrector.py", "--run-id", ids["train"], "--samples", args.samples,
                              "--quality-head", "true"], args.train_args),
        ("calib", "calibration.json", ["scripts/p5_calibrate_v1.py", "--run-id", ids["calib"],
                                       "--train-run", ids["train"]], args.calib_args),
        ("eval", "results.parquet", ["scripts/p5_eval_v1.py", "--run-id", ids["eval"], "--train-run", ids["train"],
                                     "--calib-run", ids["calib"]], args.eval_args),
    ]
    for name, done_file, cmd, extra in steps:
        if rooted_path(args.paths, "runs_root", ids[name]).joinpath(done_file).exists():
            print(f"[p5_v1] {name}: {ids[name]} already has {done_file}, skipping", flush=True)
            continue
        full = [sys.executable, *cmd, "--paths", args.paths, *shlex.split(extra)]
        print(f"[p5_v1] {name}: {shlex.join(full)}", flush=True)
        t0 = time.time()
        subprocess.run(full, check=True)
        print(f"[p5_v1] {name} done in {time.time() - t0:.0f}s", flush=True)
    print(f"[p5_v1] all done: {ids}", flush=True)


if __name__ == "__main__":
    main()
