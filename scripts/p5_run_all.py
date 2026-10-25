"""Corrector v0 end to end in ONE Slurm job (one queue wait instead of three): samples -> train -> eval.

    sbatch --account=eecs542f26s001_class --time=06:00:00 --mem=64G --job-name=p5_all \
        slurm/run_gpu.sbatch scripts/p5_run_all.py --paths configs/paths/greatlakes.yaml --suffix 0

Run ids are demo_p5_{samples,train,eval}_<suffix>. A step whose output already exists is skipped, so a job
that died (e.g. time limit) can be resubmitted with the same command. Extra per-step arguments:
--samples-args / --train-args / --eval-args "<args>" (e.g. --samples-args "--max-images 20").
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
    ap.add_argument("--prefix", default="demo_p5")
    ap.add_argument("--samples-args", default="")
    ap.add_argument("--train-args", default="")
    ap.add_argument("--eval-args", default="")
    args = ap.parse_args()

    ids = {k: f"{args.prefix}_{k}_{args.suffix}" for k in ("samples", "train", "eval")}
    steps = [
        ("samples", "meta.jsonl", ["scripts/p5_make_samples.py", "--run-id", ids["samples"]], args.samples_args),
        ("train", "best.pt", ["scripts/p5_train_corrector.py", "--run-id", ids["train"], "--samples", ids["samples"]],
         args.train_args),
        ("eval", "results.parquet", ["scripts/p5_eval_corrector.py", "--run-id", ids["eval"], "--train-run", ids["train"]],
         args.eval_args),
    ]
    for name, done_file, cmd, extra in steps:
        if rooted_path(args.paths, "runs_root", ids[name]).joinpath(done_file).exists():
            print(f"[p5_all] {name}: {ids[name]} already has {done_file}, skipping", flush=True)
            continue
        full = [sys.executable, *cmd, "--paths", args.paths, *shlex.split(extra)]
        print(f"[p5_all] {name}: {shlex.join(full)}", flush=True)
        t0 = time.time()
        subprocess.run(full, check=True)
        print(f"[p5_all] {name} done in {time.time() - t0:.0f}s", flush=True)
    print(f"[p5_all] all done: {ids}", flush=True)


if __name__ == "__main__":
    main()
