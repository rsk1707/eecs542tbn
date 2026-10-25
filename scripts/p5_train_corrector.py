"""Train corrector v0 on samples from p5_make_samples.py (GPU, ~20-30 min). See promptcorrect/corrector/train.py.

    sbatch --account=eecs542f26s001_class --time=01:30:00 --mem=64G --job-name=p5_train \
        slurm/run_gpu.sbatch scripts/p5_train_corrector.py --paths configs/paths/greatlakes.yaml \
        --run-id demo_p5_train_0 --samples demo_p5_samples_0

Validation = 10% of the sample images (image-level holdout). Writes $RUNS_ROOT/<run-id>/{best.pt, last.pt,
history.jsonl, final_val.json, training_curves.png, manifest.json}.
"""

import argparse
import json
import subprocess
from dataclasses import asdict

import torch

from promptcorrect.config import set_seed
from promptcorrect.corrector.dataset import load_samples, split_by_image
from promptcorrect.corrector.figures import training_curves
from promptcorrect.corrector.model import load_checkpoint
from promptcorrect.corrector.train import TrainConfig, evaluate, train
from promptcorrect.data.prepare import rooted_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--samples", required=True, help="run id of the samples (relative to runs_root)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, help="quick test")
    ap.add_argument("--data-on-cpu", action="store_true", help="keep samples in RAM instead of GPU memory")
    for f, v in asdict(TrainConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=(lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool)
                        else type(v), default=v)
    args = ap.parse_args()
    cfg = TrainConfig(**{f: getattr(args, f) for f in asdict(TrainConfig())})

    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    if (out_dir / "best.pt").exists():
        raise FileExistsError(f"{out_dir}/best.pt exists; choose a new --run-id")
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_dir = rooted_path(args.paths, "runs_root", args.samples)
    ds = load_samples(sample_dir, args.max_samples)
    if not args.data_on_cpu and device == "cuda":
        ds = ds.to(device)
    tr_idx, va_idx = split_by_image(ds, args.val_frac)
    train_ds, val_ds = ds.subset(tr_idx), ds.subset(va_idx)
    del ds

    result = train(train_ds, val_ds, cfg, out_dir, device)
    model, ckpt = load_checkpoint(out_dir / "best.pt", device)
    final = {"best_epoch": ckpt["epoch"], **evaluate(model, val_ds, cfg, device, n_click=None)}
    (out_dir / "final_val.json").write_text(json.dumps(final, indent=2) + "\n")
    training_curves(result["history"], out_dir / "training_curves.png")
    manifest = {"samples": args.samples, "n_train": len(train_ds), "n_val": len(val_ds),
                "n_val_images": len({m["image_id"] for m in val_ds.meta}), "train_config": asdict(cfg),
                "n_params": result["n_params"], "seconds": result["seconds"], "best_val_loss": result["best_val_loss"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
