"""Choose corrector v1's click-or-abstain thresholds on held-out TRAINING images with SAM 3 (GPU, ~10-20 min).
See promptcorrect/corrector/calibrate.py. Normally run through scripts/p5_run_v1.py.

    sbatch --account=eecs542f26s001_class --time=01:00:00 --mem=32G --job-name=p5v1_calib \
        slurm/run_gpu.sbatch scripts/p5_calibrate_v1.py --paths configs/paths/greatlakes.yaml \
        --run-id p5v1_calib_0 --train-run p5v1_train_0

The held-out samples are the image-level holdout the train run used (same val_frac and salt), so the val
objects are never touched. Writes $RUNS_ROOT/<run-id>/{calib_rows.parquet, curve_feedback.csv,
curve_nofeedback.csv, calibration.json, manifest.json}.
"""

import argparse
import json
from types import SimpleNamespace

import torch

from promptcorrect.config import resolve_git_commit, set_seed
from promptcorrect.corrector.calibrate import choose_thresholds, run_calibration, threshold_curve
from promptcorrect.corrector.dataset import split_by_image
from promptcorrect.corrector.model import load_checkpoint
from promptcorrect.corrector.policy import CorrectorPolicy
from promptcorrect.corrector.samples import load_coco_image
from promptcorrect.data.prepare import rooted_path
from promptcorrect.data.records import read_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--images", default="coco/train2017", help="relative to data_root")
    ap.add_argument("--err-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--max-harm", type=float, default=0.05, help="max harm rate on held-out good masks (fraction)")
    ap.add_argument("--good-iou", type=float, default=0.75)
    ap.add_argument("--max-images", type=int, help="first N held-out images only (quick test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    if (out_dir / "calibration.json").exists():
        raise FileExistsError(f"{out_dir}/calibration.json exists; choose a new --run-id")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = rooted_path(args.paths, "runs_root", args.train_run)
    model, ckpt = load_checkpoint(train_dir / args.ckpt, device)
    if not model.has_quality:
        print("[calib] WARNING: checkpoint has no quality head; only the error threshold is calibrated", flush=True)
    train_manifest = json.loads((train_dir / "manifest.json").read_text())
    sample_dir = rooted_path(args.paths, "runs_root", train_manifest["samples"])
    samples_manifest = json.loads((sample_dir / "manifest.json").read_text())
    metas = [json.loads(l) for l in (sample_dir / "meta.jsonl").read_text().splitlines() if l.strip()]
    _, val_idx = split_by_image(SimpleNamespace(meta=metas), train_manifest.get("val_frac", 0.1))
    held = [metas[i] for i in val_idx]
    records = {r.obj_id: r for r in read_records(sample_dir / "records.jsonl")}
    print(f"[calib] {len(held)} held-out samples in {len({m['image_id'] for m in held})} training images", flush=True)

    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    images = rooted_path(args.paths, "data_root", args.images)
    segmenter = Sam3Segmenter(image_dirs=[images], track_presence=False)
    policy = CorrectorPolicy(model, device, samples_manifest["crop_size"], samples_manifest["context"])
    df = run_calibration(segmenter, policy, held, records, load_coco_image([images]),
                         err_thresholds=tuple(args.err_thresholds), max_images=args.max_images)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "calib_rows.parquet", index=False)

    result = {}
    for fb, name in ((True, "feedback"), (False, "nofeedback")):
        curve = threshold_curve(df, fb, good_iou=args.good_iou)
        curve.to_csv(out_dir / f"curve_{name}.csv", index=False)
        result[name] = choose_thresholds(curve, args.max_harm)
        always = curve[(curve.tau == float("inf"))].sort_values("delta_all", ascending=False).iloc[0]
        result[name]["no_gate_reference"] = {k: float(always[k]) for k in
                                             ("err_threshold", "delta_all", "delta_fail", "harm_good_pct", "clicked_pct")}
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {"train_run": args.train_run, "ckpt": args.ckpt, "ckpt_epoch": ckpt.get("epoch"),
                "samples_run": train_manifest["samples"], "n_samples": len(df) // max(1, 2 * len(args.err_thresholds)),
                "n_images": int(df.image_id.nunique()) if len(df) else 0, "max_images": args.max_images,
                "err_thresholds": args.err_thresholds, "max_harm": args.max_harm, "good_iou": args.good_iou,
                "git_commit": resolve_git_commit(),
                "notes": "held-out = split_by_image(val_frac, salt p5v0) of the training samples; prompts = each "
                         "sample's own box (clean or P4-perturbed); one corrector click per sample"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
