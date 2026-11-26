"""Corrector v1 vs no correction, P4's rule baselines and P3's ideal clicks on the demo val objects (GPU).
See promptcorrect/corrector/evaluate_v1.py. Normally run through scripts/p5_run_v1.py.

    sbatch --account=eecs542f26s001_class --time=03:00:00 --mem=32G --job-name=p5v1_eval \
        slurm/run_gpu.sbatch scripts/p5_eval_v1.py --paths configs/paths/greatlakes.yaml \
        --run-id p5v1_eval_0 --train-run p5v1_train_0 --calib-run p5v1_calib_0 [--v0-train-run demo_p5_train_0]

Thresholds come from the calibration run (held-out training images) for the chosen --feedback setting;
--err-threshold / --quality-threshold override them (say so in the report if you do). Prompts: each object's
clean GT box plus --n-perturbed P4 perturbed boxes. Writes $RUNS_ROOT/<run-id>/{results.parquet (ResultRows),
summary_final.csv, go_no_go.csv, iou_by_step.csv, manifest.json, config.yaml} and a row in
results/runs_index.csv (commit that file).
"""

import argparse
import json
import time

import pandas as pd
import torch

from promptcorrect.config import RunConfig, resolve_git_commit, save_run_config, set_seed
from promptcorrect.corrector.evaluate_v1 import (BASELINES, go_no_go, iou_by_step, load_eval_records, run_eval_v1,
                                                 summary_table)
from promptcorrect.corrector.model import load_checkpoint
from promptcorrect.corrector.policy import CorrectorPolicy
from promptcorrect.corrector.samples import load_coco_image
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.eval.results import read_results, update_runs_index, write_results


def _bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--calib-run", help="calibration run id; without it --err-threshold/--quality-threshold are used")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--method", default="corrector_v1")
    ap.add_argument("--feedback", type=_bool, default=True, help="feed previous logits back (v1 and the oracle)")
    ap.add_argument("--err-threshold", type=float)
    ap.add_argument("--quality-threshold", type=float)
    ap.add_argument("--v0-train-run", help="also evaluate this v0 checkpoint as corrector_v0 (threshold 0.5, "
                                           "no gate, no feedback: the v0 setting)")
    ap.add_argument("--steps", type=int, default=1, help="correction steps for every method (v1 trained for 1)")
    ap.add_argument("--n-perturbed", type=int, default=2)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--records", default="objects_val.jsonl", help="relative to data_root (demo val objects)")
    ap.add_argument("--annotations", default="coco/annotations/instances_val2017.json")
    ap.add_argument("--split", default="splits_demo.json")
    ap.add_argument("--images", default="coco/val2017", help="relative to data_root")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--boundary-all", action="store_true")
    ap.add_argument("--no-index", action="store_true", help="do not add a row to results/runs_index.csv")
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
    sm = json.loads(rooted_path(args.paths, "runs_root", samples_run).joinpath("manifest.json").read_text())

    err_thr, q_thr, calib = 0.5, None, None
    if args.calib_run:
        calib = json.loads(rooted_path(args.paths, "runs_root", args.calib_run).joinpath("calibration.json").read_text())
        chosen = calib["feedback" if args.feedback else "nofeedback"]
        err_thr, q_thr = chosen["err_threshold"], chosen["tau"]
    err_thr = args.err_threshold if args.err_threshold is not None else err_thr
    q_thr = args.quality_threshold if args.quality_threshold is not None else q_thr
    policies = {args.method: CorrectorPolicy(model, device, sm["crop_size"], sm["context"], err_thr, q_thr,
                                             args.feedback)}
    v0_info = None
    if args.v0_train_run:
        v0_dir = rooted_path(args.paths, "runs_root", args.v0_train_run)
        m0, c0 = load_checkpoint(v0_dir / "best.pt", device)
        s0 = json.loads(rooted_path(args.paths, "runs_root", json.loads((v0_dir / "manifest.json").read_text())["samples"])
                        .joinpath("manifest.json").read_text())
        policies["corrector_v0"] = CorrectorPolicy(m0, device, s0["crop_size"], s0["context"], 0.5, None, False)
        v0_info = {"train_run": args.v0_train_run, "epoch": c0.get("epoch")}
    print(f"[eval_v1] {args.method}: err_threshold={err_thr} quality_threshold={q_thr} feedback={args.feedback}",
          flush=True)

    records, records_file, records_source = load_eval_records(args.paths, args.records, args.annotations, args.split,
                                                              out_dir)
    images = rooted_path(args.paths, "data_root", args.images)

    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    segmenter = Sam3Segmenter(image_dirs=[images], track_presence=False)
    t0 = time.time()
    rows = run_eval_v1(segmenter, records, load_coco_image([images]), run_id=args.run_id, policies=policies,
                       baselines=() if args.no_baselines else BASELINES, oracle=not args.no_oracle,
                       oracle_feedback=args.feedback, steps=args.steps, n_perturbed=args.n_perturbed,
                       include_clean=not args.no_clean, seed=args.seed, max_images=args.max_images,
                       boundary_all=args.boundary_all)
    seconds = time.time() - t0
    write_results(rows, out_dir)
    df = read_results(out_dir)

    commit = resolve_git_commit()
    save_run_config(RunConfig(run_id=args.run_id, git_commit=commit, seed=args.seed, backend="sam3",
                              train_data="synthetic", perturbations=["shift", "scale", "truncate"],
                              max_steps=args.steps, splits_file=args.split), out_dir)
    table = summary_table(df)
    table.to_csv(out_dir / "summary_final.csv")
    gng = go_no_go(table, args.method)
    gng.to_csv(out_dir / "go_no_go.csv", index=False)
    iou_by_step(df, args.steps).to_csv(out_dir / "iou_by_step.csv")
    manifest = {
        "train_run": args.train_run, "ckpt": args.ckpt, "ckpt_epoch": ckpt.get("epoch"), "samples_run": samples_run,
        "calib_run": args.calib_run, "method": args.method, "err_threshold": err_thr, "quality_threshold": q_thr,
        "thresholds_from": "calibration (held-out training images)" if args.calib_run and args.err_threshold is None
        and args.quality_threshold is None else "command line",
        "feedback": args.feedback, "oracle_feedback": args.feedback, "baselines_feedback": False,
        "v0": v0_info, "steps": args.steps, "n_perturbed": args.n_perturbed, "include_clean": not args.no_clean,
        "max_images": args.max_images, "seconds": round(seconds, 1), "n_rows": len(df),
        "n_objects": int(df.obj_id.nunique()), "n_images": int(df.image_id.nunique()), "git_commit": commit,
        "inputs": {"records_sha256": file_sha256(records_file), "records_source": records_source},
        "notes": "step 0 shared by all methods; P4 baselines never feed logits back (their definition); "
                 "boundary_iou only at step 0 unless --boundary-all; SAM 3 hole filling off",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.no_index and args.max_images is None:
        head = {}
        try:
            r = table.loc[("gt", "start<0.75", args.method)]
            head = {"delta_fail_gt": float(r.delta),
                    "harm_good_gt_pct": float(table.loc[("gt", "start>=0.75", args.method)].harm_pct)}
        except KeyError:
            pass
        update_runs_index(args.run_id, "P5", out_dir, notes=f"{args.method} eval, feedback={args.feedback}, "
                          f"thr=({err_thr}, {q_thr}), train={args.train_run}", headline=head, git_commit=commit)

    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_columns", 20):
        for src in ("gt", "perturbed"):
            for grp in ("start<0.75", "start>=0.75"):
                try:
                    print(f"\n== {src} / {grp}\n{table.loc[(src, grp)].to_string()}")
                except KeyError:
                    pass
        print(f"\n== go / no-go\n{gng.to_string(index=False)}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
