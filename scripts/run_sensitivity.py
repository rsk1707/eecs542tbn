"""Sensitivity sweep (and optionally baselines vs ideal clicks) on perturbed GT boxes. See
promptcorrect/perturb/sweep.py.

    # IoU vs shift / scale / truncation, no correction (~18 boxes per object, batched per image):
    sbatch --account=eecs542f26s001_class --time=01:00:00 slurm/run_gpu.sbatch \
        scripts/run_sensitivity.py --paths configs/paths/greatlakes.yaml --run-id demo_sens_1

    # baselines and ideal clicks on a coarser grid (one decoder call per click):
    sbatch --account=eecs542f26s001_class --time=02:00:00 slurm/run_gpu.sbatch \
        scripts/run_sensitivity.py --paths configs/paths/greatlakes.yaml --run-id demo_baselines_1 \
        --methods all --steps 3 --levels shift=0.2,0.4 scale=0.67,1.5 truncate=0.3

Local check without a GPU: `--backend dummy --max-images 5` (needs the records file; dummy IoUs mean nothing).
Writes $RUNS_ROOT/<run-id>/{results.parquet, config.yaml, sweep_manifest.json, summary_*.csv}.
"""

import argparse
import json
import time

import pandas as pd

from promptcorrect.config import RunConfig, resolve_git_commit, save_run_config, set_seed
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import read_records
from promptcorrect.perturb.box import SWEEP_LEVELS, TYPES
from promptcorrect.perturb.sweep import METHODS, format_report, run_sweep, summarize


def parse_levels(items: list[str] | None) -> dict[str, tuple[float, ...]]:
    if not items:
        return dict(SWEEP_LEVELS)
    levels = {}
    for item in items:
        name, _, vals = item.partition("=")
        if name not in TYPES or not vals:
            raise SystemExit(f"--levels expects type=v1,v2 with type in {TYPES}, got {item!r}")
        levels[name] = tuple(float(v) for v in vals.split(","))
    return levels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--records", default="objects_val.jsonl", help="relative to data_root")
    ap.add_argument("--images", default="coco/val2017", help="relative to data_root")
    ap.add_argument("--backend", default="sam3", choices=["sam3", "dummy"])
    ap.add_argument("--levels", nargs="+", help="override the grid, e.g. shift=0.1,0.3 scale=0.5,2")
    ap.add_argument("--methods", nargs="+", default=["b0_none"], help=f"subset of {METHODS}, or 'all'")
    ap.add_argument("--steps", type=int, default=3, help="correction steps per method")
    ap.add_argument("--max-images", type=int, help="first N images only (quick test)")
    ap.add_argument("--boundary-all", action="store_true", help="Boundary IoU at every step (slow)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    methods = METHODS if args.methods == ["all"] else tuple(args.methods)
    levels = parse_levels(args.levels)
    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    if (out_dir / "results.parquet").exists():
        raise FileExistsError(f"{out_dir}/results.parquet exists; choose a new --run-id")
    set_seed(args.seed)

    records_file = rooted_path(args.paths, "data_root", args.records)
    records = list(read_records(records_file))
    if args.backend == "sam3":
        from promptcorrect.segmenters.sam3 import Sam3Segmenter

        segmenter = Sam3Segmenter(image_dirs=[rooted_path(args.paths, "data_root", args.images)], track_presence=False)
    else:
        from promptcorrect.segmenters.dummy import DummySegmenter

        segmenter = DummySegmenter(records)

    t0 = time.time()
    df = run_sweep(segmenter, records, run_id=args.run_id, levels=levels, methods=methods, steps=args.steps,
                   seed=args.seed, boundary_all=args.boundary_all, max_images=args.max_images)
    seconds = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "results.parquet", index=False)
    commit = resolve_git_commit()
    save_run_config(RunConfig(run_id=args.run_id, git_commit=commit, seed=args.seed, backend=args.backend,
                              perturbations=list(levels), max_steps=args.steps), out_dir)
    manifest = {
        "levels": {k: list(v) for k, v in levels.items()}, "methods": list(methods), "steps": args.steps,
        "backend": args.backend, "max_images": args.max_images, "seconds": round(seconds, 1), "n_rows": len(df),
        "n_objects": int(df.obj_id.nunique()), "n_images": int(df.image_id.nunique()), "git_commit": commit,
        "inputs": {"records_sha256": file_sha256(records_file)},
        "notes": "boundary_iou is step 0 only unless --boundary-all; SAM 3 hole filling is off (see CLAUDE.md)",
    }
    (out_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tables = summarize(df)
    for name, t in tables.items():
        t.to_csv(out_dir / f"summary_{name}.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(format_report(tables))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
