"""RQ0 study on the prepared demo objects (GPU node). See promptcorrect/eval/rq0.py. Owner: P3.

    sbatch --account=eecs542f26s001_class --time=03:00:00 --job-name=rq0 slurm/run_gpu.sbatch \
        scripts/run_rq0.py --paths configs/paths/greatlakes.yaml --run-id demo_rq0_1

Quick check first: add `--max-images 20` and a different run id. `--sources gt` needs no detection files.
Baselines vs ideal clicks (P4's rules never feed the previous logits back, so neither does the oracle here):
    ... --run-id demo_rq0_baselines_1 --no-prev-mask \
        --methods oracle_clicks b1_random_k b2_multimask b3_box_refit b3_unc_click
Writes $RUNS_ROOT/<run-id>/{results.parquet, config.yaml, manifest.json} and prints a short summary.
"""

import argparse
import json
import time

from promptcorrect.config import RunConfig, resolve_git_commit, save_run_config, set_seed
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import read_records
from promptcorrect.eval.aggregate import all_tables, as_text
from promptcorrect.eval.results import RESULTS_FILE, to_frame, write_results
from promptcorrect.eval.rq0 import BASELINES, DETECTION_FILE, SOURCES, load_detections, run_rq0


def summary(df) -> str:
    t = all_tables(df)
    return as_text({k: t[k] for k in ("failure", "methods_failures") if k in t})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--records", default="objects_val.jsonl", help="relative to data_root")
    ap.add_argument("--annotations", default="coco/annotations/instances_val2017.json", help="relative to data_root")
    ap.add_argument("--images", default="coco/val2017", help="relative to data_root")
    ap.add_argument("--detections", default="detections/{}/val.jsonl", help="relative to cache_root; {} = detector")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--methods", nargs="+", default=["oracle_clicks"], choices=("oracle_clicks",) + BASELINES,
                    help="correction methods run from every box prompt")
    ap.add_argument("--max-clicks", type=int, default=5, help="step budget per method (0 = none)")
    ap.add_argument("--no-prev-mask", action="store_true", help="do not feed the previous logits back with each click")
    ap.add_argument("--max-images", type=int, help="first N images only (quick check)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    if (out_dir / RESULTS_FILE).exists():  # fail before the GPU work, not after
        raise FileExistsError(f"{out_dir / RESULTS_FILE} exists; choose a new --run-id")
    set_seed(args.seed)

    records_file = rooted_path(args.paths, "data_root", args.records)
    ann_file = rooted_path(args.paths, "data_root", args.annotations)
    det_files = {name: rooted_path(args.paths, "cache_root", args.detections.format(name))
                 for name in {DETECTION_FILE[s] for s in args.sources if s != "gt"}}
    records = list(read_records(records_file))
    annotations = json.loads(ann_file.read_text(encoding="utf-8"))
    detections = {name: load_detections(f) for name, f in det_files.items()}

    from promptcorrect.segmenters.sam3 import Sam3Segmenter  # GPU deps only past this point

    segmenter = Sam3Segmenter(image_dirs=[rooted_path(args.paths, "data_root", args.images)], track_presence=False)
    t0 = time.time()
    rows = run_rq0(segmenter, records, annotations, detections, run_id=args.run_id, sources=tuple(args.sources),
                   methods=tuple(args.methods), max_clicks=args.max_clicks, use_prev_mask=not args.no_prev_mask,
                   max_images=args.max_images, seed=args.seed)
    seconds = time.time() - t0

    write_results(rows, out_dir)
    save_run_config(RunConfig(run_id=args.run_id, git_commit=resolve_git_commit(), seed=args.seed, backend="sam3",
                              max_steps=args.max_clicks), out_dir)
    df = to_frame(rows)
    manifest = {
        "sources": args.sources, "methods": args.methods, "max_clicks": args.max_clicks, "use_prev_mask": not args.no_prev_mask,
        "max_images": args.max_images, "seconds": round(seconds, 1), "n_rows": len(df),
        "n_objects": int(df.obj_id.nunique()), "n_images": int(df.image_id.nunique()),
        "inputs": {"records_sha256": file_sha256(records_file), "annotations_sha256": file_sha256(ann_file),
                   **{f"{n}_sha256": file_sha256(f) for n, f in det_files.items()}},
        "notes": "boundary_iou at step 0 only; SAM 3 hole filling off; sam3_text detections at the export threshold",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary(df))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
