"""RQ0 preliminary study on the prepared demo objects (run on a GPU node). See promptcorrect/experiments/rq0.py.

    sbatch --account=eecs542f26s001_class --time=02:00:00 slurm/run_gpu.sbatch \
        scripts/run_rq0.py --paths configs/paths/greatlakes.yaml --run-id demo_rq0_1

Needs (all from the Day 2 jobs in docs/environment.md): objects_val.jsonl, the COCO val annotations,
and the Grounding DINO / SAM 3 text detections. `--sources gt_box` runs the ceiling only and needs no
detections and no `match`. Matching uses P3's `match` once it is implemented, otherwise a stand-in with
the same contract (experiments/match_fallback.py); rq0_manifest.json records which one. Writes $RUNS_ROOT/<run-id>/{results.parquet, queries.parquet, config.yaml,
rq0_manifest.json, summary_*.csv}; prints the headline tables.
"""

import argparse
import json
import subprocess
import time

import pandas as pd

from promptcorrect.config import RunConfig, save_run_config, set_seed
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import read_records
from promptcorrect.experiments.rq0 import ORACLE_SOURCES, SOURCES, load_detections, run_rq0
from promptcorrect.experiments.rq0_summary import format_report, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--records", default="objects_val.jsonl", help="relative to data_root")
    ap.add_argument("--annotations", default="coco/annotations/instances_val2017.json", help="relative to data_root")
    ap.add_argument("--images", default="coco/val2017", help="relative to data_root")
    ap.add_argument("--gd-detections", default="detections/grounding_dino/val.jsonl", help="relative to cache_root")
    ap.add_argument("--sam3-detections", default="detections/sam3_text/val.jsonl", help="relative to cache_root")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--oracle-steps", type=int, default=5, help="ideal clicks per object (0 = skip)")
    ap.add_argument("--max-images", type=int, help="first N images only (quick test)")
    ap.add_argument("--boundary-all", action="store_true", help="Boundary IoU at every step (slow)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    if (out_dir / "results.parquet").exists():
        raise FileExistsError(f"{out_dir}/results.parquet exists; choose a new --run-id")
    set_seed(args.seed)

    records_file = rooted_path(args.paths, "data_root", args.records)
    ann_file = rooted_path(args.paths, "data_root", args.annotations)
    images = rooted_path(args.paths, "data_root", args.images)
    gd_file = rooted_path(args.paths, "cache_root", args.gd_detections)
    sam_file = rooted_path(args.paths, "cache_root", args.sam3_detections)
    need_gd = "grounding_dino_box" in args.sources
    need_sam = any(s.startswith("sam3_text") for s in args.sources)

    records = list(read_records(records_file))
    annotations = json.loads(ann_file.read_text(encoding="utf-8"))
    gd = load_detections(gd_file) if need_gd else {}
    sam = load_detections(sam_file) if need_sam else {}

    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    segmenter = Sam3Segmenter(image_dirs=[images], track_presence=False)
    t0 = time.time()
    try:
        df, queries = run_rq0(segmenter, records, annotations, gd, sam, run_id=args.run_id, sources=tuple(args.sources),
                              oracle_steps=args.oracle_steps, boundary_all=args.boundary_all, max_images=args.max_images)
    except NotImplementedError as e:
        raise SystemExit(f"{e}\nA function the study needs is still a stub (see the message above).") from e
    seconds = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "results.parquet", index=False)
    queries.to_parquet(out_dir / "queries.parquet", index=False)
    save_run_config(RunConfig(run_id=args.run_id, git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), seed=args.seed, backend="sam3", max_steps=args.oracle_steps), out_dir)
    manifest = {
        "sources": args.sources, "oracle_sources": [s for s in ORACLE_SOURCES if s in args.sources],
        "oracle_steps": args.oracle_steps, "max_images": args.max_images, "seconds": round(seconds, 1),
        "match_impl": df.attrs.get("match_impl"), "n_rows": len(df), "n_objects": int(df.obj_id.nunique()), "n_images": int(df.image_id.nunique()),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "inputs": {"records_sha256": file_sha256(records_file), "annotations_sha256": file_sha256(ann_file),
                   **({"grounding_dino_sha256": file_sha256(gd_file)} if need_gd else {}),
                   **({"sam3_text_sha256": file_sha256(sam_file)} if need_sam else {})},
        "notes": "boundary_iou is step 0 only unless --boundary-all; SAM 3 hole filling is off (see CLAUDE.md)",
    }
    (out_dir / "rq0_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tables = summarize(df)
    for name, t in tables.items():
        t.to_csv(out_dir / f"summary_{name}.csv")
    print(format_report(tables))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
