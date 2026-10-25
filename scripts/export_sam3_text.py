"""Export SAM 3 text-prompt detections for the prepared demo objects (run on a GPU node).

Same phrase policy as P2's Grounding DINO export: only unique categories of the retained GT objects in
each image are queried. Output rows are {image_id, phrase, box, score, mask_rle}. A manifest next to the
output records the settings and one entry per query (detections kept, invalid ones dropped, SAM 3's raw
presence probability), so an empty result is distinguishable from a missing run.

    sbatch --account=eecs542f26s001_class --time=01:00:00 slurm/run_gpu.sbatch \
        scripts/export_sam3_text.py --paths configs/paths/greatlakes.yaml --threshold 0.5

The confidence threshold changes the RQ0 failure rates (0.5 is SAM 3's default); the team must fix one
value and record it in CLAUDE.md before the final runs.
"""

import argparse
import importlib.metadata
import json
import random
import subprocess
import time

from promptcorrect.data.io import new_text_file
from promptcorrect.data.prepare import file_sha256, rooted_path
from promptcorrect.data.records import read_records
from promptcorrect.detectors.export import export_detections
from promptcorrect.detectors.sam3_text import RecordingSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", required=True, help="paths YAML, e.g. configs/paths/greatlakes.yaml")
    parser.add_argument("--records", default="objects_val.jsonl", help="relative to data_root")
    parser.add_argument("--images", default="coco/val2017", help="image directory, relative to data_root")
    parser.add_argument("--output", default="detections/sam3_text/val.jsonl", help="relative to cache_root")
    parser.add_argument("--ckpt", help="SAM 3 checkpoint (default: <ckpt_root>/sam3/sam3.pt)")
    parser.add_argument("--threshold", type=float, default=0.5, help="SAM 3 detection confidence threshold")
    parser.add_argument("--no-presence", action="store_true", help="skip the extra pass that reads the raw presence")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    import torch

    from promptcorrect.config import load_paths
    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records_file = rooted_path(args.paths, "data_root", args.records)
    image_dir = rooted_path(args.paths, "data_root", args.images)
    output = rooted_path(args.paths, "cache_root", args.output)
    manifest_file = output.with_suffix(".manifest.json")
    if output.exists() or manifest_file.exists():
        raise FileExistsError(f"{output} or its manifest already exists; move it away or choose --output")
    ckpt = args.ckpt or str(load_paths().ckpt_root / "sam3" / "sam3.pt")

    records = list(read_records(records_file))
    t0 = time.time()
    segmenter = Sam3Segmenter(ckpt_path=ckpt, image_dirs=[image_dir], confidence_threshold=args.threshold,
                              track_presence=not args.no_presence)
    recorder = RecordingSegmenter(segmenter)
    count = export_detections(records, output, segmenter=recorder)
    seconds = time.time() - t0

    manifest = {
        "backend": "sam3", "checkpoint": ckpt, "checkpoint_sha256": file_sha256(ckpt),
        "confidence_threshold": args.threshold, "fill_small_holes": segmenter.fill_small_holes,
        "seed": args.seed, "device": segmenter.device, "seconds": round(seconds, 1),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "records_sha256": file_sha256(records_file), "detections_sha256": file_sha256(output),
        "n_detections": count, "n_queries": len(recorder.queries),
        "n_dropped_invalid": sum(q["n_dropped_invalid"] for q in recorder.queries),
        "phrase_policy": "unique categories of retained objects per image",
        "queries": recorder.queries,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow")},
    }
    with new_text_file(manifest_file) as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "queries"}, indent=2))


if __name__ == "__main__":
    main()
