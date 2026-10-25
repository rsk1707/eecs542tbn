"""Export Grounding DINO detections for the prepared demo objects."""

import argparse
import json

from promptcorrect.data.prepare import rooted_path
from promptcorrect.detectors.run import run_grounding_dino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--annotations", default="coco/annotations/instances_val2017.json")
    parser.add_argument("--records", default="objects_val.jsonl")
    parser.add_argument("--images", default="coco/val2017")
    parser.add_argument("--output", default="detections/grounding_dino/val.jsonl")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = lambda path: rooted_path(args.paths, "DATA_ROOT", path)
    manifest = run_grounding_dino(
        data(args.annotations), data(args.records), data(args.images),
        rooted_path(args.paths, "CACHE_ROOT", args.output), model_id=args.model_id,
        revision=args.revision, device=args.device, threshold=args.threshold,
        text_threshold=args.text_threshold, seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
