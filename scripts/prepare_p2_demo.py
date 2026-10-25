"""Build the fixed COCO demo subset and object JSONL."""

import argparse
import json

from promptcorrect.data.prepare import prepare_demo, rooted_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--annotations", default="coco/annotations/instances_val2017.json")
    parser.add_argument("--split-file", default="splits_demo.json")
    parser.add_argument("--objects-file", default="objects_val.jsonl")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    resolve = lambda path: rooted_path(args.paths, "DATA_ROOT", path)
    manifest = prepare_demo(resolve(args.annotations), resolve(args.split_file),
                            resolve(args.objects_file), count=args.count, seed=args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
