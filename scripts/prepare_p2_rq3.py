"""Draft a category review, or prepare leakage-safe COCO/LVIS records (CPU only)."""

import argparse
import json

from promptcorrect.data.prepare import rooted_path
from promptcorrect.data.prepare_rq3 import prepare_rq3, write_category_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft-review", help="write unreviewed synset candidates")
    prepare = commands.add_parser("prepare", help="require reviewed decisions and build records")
    for command in (draft, prepare):
        command.add_argument("--coco-val", default="coco/annotations/instances_val2017.json")
        command.add_argument("--lvis-val", default="lvis/lvis_v1_val.json")
    draft.add_argument("--coco-synsets", required=True)
    draft.add_argument("--output", required=True)
    prepare.add_argument("--coco-train", default="coco/annotations/instances_train2017.json")
    prepare.add_argument("--category-review", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--train-count", type=int)
    prepare.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    resolve = lambda path: rooted_path(args.paths, "data_root", path)
    if args.command == "draft-review":
        result = write_category_review(resolve(args.coco_val), resolve(args.lvis_val),
                                       resolve(args.coco_synsets), resolve(args.output))
    else:
        result = prepare_rq3(resolve(args.coco_train), resolve(args.coco_val),
                             resolve(args.lvis_val), resolve(args.category_review),
                             resolve(args.output_dir), train_count=args.train_count, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
