"""Inventory of the Day 2 inputs: object records and the two detection streams (CPU, seconds).

Use it to check the three prepare/export jobs finished sensibly, and for the data/detector slide:
    python scripts/summarize_inputs.py --paths configs/paths/greatlakes.yaml
Prints counts only (no model results).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from promptcorrect.data.prepare import rooted_path
from promptcorrect.data.records import read_records


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--records", default="objects_val.jsonl")
    args = ap.parse_args()

    rec_file = rooted_path(args.paths, "data_root", args.records)
    records = list(read_records(rec_file))
    images = {r.image_id for r in records}
    print(f"records: {rec_file}\n  {len(records)} objects in {len(images)} images")
    print("  size bins:", dict(Counter(r.size_bin for r in records)))
    cats = Counter(r.category_name for r in records)
    print(f"  {len(cats)} categories; top 5:", cats.most_common(5))
    queries = {(r.image_id, r.category_name) for r in records}
    print(f"  {len(queries)} (image, phrase) queries expected from the GT-present phrase policy")

    for name in ("grounding_dino", "sam3_text"):
        det_file = rooted_path(args.paths, "cache_root", f"detections/{name}/val.jsonl")
        man_file = det_file.with_suffix(".manifest.json")
        print(f"\n{name}: {det_file}")
        if not det_file.exists() or not man_file.exists():
            print("  MISSING (output or manifest not found): the job has not completed successfully")
            continue
        rows = jsonl(det_file)
        man = json.loads(man_file.read_text())
        got = Counter((r["image_id"], r["phrase"]) for r in rows)
        print(f"  {len(rows)} detections for {len(got)}/{len(queries)} queries "
              f"(queries with zero detections: {len(queries) - len(got)})")
        per_query = Counter(got.values())
        print("  detections per query (queries with >= 1):", dict(sorted(per_query.items())))
        print("  score min/median/max:", end=" ")
        s = sorted(r["score"] for r in rows)
        print(f"{s[0]:.2f} / {s[len(s) // 2]:.2f} / {s[-1]:.2f}" if s else "n/a")
        print("  manifest:", {k: man.get(k) for k in ("model_id", "backend", "confidence_threshold", "threshold",
                                                       "text_threshold", "device", "seconds", "n_dropped_invalid", "git_commit")
                            if man.get(k) is not None})
        pres = [q["presence"] for q in man.get("queries", []) if q.get("presence") is not None]
        if pres:
            pres.sort()
            print(f"  SAM 3 presence per query: min {pres[0]:.3f}, median {pres[len(pres) // 2]:.3f}, max {pres[-1]:.3f}")
        if any("mask_rle" in r for r in rows[:5]):
            print("  rows include mask_rle: yes")


if __name__ == "__main__":
    main()
