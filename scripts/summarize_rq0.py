"""Print (and save as CSV) the headline RQ0 tables for a finished run, without rerunning anything.

    python scripts/summarize_rq0.py --paths configs/paths/greatlakes.yaml --run-id demo_rq0_1
"""

import argparse

import pandas as pd

from promptcorrect.data.prepare import rooted_path
from promptcorrect.experiments.rq0_summary import format_report, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    out_dir = rooted_path(args.paths, "runs_root", args.run_id)
    df = pd.read_parquet(out_dir / "results.parquet")
    tables = summarize(df)
    for name, t in tables.items():
        t.to_csv(out_dir / f"summary_{name}.csv")
    print(format_report(tables))


if __name__ == "__main__":
    main()
