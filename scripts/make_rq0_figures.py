"""Slide-ready figures for a finished RQ0 run (CPU, seconds): failure rates, ideal-click curves, SAM 3
overconfidence, failures by object size. See promptcorrect/viz/rq0_figures.py for the design choices.

    python scripts/make_rq0_figures.py --paths configs/paths/greatlakes.yaml --run-id demo_rq0_1
Writes PNGs to $RUNS_ROOT/<run-id>/figures/ (copy them to your laptop with scp/rsync for the slides).
"""

import argparse

import pandas as pd

from promptcorrect.data.prepare import rooted_path
from promptcorrect.viz.rq0_figures import make_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_dir = rooted_path(args.paths, "runs_root", args.run_id)
    df = pd.read_parquet(run_dir / "results.parquet")
    for p in make_all(df, run_dir / "figures"):
        print(p)


if __name__ == "__main__":
    main()
