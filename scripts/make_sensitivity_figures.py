"""Slide-ready figures for a finished sensitivity / baselines run (CPU, seconds). See
promptcorrect/viz/sensitivity_figures.py.

    python scripts/make_sensitivity_figures.py --paths configs/paths/greatlakes.yaml --run-id demo_sens_1
Writes PNGs to $RUNS_ROOT/<run-id>/figures/.
"""

import argparse

import pandas as pd

from promptcorrect.data.prepare import rooted_path
from promptcorrect.viz.sensitivity_figures import make_all


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
