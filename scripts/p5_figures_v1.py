"""Corrector v1 slide / poster figures from an eval run's ResultRows and its calibration curve (CPU, seconds).
See promptcorrect/corrector/figures_v1.py.

    python scripts/p5_figures_v1.py --results $RUNS_ROOT/p5v1_eval_0/results.parquet \
        --calib results/p5v1_calib_0 --out results/p5v1_eval_0/figures

Writes v1_tradeoff.png, v1_bars.png, v1_severity.png, v1_calibration.png.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from promptcorrect.corrector.figures_v1 import bars, calibration, severity, tradeoff
from promptcorrect.eval.results import read_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results.parquet (or its run dir)")
    ap.add_argument("--calib", required=True, help="calibration run dir (calibration.json, curve_*.csv)")
    ap.add_argument("--feedback", default="feedback", choices=("feedback", "nofeedback"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = read_results(args.results)
    calib = Path(args.calib)
    chosen = json.loads((calib / "calibration.json").read_text())[args.feedback]
    curve = pd.read_csv(calib / f"curve_{args.feedback}.csv")
    for p in (tradeoff(df, out / "v1_tradeoff.png"), bars(df, out / "v1_bars.png"),
              severity(df, out / "v1_severity.png"), calibration(curve, chosen, out / "v1_calibration.png")):
        print(p)


if __name__ == "__main__":
    main()
