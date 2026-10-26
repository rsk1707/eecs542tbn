"""Tables and figures for one RQ0 run (CPU only; runs in seconds). Owner: P3.

    python scripts/summarize_rq0.py <run_dir> [--out results/<run_id>] [--index-owner P3]

Writes <out>/tables/*.csv, <out>/tables.txt and <out>/figures/*.png (default <out> = <run_dir>/summary).
`--out results/<run_id>` puts them in git (summary tables and figures only, never per-object rows).
`--index-owner` also adds the run's headline numbers to results/runs_index.csv (commit taken from its config.yaml).
"""

import argparse
from pathlib import Path

import yaml

from promptcorrect.eval.aggregate import all_tables, as_text, headline
from promptcorrect.eval.results import read_results, update_runs_index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--index-owner", help="add the run to results/runs_index.csv under this owner")
    ap.add_argument("--notes", default="")
    ap.add_argument("--runs-path", help="where the run lives, for the index (default: run_dir), e.g. its GreatLakes path")
    args = ap.parse_args()

    df = read_results(args.run_dir)
    out = args.out or args.run_dir / "summary"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    tables = all_tables(df)
    for name, t in tables.items():
        t.to_csv(out / "tables" / f"{name}.csv")
    (out / "tables.txt").write_text(as_text(tables) + "\n")
    print(as_text(tables))

    if not args.no_figures:
        from promptcorrect.viz.rq0_figures import save_all

        for p in save_all(df, out / "figures"):
            print("wrote", p)
    if args.index_owner:
        run_id = df.run_id.iloc[0]
        commit = yaml.safe_load((args.run_dir / "config.yaml").read_text())["git_commit"]
        print("index:", update_runs_index(run_id, args.index_owner, args.runs_path or args.run_dir.resolve(), args.notes,
                                          headline(df), git_commit=commit))


if __name__ == "__main__":
    main()
