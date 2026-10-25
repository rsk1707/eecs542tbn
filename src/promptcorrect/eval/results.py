"""Results writer (docs/interfaces.md section 5, O11).

Per-object rows go to `$RUNS_ROOT/{run_id}/results.parquet` (never git). One row per run
goes to the git-tracked `results/runs_index.csv`. Only raw values are stored; success rates,
harm rate etc. are computed at analysis time.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from promptcorrect.config import REPO_ROOT, resolve_git_commit
from promptcorrect.eval.methods import check_method
from promptcorrect.types import ResultRow

RESULTS_FILE = "results.parquet"
RUNS_INDEX = REPO_ROOT / "results" / "runs_index.csv"
KEY = ["obj_id", "method", "step"]
INDEX_COLUMNS = ["run_id", "owner", "date", "git_commit", "runs_path", "notes"]


def _dtype(type_str: str) -> str:
    # Annotations are strings (`from __future__ import annotations` in types.py), e.g. "float | None".
    base = type_str.split("|")[0].strip()
    return {"str": "string", "int": "int64", "float": "float64", "bool": "bool"}[base]


# Derived from the dataclass, so a column added to ResultRow is picked up here automatically.
SCHEMA = {f.name: _dtype(f.type) for f in fields(ResultRow)}


def to_frame(rows: list[ResultRow]) -> pd.DataFrame:
    """Validate rows and return them as a DataFrame with the fixed column order and dtypes."""
    df = pd.DataFrame([asdict(r) for r in rows], columns=list(SCHEMA)).astype(SCHEMA)
    if df.empty:
        return df
    if df["run_id"].nunique() != 1:
        raise ValueError(f"one run per file, got run_ids {sorted(df['run_id'].unique())}")
    for m in df["method"].unique():
        check_method(m)
    if (df["step"] < 0).any():
        raise ValueError("step must be >= 0")
    dup = df.duplicated(KEY, keep=False)
    if dup.any():
        raise ValueError(f"duplicate (obj_id, method, step) rows:\n{df.loc[dup, KEY].head()}")
    return df


def write_results(rows: list[ResultRow], run_dir: str | Path, overwrite: bool = False) -> Path:
    """Write `run_dir/results.parquet`. Refuses to replace an existing file unless overwrite=True."""
    df = to_frame(rows)
    run_dir = Path(run_dir)
    if not df.empty and df["run_id"].iloc[0] != run_dir.name:
        raise ValueError(f"run_id {df['run_id'].iloc[0]!r} does not match run dir {run_dir.name!r}")
    out = run_dir / RESULTS_FILE
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pick a new run_id or pass overwrite=True")
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_results(path: str | Path) -> pd.DataFrame:
    """Read one run's results (a run dir or the parquet file itself)."""
    path = Path(path)
    if path.is_dir():
        path = path / RESULTS_FILE
    return pd.read_parquet(path).astype(SCHEMA)


def update_runs_index(
    run_id: str,
    owner: str,
    runs_path: str | Path,
    notes: str = "",
    headline: dict[str, float] | None = None,
    index_path: str | Path = RUNS_INDEX,
) -> Path:
    """Add (or replace) this run's row in runs_index.csv. `headline` adds columns, e.g. {"miou": 0.81}."""
    row = {
        "run_id": run_id,
        "owner": owner,
        "date": dt.date.today().isoformat(),
        "git_commit": resolve_git_commit(),
        "runs_path": str(runs_path),
        "notes": notes,
        **(headline or {}),
    }
    index_path = Path(index_path)
    old = pd.read_csv(index_path, dtype=str) if index_path.exists() else pd.DataFrame(columns=INDEX_COLUMNS)
    old = old[old["run_id"] != run_id]
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(index_path, index=False)
    return index_path
