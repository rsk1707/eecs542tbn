"""Path roots, run configs, seeding. Plain YAML -> dataclass; no config framework."""

from __future__ import annotations

import os
import random
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class Paths:
    data_root: Path
    cache_root: Path
    ckpt_root: Path
    runs_root: Path

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id


def load_paths(name: str | None = None) -> Paths:
    """Load configs/paths/<name>.yaml. `name` defaults to $PC_PATHS, then "local".

    Relative entries resolve against the repo root. Directories are not created here.
    """
    name = name or os.environ.get("PC_PATHS", "local")
    cfg_file = CONFIG_DIR / "paths" / f"{name}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"No paths config {cfg_file} (set PC_PATHS to local or greatlakes)")
    raw = yaml.safe_load(cfg_file.read_text())
    resolved = {}
    for key in ("data_root", "cache_root", "ckpt_root", "runs_root"):
        if key not in raw:
            raise KeyError(f"{cfg_file} is missing '{key}'")
        p = Path(raw[key])
        resolved[key] = p if p.is_absolute() else REPO_ROOT / p
    return Paths(**resolved)


@dataclass
class RunConfig:
    """Schema: docs/interfaces.md section 6. Unknown keys raise an error."""

    run_id: str
    git_commit: str = "auto"
    seed: int = 0
    backend: str = "sam3"               # "sam3" | "sam21" | "dummy"
    train_data: str = "synthetic"       # synthetic | real | mixed | matched
    perturbations: list[str] = field(default_factory=lambda: ["shift", "scale", "truncate"])
    max_steps: int = 3
    splits_file: str = "splits.json"    # relative to data_root; fixed once, never regenerated


def resolve_git_commit() -> str:
    """Current commit hash, with '+dirty' if the working tree has uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_run_config(path: str | Path) -> RunConfig:
    """Load a run config YAML; resolve `git_commit: auto`; reject unknown keys."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    allowed = {f.name for f in fields(RunConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)} (allowed: {sorted(allowed)})")
    cfg = RunConfig(**raw)
    if cfg.git_commit == "auto":
        cfg.git_commit = resolve_git_commit()
    return cfg


def save_run_config(cfg: RunConfig, out_dir: str | Path) -> Path:
    """Write the *resolved* config next to a run's results as config.yaml."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "config.yaml"
    out.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
