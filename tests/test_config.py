from pathlib import Path

import pytest
import yaml

from promptcorrect.config import REPO_ROOT, load_paths, load_run_config, save_run_config, set_seed


def test_load_paths_local_resolves_under_repo():
    p = load_paths("local")
    assert p.data_root.is_absolute() and REPO_ROOT in p.data_root.parents
    assert p.run_dir("demo_x_0") == p.runs_root / "demo_x_0"


def test_load_paths_unknown_name():
    with pytest.raises(FileNotFoundError):
        load_paths("does_not_exist")


def test_example_run_config_resolves_git_commit():
    cfg = load_run_config(REPO_ROOT / "configs" / "experiments" / "demo_example.yaml")
    assert cfg.git_commit != "auto"
    assert cfg.perturbations == ["shift", "scale", "truncate"]


def test_unknown_key_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump({"run_id": "x", "not_a_key": 1}))
    with pytest.raises(ValueError, match="not_a_key"):
        load_run_config(f)


def test_save_run_config(tmp_path):
    cfg = load_run_config(REPO_ROOT / "configs" / "experiments" / "demo_example.yaml")
    out = save_run_config(cfg, tmp_path / "run")
    assert Path(out).exists()
    assert yaml.safe_load(Path(out).read_text())["run_id"] == cfg.run_id


def test_set_seed_runs():
    set_seed(0)


def test_relative_paths_resolve_against_the_yaml_folder(tmp_path, monkeypatch):
    """Same rule as data.prepare.rooted_path, so scripts and load_paths() agree."""
    from promptcorrect import config
    from promptcorrect.data.prepare import rooted_path

    (tmp_path / "paths").mkdir()
    yml = tmp_path / "paths" / "x.yaml"
    yml.write_text("\n".join(["data_root: ../d", "cache_root: ../c", "ckpt_root: /abs/k", "runs_root: r", ""]))
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    p = config.load_paths("x")
    assert p.data_root == (tmp_path / "d").resolve() and p.runs_root == (tmp_path / "paths" / "r").resolve()
    assert p.ckpt_root.as_posix().endswith("/abs/k")
    for key in ("data_root", "cache_root", "runs_root"):                  # identical to the scripts' resolver
        assert rooted_path(yml, key, "x").resolve() == getattr(p, key) / "x"   # same place, maybe spelled with ".."
