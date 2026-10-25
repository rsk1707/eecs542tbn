# Python environment on GreatLakes

GreatLakes cannot build container images for us (`singularity build --fakeroot` fails with "no valid mapping entry found"), so we use **one shared Python 3.12 virtual environment on scratch**. It is built by [`slurm/setup_env.sbatch`](../slurm/setup_env.sbatch) from the same pins that were validated for the container recipe: torch 2.10.0 + torchvision 0.25.0 (CUDA 12.6 wheels), SAM 3 and SAM 2.1 at pinned commits, and every other package from [`pyproject.toml`](../pyproject.toml), installed in a single `pip` pass.

`pyproject.toml` is the single source of dependency versions. The script's own pins (torch, SAM commits) are at the top of the script.

## Using it (everyone)

```bash
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
source $PC_ROOT/venv/bin/activate
export PYTHONPATH=$PWD/src            # from the repo root; the project itself is not pip-installed
python scripts/<script>.py --config configs/experiments/<x>.yaml
```

In an sbatch script, put those three lines before the `python` command and request the GPU partition (`--partition=spgpu --gres=gpu:1`). Model weights and data are paths from `configs/paths/greatlakes.yaml`; set `PC_PATHS=greatlakes`.

Check the environment any time (expect all `OK`; `cuda available: True` only on a GPU node):

```bash
python container/check_env.py
```

## Building or rebuilding it (Srikrishnan only)

Run from the repo root on GreatLakes, as a batch job (not on the login node):

```bash
cd ~/eecs542tbn && git pull
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
sbatch --account=eecs542f26s001_class slurm/setup_env.sbatch
tail -f pc_env-<jobid>.out
```

What it does, in order: finds Python 3.12 (system, or a standalone one installed with `uv` under `$PC_ROOT/tools`) -> creates `$PC_ROOT/venv-<date>` -> `pip`, `wheel`, `setuptools<81` -> CUDA torch -> clones SAM 3 / SAM 2.1 into `$PC_ROOT/src_deps` at the pinned commits -> dependency lists from `pyproject.toml` -> one resolver pass -> `pip check` -> `check_env.py`.

- It only repoints the symlink `$PC_ROOT/venv` to the new directory if `check_env.py` shows no `FAIL`. Teammates never see a half-built environment, and the previous environment stays until you delete it.
- A same-day rebuild needs `FORCE=1 sbatch ...` (it deletes that day's directory first).
- To use a specific Python: `PYTHON=/path/to/python3.12 sbatch ...`.
- Roughly 4 cores for 20-40 minutes (about 3 GB of downloads), a few core-hours of the account budget.

After the first build, run `python container/check_env.py` on a GPU node (`srun --partition=spgpu --gres=gpu:1 ...`) and expect `cuda available: True`. Also run `nvidia-smi | head -4` and note the driver version (the CUDA 12.6 wheels need a recent enough driver).

## Changing a dependency version

Edit `pyproject.toml` (or the pins at the top of the script for torch / SAM commits), push, `git pull` on GreatLakes, and rebuild as above. Tell the team when the symlink moves.

## Known limits

- **Not tested on GreatLakes yet.** The Python install steps were replayed on Windows with the CPU build of torch (see the "What was validated" section in [`container.md`](container.md)). The first real run is the real test. One expected difference: on Windows `import sam3` fails with `No module named 'triton'`; Linux CUDA torch installs `triton` itself, so it should work on GreatLakes. If `check_env.py` shows `FAIL sam3` there, read the error before changing anything.
- Less isolated than a container: the venv uses scratch paths and the cluster's system libraries. Scratch purges files not accessed for ~60 days; the environment can be rebuilt with one command.
- Teammates need read/execute access to `$PC_ROOT` (the `setfacl` step in `CLAUDE.md` section 8).
