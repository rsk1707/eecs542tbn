# Building the container image (NOT USABLE ON GREATLAKES)

> **Status 2026-10-25:** GreatLakes cannot build container images: `singularity build --fakeroot` fails with "no valid mapping entry found for srikrish". We therefore use a shared Python venv instead; see [`environment.md`](environment.md). This document is kept for a possible later route (an image built in CI or on a machine with root) and for the "What was validated" notes, which also apply to the venv.

GreatLakes provides **SingularityCE 4.4.1** (`module load singularity`; commands are `singularity ...`, env vars are `SINGULARITY_*`). The `.def` recipe format is the same as Apptainer's. Apptainer is only used in the WSL fallback below.

Recipe: [`container/promptcorrect.def`](../container/promptcorrect.def). The container build itself has **not been run yet**; the Python install steps were replayed locally (see "What was validated" below).
The image holds only dependencies (Python 3.12, PyTorch 2.10.0 + torchvision 0.25.0 for CUDA 12.6, SAM 3 and SAM 2.1 pinned to exact commits, COCO tooling). Code, data, weights and caches are bind-mounted at runtime, so code changes never need a rebuild.

**Dependency versions have one source: [`pyproject.toml`](../pyproject.toml).** The recipe copies it into the image and installs the lists from it. To change a Python package version, edit `pyproject.toml` only, then rebuild. The recipe itself only holds what pyproject can't express (CUDA torch, SAM 3 / SAM 2.1 from git, a few cluster-only extras).

**Build from the repo root** (the recipe copies `pyproject.toml` and `container/check_env.py` relative to where you run the build).

**Primary path: build on GreatLakes (steps 0a-0d).** A build is roughly 4 cores for under an hour on `standard` (a few core-hours of the account's budget; check your usage after the first run), so avoid rebuilding often. **Fallback: build in WSL (steps 1-4)**, which costs no credits but needs an upload.

## 0. Build on GreatLakes

### 0a. Setup

```bash
module load singularity
singularity --version                            # SingularityCE 4.4.1
S=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish
export PC_ROOT=$S/eecs542tbn
mkdir -p $PC_ROOT/containers $S/.sing-cache $S/.sing-tmp
export SINGULARITY_CACHEDIR=$S/.sing-cache SINGULARITY_TMPDIR=$S/.sing-tmp   # on scratch: /tmp and $HOME are too small
```

### 0b. Check that rootless builds work (about 3 minutes)

```bash
srun --account=eecs542f26s001_class --partition=standard --cpus-per-task=2 --mem=4G --time=00:15:00 --pty bash
module load singularity
S=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish      # re-set in the new shell
export SINGULARITY_CACHEDIR=$S/.sing-cache SINGULARITY_TMPDIR=$S/.sing-tmp
cd $S && singularity build --fakeroot test.sif docker://alpine && echo FAKEROOT_OK
exit
```
`FAKEROOT_OK` means you can build here. A fakeroot / `/etc/subuid` mapping error means you cannot, so use the WSL fallback (steps 1-4).

### 0c. Run the real build as a batch job

```bash
cd ~/eecs542tbn && git pull            # repo root, up to date
S=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish; export PC_ROOT=$S/eecs542tbn
sbatch --account=eecs542f26s001_class --partition=standard --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --job-name=pc_build --output=pc_build-%j.out \
  --wrap "module load singularity; export SINGULARITY_CACHEDIR=$S/.sing-cache SINGULARITY_TMPDIR=$S/.sing-tmp; singularity build --fakeroot $PC_ROOT/containers/promptcorrect-\$(date +%F).sif container/promptcorrect.def"
```
Watch it with `squeue -u $USER` and `tail -f pc_build-<jobid>.out`. About 20-40 minutes; the result is over 10 GB. Success = the log ends with the `OK`/`FAIL` list from `check_env.py` and the `.sif` exists in `$PC_ROOT/containers/`.

### 0d. Publish it for the team

```bash
cd $PC_ROOT/containers && ln -sfn promptcorrect-<date>.sif current.sif
```
Teammates read it through the ACLs set on `$PC_ROOT` (see `CLAUDE.md` section 8). Then continue at step 5 (GPU check).

## 1. WSL2 prerequisites (Windows 11)

You already have WSL2 with an Ubuntu 24.04 distro. Check with `wsl -l -v` in PowerShell, then open it (`wsl -d <name>`).
Requirements: about 30 GB free disk (the image is roughly 8–12 GB and the build needs scratch space) and at least 6 GB RAM for WSL. WSL defaults to half your RAM; to raise it, create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=10GB
```
then run `wsl --shutdown` and reopen.

**Work inside the Linux filesystem (`~`), not `/mnt/c/...`**. It is much faster and avoids permission problems.

## 2. Install Apptainer in WSL

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
apptainer --version
```

If the PPA doesn't work, download the `.deb` for your Ubuntu from the Apptainer GitHub releases page and run `sudo apt install ./apptainer_<version>_amd64.deb`.
GreatLakes runs SingularityCE 4.4.1. A `.sif` built by Apptainer in WSL normally runs there (the SIF format is shared), but this has not been tested; confirm with `check_env.py` after uploading.

## 3. Build

```bash
git clone https://github.com/<org>/eecs542tbn.git ~/eecs542tbn     # or git pull if already cloned
cd ~/eecs542tbn
mkdir -p ~/apptainer-tmp ~/apptainer-cache
export APPTAINER_TMPDIR=$HOME/apptainer-tmp APPTAINER_CACHEDIR=$HOME/apptainer-cache
DATE=$(date +%F)
sudo APPTAINER_TMPDIR=$APPTAINER_TMPDIR APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR \
    apptainer build ~/promptcorrect-$DATE.sif container/promptcorrect.def
```

Takes about 15–40 minutes. Do not commit the `.sif` (`*.sif` is gitignored; `.gitattributes` marks it binary).

The build ends by running `container/check_env.py` (`%test`), which prints `OK`/`FAIL` plus the version for every dependency (it never fails the build; read the output). Run it again any time:
```bash
apptainer exec ~/promptcorrect-$DATE.sif python /opt/check_env.py
```
Everything should say `OK`. CUDA won't be available in WSL unless you have an NVIDIA GPU and WSL driver set up. That's fine; the GPU check happens on GreatLakes (step 5).

## 4. Transfer to GreatLakes

Use the transfer node for big files. Run from inside WSL:

```bash
sha256sum ~/promptcorrect-$DATE.sif | tee ~/promptcorrect-$DATE.sif.sha256
rsync -avP --partial ~/promptcorrect-$DATE.sif \
  <uniqname>@greatlakes-xfer.arc-ts.umich.edu:/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn/containers/
```
`--partial` lets you resume if the connection drops. If SSH from your machine is awkward, Globus (personal endpoint on Windows) is the alternative.

On GreatLakes, verify and publish it for the team:
```bash
cd /scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn/containers
sha256sum promptcorrect-$DATE.sif          # must match the value printed in WSL
chmod g+r promptcorrect-$DATE.sif
ln -sfn promptcorrect-$DATE.sif current.sif
```

## 5. Test on a GPU node

```bash
srun --account=eecs542f26s001_class --partition=spgpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 --pty bash
module load singularity
S=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
singularity exec --nv --bind /scratch/eecs542f26s001_class_root/eecs542f26s001_class $S/containers/current.sif \
    python /opt/check_env.py      # expect all OK and "cuda available: True"
nvidia-smi | head -4              # note the driver version (CUDA 12.6 wheels need a recent enough driver)
```
Then run the SAM 3 smoke test (text, box and point prompts on one image) and record what worked in `CLAUDE.md` section 12.

## Updating the image

Edit `container/promptcorrect.def`, rebuild with a new date, upload, and move the `current.sif` symlink. Tell the team in the same PR. Keep the previous `.sif` until the new one is confirmed.

## Runtime environment variables (never baked into the image)

```bash
export HF_HOME=$S/hf_cache            # model downloads go to scratch, not $HOME
export HF_TOKEN=...                    # your own token, from your shell only
export SINGULARITYENV_HF_HOME=$HF_HOME   # pass into the container (SingularityCE prefix)
export SINGULARITYENV_HF_TOKEN=$HF_TOKEN
```

## What was validated (and what was not)

Before the first build, the recipe's Python stages were replayed on real Python 3.12 (Windows, CPU build of torch 2.10.0): pinned SAM 3 / SAM 2.1 commits, the dependency list extracted from `pyproject.toml` by the recipe's own line, one `pip` resolver pass, `pip check`, and `container/check_env.py`. That caught and fixed:
- `pkg_resources` missing (newest `setuptools` removed it, `import sam3` needs it) -> `setuptools<81`.
- `pyarrow` 26 needs NumPy 2 at import time (undeclared), SAM 3 forces `numpy<2` -> `pyarrow<26`, `numpy<2`.
- SAM 2.1 needs `--no-build-isolation`; torch must be installed first from the CUDA index; `transformers` must stay `<5`.

**Not validated** (only the first real build will show): the `apt` packages and Apptainer/Singularity syntax, the CUDA build of torch (the replay used the CPU build), and `import sam3` on Linux. On Windows it fails only with `No module named 'triton'`; Linux CUDA wheels of torch install `triton` themselves, so this is expected not to happen in the image. If `check_env.py` shows `FAIL sam3` after the real build, read the error before changing anything.

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `no space left on device` during build | Set `APPTAINER_TMPDIR` to a path in WSL's Linux filesystem; check `df -h ~`. |
| Build killed / out of memory | Raise WSL memory in `.wslconfig` (step 1). |
| `pip` can't find a torch wheel | Check the `torch==` / `torchvision==` pair in the `.def` and the cu126 index URL (torch 2.10.0 pairs with torchvision 0.25.0). |
| `sam3` import fails | Read the SAM 3 README for current install requirements; update the `%post` block. SAM 3 needs Python >= 3.12 and `numpy<2`. |
| `FATAL: ... pyproject.toml: no such file` during `%files` | You ran the build from outside the repo root. `cd` to the repo root first. |
| `torch.cuda.is_available()` is False on a GPU node | Missing `--nv`, or the node's driver is too old for the CUDA 12.6 wheels (check `nvidia-smi`). |
| CUDA not available on GreatLakes | You forgot `--nv`, or you are on a CPU node. |
| `.sbatch` scripts fail with `^M` errors | CRLF line endings. Re-checkout with LF (`.gitattributes` should prevent this). |
| Pulling/building fills `$HOME` quota | Set `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` to scratch (`APPTAINER_*` in WSL). |
