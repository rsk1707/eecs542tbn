# Building the Apptainer image and moving it to GreatLakes

Recipe: [`container/promptcorrect.def`](../container/promptcorrect.def). It is an **untested draft**, so expect to adjust versions on the first build.
The image holds only dependencies (Python 3.12, PyTorch 2.7 cu126, SAM 3, SAM 2.1, COCO/LVIS tooling). Code, data, weights and caches are bind-mounted at runtime, so code changes never need a rebuild.

**Option A (preferred, if it works):** build directly on GreatLakes with `--fakeroot` (step 0).
**Option B (fallback):** build on a Windows machine through WSL2 and upload the `.sif` (steps 1–5).

## 0. Can GreatLakes build it itself?

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu      # UM VPN may be needed off campus; Duo 2FA
module avail apptainer singularity               # find the module name
module load apptainer                            # or singularity
cd /scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish
export APPTAINER_CACHEDIR=$PWD/.apptainer-cache APPTAINER_TMPDIR=$PWD/.apptainer-tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
apptainer build --fakeroot test.sif docker://alpine   # works? then use Option A for the real build
```

If that works, run the real build as a Slurm job on `standard` (not on the login node) with the same `--fakeroot` flag and the `.def` from the repo. If it fails with a fakeroot or user-namespace error, use Option B.

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
Check the GreatLakes version too (`module load apptainer && apptainer --version`). A `.sif` built with a similar or older version runs fine there.

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

Quick CPU sanity check:
```bash
apptainer exec ~/promptcorrect-$DATE.sif python -c "import torch, sam3, sam2, pycocotools; print(torch.__version__)"
```
CUDA won't be available in WSL unless you have an NVIDIA GPU and WSL driver set up. That's fine; the GPU check happens on GreatLakes (step 5).

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
module load apptainer     # or singularity
S=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
apptainer exec --nv --bind /scratch/eecs542f26s001_class_root/eecs542f26s001_class $S/containers/current.sif \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Then run the SAM 3 smoke test (text, box and point prompts on one image) and record what worked in `CLAUDE.md` section 12.

## Updating the image

Edit `container/promptcorrect.def`, rebuild with a new date, upload, and move the `current.sif` symlink. Tell the team in the same PR. Keep the previous `.sif` until the new one is confirmed.

## Runtime environment variables (never baked into the image)

```bash
export HF_HOME=$S/hf_cache            # model downloads go to scratch, not $HOME
export HF_TOKEN=...                    # your own token, from your shell only
export APPTAINERENV_HF_HOME=$HF_HOME   # pass into the container
export APPTAINERENV_HF_TOKEN=$HF_TOKEN
```

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `no space left on device` during build | Set `APPTAINER_TMPDIR` to a path in WSL's Linux filesystem; check `df -h ~`. |
| Build killed / out of memory | Raise WSL memory in `.wslconfig` (step 1). |
| `pip` can't find a torch wheel | Check the `torch==` / `torchvision==` pair in the `.def` and the cu126 index URL. |
| `sam3` import fails | Read the SAM 3 README for current install requirements; update the `%post` block. |
| CUDA not available on GreatLakes | You forgot `--nv`, or you are on a CPU node. |
| `.sbatch` scripts fail with `^M` errors | CRLF line endings. Re-checkout with LF (`.gitattributes` should prevent this). |
| Pulling/building fills `$HOME` quota | Set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to scratch. |
