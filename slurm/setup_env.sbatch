#!/bin/bash
#SBATCH --job-name=pc_env
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pc_env-%j.out
#
# Builds the shared Python environment on scratch (GreatLakes cannot build containers: no fakeroot).
# Same install steps that were validated for container/promptcorrect.def: CUDA torch first, SAM 3 and
# SAM 2.1 at pinned commits, then ONE pip pass over the dependency lists in pyproject.toml.
#
# Run from the REPO ROOT (it reads pyproject.toml and container/check_env.py from there):
#   export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
#   sbatch --account=eecs542f26s001_class slurm/setup_env.sbatch
#
# It builds into $PC_ROOT/venv-<date> and only repoints the symlink $PC_ROOT/venv when everything
# succeeded, so teammates never see a half-built environment. Everyone then uses:
#   source $PC_ROOT/venv/bin/activate
#
# Options (environment variables):
#   PYTHON=/path/to/python3.12   use this interpreter instead of searching / installing one
#   FORCE=1                      delete and rebuild today's venv-<date> if it already exists
set -euo pipefail
: "${PC_ROOT:?set PC_ROOT to the shared eecs542tbn directory}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$REPO/pyproject.toml" && -f "$REPO/container/check_env.py" ]] || {
  echo "Run this from the repo root (pyproject.toml not found in $REPO)"; exit 1; }

# ---- pins (keep in sync with container/promptcorrect.def if that file is ever used again) ----
TORCH_VERSION=2.10.0
TORCHVISION_VERSION=0.25.0
TORCH_INDEX=https://download.pytorch.org/whl/cu126
SAM3_COMMIT=f1647e8d0c1d280991f4d1b7d974c2704537fe66
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4

STAMP=$(date +%F)
VENV="$PC_ROOT/venv-$STAMP"
DEPS="$PC_ROOT/src_deps"          # SAM 3 / SAM 2.1 checkouts (editable installs point here)
TOOLS="$PC_ROOT/tools"
export TMPDIR="$PC_ROOT/tmp"; mkdir -p "$TMPDIR" "$DEPS" "$TOOLS"
export UV_CACHE_DIR="$PC_ROOT/.uv-cache"

if [[ -e "$VENV" ]]; then
  if [[ "${FORCE:-0}" == "1" ]]; then rm -rf "$VENV"; else
    echo "$VENV already exists. Re-run with FORCE=1 to rebuild it."; exit 1; fi
fi

is_py312() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; }

echo "== 1) Python 3.12"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.12 python3; do
    if command -v "$c" >/dev/null && is_py312 "$c"; then PY=$(command -v "$c"); break; fi
  done
fi
if [[ -n "$PY" ]]; then
  echo "using $PY"
  "$PY" -m venv "$VENV"
else
  echo "no system Python 3.12; installing a standalone one with uv into $TOOLS"
  if [[ ! -x "$TOOLS/uv/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$TOOLS/uv" sh
  fi
  export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
  "$TOOLS/uv/uv" python install 3.12
  "$TOOLS/uv/uv" venv --seed --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version
is_py312 python || { echo "venv is not Python 3.12"; exit 1; }

echo "== 2) pip, wheel, setuptools<81 (newer setuptools drop pkg_resources, which import sam3 needs)"
pip install --no-cache-dir --upgrade pip wheel "setuptools<81"

echo "== 3) torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION from the CUDA 12.6 index"
pip install --no-cache-dir "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"

echo "== 4) SAM 3 and SAM 2.1 at pinned commits"
checkout() {  # checkout <url> <dir> <commit>
  if [[ ! -d "$2/.git" ]]; then git clone -q "$1" "$2"; fi
  git -C "$2" fetch -q origin
  git -C "$2" checkout -q "$3"
  echo "$2 @ $(git -C "$2" rev-parse HEAD)"
}
checkout https://github.com/facebookresearch/sam3.git "$DEPS/sam3" "$SAM3_COMMIT"
checkout https://github.com/facebookresearch/sam2.git "$DEPS/sam2" "$SAM2_COMMIT"

echo "== 5) dependency lists from pyproject.toml"
python -c 'import sys, tomllib; p = tomllib.load(open(sys.argv[1], "rb"))["project"]; print("\n".join(p["dependencies"] + p["optional-dependencies"]["gpu"] + p["optional-dependencies"]["dev"]))' "$REPO/pyproject.toml" > "$TMPDIR/requirements-$STAMP.txt"
cat "$TMPDIR/requirements-$STAMP.txt"

echo "== 6) ONE resolver pass (SAM 3's pins, e.g. numpy<2, constrain the whole set)"
# SAM 2.1: skip the optional CUDA extension (no nvcc needed); --no-build-isolation because its
# setup.py expects torch to be importable at build time.
SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-build-isolation \
  -r "$TMPDIR/requirements-$STAMP.txt" -e "$DEPS/sam3" -e "$DEPS/sam2" \
  opencv-python-headless scikit-image matplotlib einops accelerate
pip check || echo "WARNING: pip check reported conflicts (see above)"

echo "== 7) import check (no GPU on this node: cuda False is expected here)"
python "$REPO/container/check_env.py" | tee "$PC_ROOT/env-check-$STAMP.txt"
if grep -q '^FAIL' "$PC_ROOT/env-check-$STAMP.txt"; then
  echo "Some imports FAILED (see above). NOT switching $PC_ROOT/venv. Fix the problem, then rebuild with FORCE=1."
  exit 1
fi

echo "== 8) publish: $PC_ROOT/venv -> venv-$STAMP"
ln -sfn "venv-$STAMP" "$PC_ROOT/venv"
echo "done. Use it with:  source $PC_ROOT/venv/bin/activate"
echo "Next: on a GPU node run  python $REPO/container/check_env.py  and expect 'cuda available: True'."
