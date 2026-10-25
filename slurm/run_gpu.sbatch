#!/bin/bash
#SBATCH --job-name=pc_gpu
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Generic GPU job: runs "python <your script and args>" in the shared environment.
# Submit from the REPO ROOT (so scripts/ and src/ are found):
#
#   export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
#   sbatch --account=eecs542f26s001_class slurm/run_gpu.sbatch scripts/smoke_sam3.py
#   sbatch --account=eecs542f26s001_class --time=04:00:00 --job-name=rq0 \
#          slurm/run_gpu.sbatch scripts/<your_script>.py --config configs/experiments/<x>.yaml
#
# Override resources on the command line (--time, --mem, --job-name, ...). Use run_cpu.sbatch for
# CPU-only work (metrics, perturbation generation): SPGPU hours are the scarce budget.
set -euo pipefail
: "${PC_ROOT:?set PC_ROOT to the shared eecs542tbn directory}"
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[[ $# -ge 1 ]] || { echo "usage: sbatch ... slurm/run_gpu.sbatch <script.py> [args...]"; exit 1; }

# shellcheck disable=SC1091
source "$PC_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO/src" PC_PATHS=greatlakes
export HF_HOME="${HF_HOME:-$PC_ROOT/hf_cache}"
cd "$REPO"

echo "host: $(hostname)   job: ${SLURM_JOB_ID:-local}   commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
echo "running: python $*"
python "$@"
