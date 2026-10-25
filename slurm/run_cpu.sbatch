#!/bin/bash
#SBATCH --job-name=pc_cpu
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Generic CPU job: runs "python <your script and args>" in the shared environment.
# Submit from the REPO ROOT:
#
#   export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
#   sbatch --account=eecs542f26s001_class slurm/run_cpu.sbatch scripts/some_script.py --config ...
#
# Override resources on the command line (--time, --mem, --cpus-per-task, --job-name, ...).
set -euo pipefail
: "${PC_ROOT:?set PC_ROOT to the shared eecs542tbn directory}"
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[[ $# -ge 1 ]] || { echo "usage: sbatch ... slurm/run_cpu.sbatch <script.py> [args...]"; exit 1; }

# shellcheck disable=SC1091
source "$PC_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO/src" PC_PATHS=greatlakes
export HF_HOME="${HF_HOME:-$PC_ROOT/hf_cache}"
cd "$REPO"

echo "host: $(hostname)   job: ${SLURM_JOB_ID:-local}   commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "running: python $*"
python "$@"
