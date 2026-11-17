#!/bin/bash -l
#SBATCH --job-name=shrug-gate
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/segmentation-gate-%j.out
#SBATCH --error=logs/segmentation-gate-%j.err

# Forward the Python CLI: sbatch scripts/hpc/segmentation_gate.sbatch train|evaluate [options]
set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ $# -lt 1 || ( "$1" != train && "$1" != evaluate ) ]]; then
    echo "Usage: sbatch $0 train|evaluate [segmentation_gate.py options]" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "Gate job ${SLURM_JOB_ID}; node $(hostname); mode $1; Python $(readlink -f .venv/bin/python)"
git rev-parse HEAD
date -u
srun uv run --no-sync python -u -B scripts/segmentation_gate.py "$@"
date -u
