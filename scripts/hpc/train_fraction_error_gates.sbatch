#!/bin/bash -l
#SBATCH --job-name=shrug-e03-fit
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/fraction-error-gates-%j.out
#SBATCH --error=logs/fraction-error-gates-%j.err

# E03 phase 1 only; forward validation-training CLI options unchanged.
set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "E03 validation training job ${SLURM_JOB_ID}; node $(hostname)"
git rev-parse HEAD
date -u
srun uv run --no-sync python -u -B scripts/train_fraction_error_gates.py "$@"
date -u
