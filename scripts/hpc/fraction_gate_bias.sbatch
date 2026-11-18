#!/bin/bash -l
#SBATCH --job-name=shrug-e04-bias
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/fraction-gate-bias-%j.out
#SBATCH --error=logs/fraction-gate-bias-%j.err

set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
git rev-parse HEAD
srun uv run --no-sync python -u -B -m experiments.fraction_gate_bias "$@"
