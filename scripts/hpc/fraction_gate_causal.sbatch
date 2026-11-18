#!/bin/bash -l
#SBATCH --job-name=shrug-e03-test
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fraction-gate-causal-%j.out
#SBATCH --error=logs/fraction-gate-causal-%j.err

set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "E03 causal job ${SLURM_JOB_ID} on $(hostname)"
git rev-parse HEAD
date -u
srun uv run --no-sync python -u -B -m experiments.fraction_gate_causal "$@"
date -u
