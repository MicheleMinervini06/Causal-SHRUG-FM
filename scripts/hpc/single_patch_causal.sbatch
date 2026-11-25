#!/bin/bash -l
#SBATCH --job-name=shrug-e06
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/e06-%A_%a.out
#SBATCH --error=logs/e06-%A_%a.err

set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
git rev-parse HEAD
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    root="${E06_ROOT:-results/causal_e06/array-${SLURM_ARRAY_JOB_ID}}"
    set -- --start "$((20 * SLURM_ARRAY_TASK_ID))" --repeats 20 \
        --output "$root/shard-$SLURM_ARRAY_TASK_ID" "$@"
fi
srun uv run --no-sync python -u -B -m experiments.single_patch_causal "$@"
