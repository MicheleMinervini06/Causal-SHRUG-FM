#!/bin/bash -l
#SBATCH --job-name=e07-landslides
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e07-landslides-%j.out
#SBATCH --error=logs/e07-landslides-%j.err

set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the E07 landslides checkout}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
git rev-parse HEAD
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    root="${E07_LANDSLIDE_ROOT:-results/causal_e07/landslides/array-${SLURM_ARRAY_JOB_ID}}"
    set -- simulate --start "$((50 * SLURM_ARRAY_TASK_ID))" --repeats 50 \
        --output "$root/shard-$SLURM_ARRAY_TASK_ID" "$@"
fi
srun uv run --no-sync python -u -B -m experiments.landslide_fraction_causal "$@"
