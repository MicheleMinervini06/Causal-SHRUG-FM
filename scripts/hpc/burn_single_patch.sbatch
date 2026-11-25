#!/bin/bash -l
#SBATCH --job-name=shrug-e07-verify
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/e07-verify-%A_%a.out
#SBATCH --error=logs/e07-verify-%A_%a.err

set -eo pipefail
if [[ "${1:-}" == submit ]]; then
    mkdir -p logs
    inputs=results/causal_e07/inputs
    for file in posterior.npz burnscars_test_metrics.csv burn_scars_density.csv burnscars_ncdd_k64.csv summary.csv; do
        [[ -r "$inputs/$file" ]] || { echo "Missing $inputs/$file" >&2; exit 1; }
    done
    prep=$(sbatch --parsable --output=logs/e07-prepare-%j.out --error=logs/e07-prepare-%j.err "$0" prepare)
    root="results/causal_e07/verification-$prep"
    echo "Preparation=$prep root=$root"
    job=$(sbatch --parsable --dependency="afterok:$prep" --array=0-249%16 "$0" run "$root")
    echo "Array=$job (a1=1, both calibrations; 250 tasks, 20 repetitions each, up to 16 running)"
    report=$(sbatch --parsable --dependency="afterok:$job" --output=logs/e07-verify-report-%j.out \
        --error=logs/e07-verify-report-%j.err "$0" collect "$root")
    echo "Report=$report"
    echo "tail -n 30 -F logs/e07-verify-${job}_0.out logs/e07-verify-${job}_0.err"
    exit 0
fi

module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
case "$1" in
    prepare) set -- prepare "results/causal_e07/verification-${SLURM_JOB_ID}" ;;
    run) set -- run "${2:?Missing verification root}" --start "$((20 * SLURM_ARRAY_TASK_ID))" --repeats 20 ;;
    collect) set -- collect "${2:?Missing verification root}" ;;
    *) echo "Launch with: bash scripts/hpc/burn_single_patch.sbatch submit" >&2; exit 2 ;;
esac
srun uv run --no-sync python -u -B -m experiments.burn_single_patch "$@"
