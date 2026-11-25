#!/bin/bash -l
#SBATCH --job-name=e07-gpu-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia&A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/e07-gpu-check-%j.out
#SBATCH --error=logs/e07-gpu-check-%j.err

set -eo pipefail
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the separate benchmark checkout}"
project=${1:?Pass the absolute path of the active E07 checkout}
export UV_PROJECT_ENVIRONMENT="$project/.venv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
output="$PWD/results/causal_e07/gpu-benchmark-${SLURM_JOB_ID}"
echo "E07 GPU benchmark on $(hostname), report $output/benchmark.json"
srun uv run --no-sync python -u -B -m experiments.benchmark_burn_gpu \
    "$project/results/causal_e07/verification-44431065" "$project/results/causal_e07/inputs" \
    "$project/logs/e07-verify-44431066_0.out" "$output"
