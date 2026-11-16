#!/bin/bash -l
#SBATCH --job-name=shrug-trials
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia&A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/segmentation-trials-%A_%a.out
#SBATCH --error=logs/segmentation-trials-%A_%a.err

# Submit from project root after committing/pushing and pulling the same code:
#   mkdir -p logs
#   sbatch scripts/hpc/train_segmentation_trials.sbatch
# %1 runs at most one independent GPU trial at a time. Array IDs map to:
#   0 = reference: batch 1, lr 5e-5
#   1 = larger batch: batch 8, lr 5e-5
#   2 = lower LR: batch 1, lr 2e-5
# All: fresh initialization, seed 0, 50 epochs, patience 50, original CoreDecoder.
set -eo pipefail
module load 2025 uv
module load slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -f scripts/run_segmentation_trial.py || ! -x .venv/bin/python ]]; then
    echo "Missing trial runner or Linux .venv. Pull the code and prepare encoder-cuda before submission." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "Array ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}; node $(hostname); project ${PWD}"
git rev-parse HEAD
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun uv run --no-sync --extra encoder-cuda python -B scripts/run_segmentation_trial.py \
    --trial-id "${SLURM_ARRAY_TASK_ID}" --run-group "array-${SLURM_ARRAY_JOB_ID}" \
    --device cuda --threads "${SLURM_CPUS_PER_TASK}"
echo "Trial ${SLURM_ARRAY_TASK_ID} finished"
date -u
