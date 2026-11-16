#!/bin/bash -l
#SBATCH --job-name=shrug-task-uq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia&A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/segmentation-uncertainty-%j.out
#SBATCH --error=logs/segmentation-uncertainty-%j.err

# After both training tasks finish, submit from the project root:
#   sbatch scripts/hpc/predict_segmentation_ensemble.sbatch TRAINING_ARRAY_ID
# Defaults to validation. Explicit later evaluation, for example:
#   sbatch scripts/hpc/predict_segmentation_ensemble.sbatch TRAINING_ARRAY_ID test bolivia_holdout
set -eo pipefail
module load 2025 uv
module load slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: sbatch $0 TRAINING_ARRAY_ID [validation|train|test|bolivia_holdout ...]" >&2
    exit 1
fi
TRAINING_ARRAY_ID="$1"
shift
SPLITS=("$@")
if [[ ${#SPLITS[@]} -eq 0 ]]; then SPLITS=(validation); fi
CHECKPOINTS=(
    "models/segmentation/config_comparison/array-44148765/batch8_lr5e-5/best_head.pt"
    "models/segmentation/ensemble/array-${TRAINING_ARRAY_ID}/batch8_lr5e-5_seed-1/best_head.pt"
    "models/segmentation/ensemble/array-${TRAINING_ARRAY_ID}/batch8_lr5e-5_seed-2/best_head.pt"
)
# A best checkpoint exists during training too; require the end-of-training summary.
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$CHECKPOINT" || ! -f "${CHECKPOINT%/*}/training_summary.json" ]]; then
        echo "Missing completed training output for ${CHECKPOINT}; wait for all members to finish." >&2
        exit 1
    fi
done
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "Task uncertainty job ${SLURM_JOB_ID}; training array ${TRAINING_ARRAY_ID}; splits ${SPLITS[*]}"
git rev-parse HEAD
date -u
srun uv run --no-sync --extra encoder-cuda python -B scripts/predict_segmentation_ensemble.py \
    --checkpoints "${CHECKPOINTS[@]}" --splits "${SPLITS[@]}" \
    --output "results/segmentation_uncertainty/array-${TRAINING_ARRAY_ID}/job-${SLURM_JOB_ID}" \
    --device cuda --threads "${SLURM_CPUS_PER_TASK}"
date -u
