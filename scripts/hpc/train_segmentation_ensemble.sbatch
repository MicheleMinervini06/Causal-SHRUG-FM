#!/bin/bash -l
#SBATCH --job-name=shrug-ensemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia&A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-2%1
#SBATCH --output=logs/segmentation-ensemble-%A_%a.out
#SBATCH --error=logs/segmentation-ensemble-%A_%a.err

# Submit from the project root after committing/pushing and pulling the code:
#   mkdir -p logs
#   sbatch scripts/hpc/train_segmentation_ensemble.sbatch
# Array task ID = model seed. Reuse the existing seed-0 batch-8 checkpoint;
# train only seeds 1 and 2, each on its own GPU (one active task by default).
# To allow both tasks concurrently: sbatch --array=1-2%2 <this script>
set -eo pipefail
module load 2025 uv
module load slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
REFERENCE_DIR="models/segmentation/config_comparison/array-44148765/batch8_lr5e-5"
if [[ ! -f scripts/run_segmentation_trial.py || ! -x .venv/bin/python ]]; then
    echo "Missing trial runner or Linux .venv. Pull the code and prepare encoder-cuda before submission." >&2
    exit 1
fi
if [[ ! -f "${REFERENCE_DIR}/config.json" || ! -f "${REFERENCE_DIR}/best_head.pt" ]]; then
    echo "Missing seed-0 reference config/checkpoint in ${REFERENCE_DIR}; preserve or restore that run." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "Ensemble array ${SLURM_ARRAY_JOB_ID}, seed ${SLURM_ARRAY_TASK_ID}; node $(hostname); project ${PWD}"
echo "Reusing seed 0: ${REFERENCE_DIR}/best_head.pt (best validation checkpoint)"
git rev-parse HEAD
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun uv run --no-sync --extra encoder-cuda python -B scripts/run_segmentation_trial.py \
    --trial-id 1 --seed "${SLURM_ARRAY_TASK_ID}" --run-group "array-${SLURM_ARRAY_JOB_ID}" \
    --reference-config "${REFERENCE_DIR}/config.json" \
    --output-root models/segmentation/ensemble \
    --metadata-root results/segmentation_ensemble \
    --device cuda --threads "${SLURM_CPUS_PER_TASK}"
echo "Ensemble seed ${SLURM_ARRAY_TASK_ID} finished"
date -u
