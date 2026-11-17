#!/bin/bash -l
#SBATCH --job-name=shrug-ue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia&A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/embedding-uncertainty-%j.out
#SBATCH --error=logs/embedding-uncertainty-%j.err

# Requires a completed reference download and uv sync --locked --extra encoder-cuda.
# sbatch --dependency=afterok:DOWNLOAD_JOB_ID scripts/hpc/compute_embedding_uncertainty.sbatch [locations]
set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
LOCATIONS="${1:-5000}"
if [[ $# -gt 1 || ! "$LOCATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: sbatch $0 [positive_locations]" >&2
    exit 1
fi
REFERENCE="data/ssl4eo-s12/reference-prefix-n${LOCATIONS}"
# Preserve caches produced by the earlier fixed-native-grid reader.
EMBEDDINGS="${REFERENCE}/embeddings-native-v2"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "Embedding uncertainty job ${SLURM_JOB_ID}; node $(hostname); reference ${REFERENCE}"
git rev-parse HEAD
date -u
srun uv run --no-sync --extra encoder-cuda python -u -B scripts/cache_pretraining_embeddings.py \
    --reference "$REFERENCE" --output "$EMBEDDINGS" \
    --device cuda --threads "${SLURM_CPUS_PER_TASK}" --batch-size 32
srun uv run --no-sync python -u -B scripts/compute_embedding_uncertainty.py \
    --reference "${EMBEDDINGS}/pretraining_cls.npz" \
    --output "results/embedding_uncertainty/prefix-n${LOCATIONS}-job-${SLURM_JOB_ID}" \
    --threads "${SLURM_CPUS_PER_TASK}"
date -u
