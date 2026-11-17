#!/bin/bash -l
#SBATCH --job-name=shrug-ue-data
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/embedding-download-%j.out
#SBATCH --error=logs/embedding-download-%j.err

# Stops after the first N location blocks, keeping all their seasonal images.
# Submit from project root: sbatch scripts/hpc/download_ssl4eo_reference.sbatch [locations]
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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "Reference download job ${SLURM_JOB_ID}; node $(hostname); project ${PWD}"
echo "Python: $(readlink -f .venv/bin/python); module Python: ${EBROOTPYTHON}"
git rev-parse HEAD
date -u
srun uv run --no-sync python -u -B scripts/download_ssl4eo_reference.py \
    --locations "$LOCATIONS" \
    --output "data/ssl4eo-s12/reference-prefix-n${LOCATIONS}"
srun uv run --no-sync python -u -B scripts/cache_pretraining_embeddings.py \
    --reference "data/ssl4eo-s12/reference-prefix-n${LOCATIONS}" --audit-only
date -u
