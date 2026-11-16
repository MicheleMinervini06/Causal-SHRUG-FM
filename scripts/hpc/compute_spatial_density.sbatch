#!/bin/bash -l
#SBATCH --job-name=shrug-density
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/spatial-density-%j.out
#SBATCH --error=logs/spatial-density-%j.err

# Prepare the public coordinate file once, using the existing project .venv:
#   uv run --no-sync python -B scripts/compute_spatial_density.py --download-only
# Submit from the project root: sbatch scripts/hpc/compute_spatial_density.sbatch [bandwidth_km]
# Default 100 km. No GPU, encoder, decoder, label reading, or dependency installs.
set -eo pipefail
module load 2025 uv
module load slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
BANDWIDTH_KM="${1:-100}"
if [[ $# -gt 1 || ! "$BANDWIDTH_KM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Usage: sbatch $0 [positive_bandwidth_km]" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python || ! -f data/ssl4eo-s12/pretraining_coordinates.csv ]]; then
    echo "Prepare the Linux .venv and run compute_spatial_density.py --download-only before submission." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "Spatial-density job ${SLURM_JOB_ID}; bandwidth ${BANDWIDTH_KM} km; project ${PWD}"
git rev-parse HEAD
date -u
srun uv run --no-sync python -B scripts/compute_spatial_density.py \
    --bandwidth-km "${BANDWIDTH_KM}" \
    --output "results/input_uncertainty/spatial_density/h-${BANDWIDTH_KM}km-job-${SLURM_JOB_ID}"
date -u
