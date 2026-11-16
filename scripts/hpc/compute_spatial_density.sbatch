#!/bin/bash -l
#SBATCH --job-name=shrug-density
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/spatial-density-%j.out
#SBATCH --error=logs/spatial-density-%j.err

# Submit from the project root: sbatch scripts/hpc/compute_spatial_density.sbatch [bandwidth_km]
# The existing .venv uses a skylake_avx512 Python from the A100 nodes: CPU gen2
# matches that architecture. Download/verify coordinates here before computing.
# Default 100 km. No GPU, encoder, decoder, label reading, or dependency installs.
set -eo pipefail
module purge
module load 2025 uv
module load slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
BANDWIDTH_KM="${1:-100}"
if [[ $# -gt 1 || ! "$BANDWIDTH_KM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Usage: sbatch $0 [positive_bandwidth_km]" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing project .venv. Prepare the environment on a matching Skylake compute node." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "Spatial-density job ${SLURM_JOB_ID}; node $(hostname); bandwidth ${BANDWIDTH_KM} km; project ${PWD}"
echo "Project Python: $(readlink -f .venv/bin/python)"
echo "Module Python: ${EBROOTPYTHON}"
git rev-parse HEAD
date -u
srun uv run --no-sync python -u -B scripts/compute_spatial_density.py --download-only
srun uv run --no-sync python -B scripts/compute_spatial_density.py \
    --bandwidth-km "${BANDWIDTH_KM}" \
    --output "results/input_uncertainty/spatial_density/h-${BANDWIDTH_KM}km-job-${SLURM_JOB_ID}"
date -u
