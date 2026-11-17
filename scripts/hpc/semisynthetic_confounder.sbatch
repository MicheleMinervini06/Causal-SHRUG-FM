#!/bin/bash -l
#SBATCH --job-name=shrug-e02
#SBATCH --partition=main
#SBATCH --constraint=gen2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/semisynthetic-confounder-%j.out
#SBATCH --error=logs/semisynthetic-confounder-%j.err

# Submit from the project root; forward Python CLI options unchanged.
set -eo pipefail
module purge
module load 2025 uv slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONUNBUFFERED=1
echo "E02 job ${SLURM_JOB_ID}; node $(hostname); Python $(readlink -f .venv/bin/python)"
git rev-parse HEAD
date -u
srun uv run --no-sync python -u -B experiments/semisynthetic_confounder.py "$@"
date -u
