#!/bin/bash -l
#SBATCH --job-name=shrug-head-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia&A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/segmentation-full-%j.out
#SBATCH --error=logs/segmentation-full-%j.err

# Submit from the project root after: mkdir -p logs
# Uses the existing encoder-cuda .venv from the successful five-epoch check.
# Override --time at submission if the measured epoch times need a larger limit.
set -eo pipefail
module load 2025 uv
module load slurm
set -u
cd "${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
if [[ ! -f scripts/train_segmentation_head.py || ! -x .venv/bin/python ]]; then
    echo "Missing project or Linux .venv. Prepare encoder-cuda on a compute node before submission." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
run_dir="models/segmentation/core_decoder_seed-0-full-job-${SLURM_JOB_ID}"
echo "Job ${SLURM_JOB_ID}; node $(hostname); project ${PWD}; output ${run_dir}"
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Repeat the established CUDA forward/backward preflight. No encoder execution.
uv run --no-sync --extra encoder-cuda python -B - <<'PY'
import torch
from scripts.segmentation_common import configure_runtime, native_logits
assert torch.version.cuda is not None, "CPU-only torch: prepare encoder-cuda extra first"
settings = configure_runtime(0, "cuda", 4)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "runtime:", settings, flush=True)
values = torch.randn(1, 2, 3, 3, device="cuda", requires_grad=True)
target = torch.zeros(1, 7, 9, dtype=torch.long, device="cuda")
loss = torch.nn.functional.cross_entropy(native_logits(values, target.shape[-2:]), target)
loss.backward()
assert torch.isfinite(values.grad).all().item(), "Nonfinite CUDA gradients"
print("CUDA forward/backward preflight passed", flush=True)
PY

# Fresh initialization, full train/validation splits, and best-validation selection.
# The five-epoch checkpoint has no optimizer state and is not resumed here.
srun uv run --no-sync --extra encoder-cuda python -B scripts/train_segmentation_head.py \
    --device cuda --epochs 50 --seed 0 --batch-size 1 \
    --learning-rate 5e-5 --weight-decay 5e-6 --warmup-epochs 3 --patience 5 \
    --threads "${SLURM_CPUS_PER_TASK}" --output "${run_dir}"
echo "Full training finished (50-epoch cap; validation early stopping): ${run_dir}"
date -u
