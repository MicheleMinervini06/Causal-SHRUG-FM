#!/bin/bash -l
#SBATCH --job-name=shrug-head-5ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia&A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/segmentation-5ep-%j.out
#SBATCH --error=logs/segmentation-5ep-%j.err

# Submit from the project root after: mkdir -p logs
# Build .venv once on an interactive compute node with:
#   module load 2025 uv
#   uv sync --locked --extra encoder-cuda
set -eo pipefail
module load 2025 uv
set -u
cd "${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
if [[ ! -f scripts/train_segmentation_head.py || ! -x .venv/bin/python ]]; then
    echo "Missing project or Linux .venv. Prepare it with uv sync --locked --extra encoder-cuda on a compute node." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
run_dir="models/segmentation/core_decoder_seed-0-5epochs-job-${SLURM_JOB_ID}"
echo "Job ${SLURM_JOB_ID}; node $(hostname); project ${PWD}; output ${run_dir}"
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Verify the prepared CUDA runtime and the formerly failing interpolation
# backward before committing to all five epochs. No encoder execution.
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

# Normal full train/validation splits; --debug would override epochs to one.
srun uv run --no-sync --extra encoder-cuda python -B scripts/train_segmentation_head.py \
    --device cuda --epochs 5 --seed 0 --batch-size 1 \
    --log-every 1 \
    --threads "${SLURM_CPUS_PER_TASK}" --patience 5 --output "${run_dir}"
echo "Five-epoch run completed: ${run_dir}"
date -u
