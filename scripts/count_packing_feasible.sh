#!/usr/bin/env bash
set -euo pipefail

module load GCCcore/11.3.0
module load Python/3.11.3
source ./env/bin/activate

export TRITON_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/triton
export XDG_CONFIG_HOME=/data/gpfs/projects/punim2662/.config
export VLLM_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/vllm
export VLLM_CACHE_ROOT=/data/gpfs/projects/punim2662/.cache/vllm
export TORCH_HOME=/data/gpfs/projects/punim2662/.cache/torch/
export TORCHINDUCTOR_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/torch/inductor
export CUDA_CACHE_PATH=/data/gpfs/projects/punim2662/.cache/nvidia/
export HF_HOME=/data/gpfs/projects/punim2662/.cache/huggingface

ROOT="${ROOT:-/data/gpfs/projects/punim2662/routing_system}"
cd "$ROOT"

# Space-separated GPU counts, e.g. "4" or "2 4 8"
NUM_GPUS="${NUM_GPUS:-4}"
MIN_THREAD_SUM_RATIO="${MIN_THREAD_SUM_RATIO:-0.9}"
# Same defaults as brute_force_setup argparse
TP_OPTIONS="${TP_OPTIONS:-1 2 4}"
THREAD_OPTIONS="${THREAD_OPTIONS:-10 20 30 40 50 60 70 80 90 100}"
SUBSET_SIZES="${SUBSET_SIZES:-1 2 3}"
MEMORY_SCALE="${MEMORY_SCALE:-1.0}"

CMD=(
  python -m resource_allocation.count_packing_feasible
  --root "$ROOT"
  --num-gpus $NUM_GPUS
  --min-thread-sum-ratio "$MIN_THREAD_SUM_RATIO"
  --tp-options $TP_OPTIONS
  --thread-options $THREAD_OPTIONS
  --subset-sizes $SUBSET_SIZES
  --memory-scale "$MEMORY_SCALE"
)

echo "Running count_packing_feasible..."
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
