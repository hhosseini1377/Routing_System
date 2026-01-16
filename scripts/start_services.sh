#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="0,1"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps 
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un) 
# nvidia-cuda-mps-control -d

# Ensure MPS is active (optional but recommended when sharing GPUs).
"${ROOT_DIR}/scripts/start_mps.sh"

# Qwen3-8B on both GPUs, prefer 80% of memory/compute.
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 \
MODEL_NAME="Qwen/Qwen3-8B" \
TENSOR_PARALLEL_SIZE=2 \
GPU_MEMORY_UTILIZATION=0.80 \
MAX_MODEL_LEN=8192 \
UVICORN_PORT=8001 \
uvicorn model_server:app --host 0.0.0.0 --port 8001 &
PID_8B=$!

# Qwen3-1.8B on both GPUs, use remaining 20%.
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
MODEL_NAME="Qwen/Qwen3-1.8B" \
TENSOR_PARALLEL_SIZE=2 \
GPU_MEMORY_UTILIZATION=0.20 \
MAX_MODEL_LEN=8192 \
uvicorn model_server:app --host 0.0.0.0 --port 8002 &
PID_1B=$!

# Router
MODEL_A_URL="http://127.0.0.1:8001" \
MODEL_B_URL="http://127.0.0.1:8002" \
uvicorn router:app --host 0.0.0.0 --port 8000 &
PID_ROUTER=$!

echo "Router on :8000, Qwen3-8B on :8001, Qwen3-1.8B on :8002"
wait "${PID_8B}" "${PID_1B}" "${PID_ROUTER}"
