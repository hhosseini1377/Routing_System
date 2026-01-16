#!/usr/bin/env bash
set -euo pipefail

MPS_PIPE_DIR="${MPS_PIPE_DIR:-/tmp/nvidia-mps}"
MPS_LOG_DIR="${MPS_LOG_DIR:-/tmp/nvidia-log}"

mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"

export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"

if ! pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
  nvidia-cuda-mps-control -d
  echo "Started NVIDIA MPS control daemon."
else
  echo "NVIDIA MPS control daemon already running."
fi
