#!/usr/bin/env bash
set -euo pipefail

MPS_PIPE_DIR="${MPS_PIPE_DIR:-/tmp/nvidia-mps}"
MPS_LOG_DIR="${MPS_LOG_DIR:-/tmp/nvidia-log}"

mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"

export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"

if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "NVIDIA MPS control daemon is reachable."
else
  rm -f "${MPS_PIPE_DIR}/"*
  nvidia-cuda-mps-control -d
  sleep 1

  if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "Started NVIDIA MPS control daemon."
  else
    echo "ERROR: MPS daemon did not become reachable." >&2
    exit 1
  fi
fi