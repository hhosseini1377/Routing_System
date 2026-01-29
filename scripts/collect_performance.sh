#!/bin/bash

# Load required modules (needed for Python 3.11.3 shared libraries)
module load GCCcore/11.3.0
module load Python/3.11.3

source ./env/bin/activate

# Change the cache directory for huggingface
    
export TRITON_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/triton
export XDG_CONFIG_HOME=/data/gpfs/projects/punim2662/.config
export VLLM_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/vllm
export VLLM_CACHE_ROOT=/data/gpfs/projects/punim2662/.cache/vllm
export TORCH_HOME=/data/gpfs/projects/punim2662/.cache/torch/
export TORCHINDUCTOR_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/torch/inductor
export CUDA_CACHE_PATH=/data/gpfs/projects/punim2662/.cache/nvidia/
export HF_HOME=/data/gpfs/projects/punim2662/.cache/huggingface

python -m collect_performance_data \
    --model-name "Qwen/Qwen3-8B" \
    --output performance_data.json \
    --memory-range 0.2 0.8 \
    --memory-steps 6 \
    --thread-range 10 90 \
    --thread-steps 8 \
    --load-range 1.0 50.0 \
    --load-steps 5 \
    --warmup-duration 10 \
    --test-duration 60