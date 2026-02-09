#!/bin/bash
set -euo pipefail

# Load required modules (needed for Python 3.11.3 shared libraries)
module load GCCcore/11.3.0
module load Python/3.11.3

source ./env/bin/activate

# Change the cache directory for huggingface
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! "${ROOT_DIR}/scripts/start_mps.sh"; then
    echo "ERROR: Failed to start MPS" >&2
    exit 1
fi

python -m profiler.collect_performance_data \
    --model-name "Qwen/Qwen3-14B" \
    --output performance_data.json \
    --memory-range 0.6 0.6 \
    --memory-steps 1 \
    --thread-range 10 100 \
    --thread-steps 10 \
    --load-range 1 13 \
    --load-steps 10 \
    --warmup-duration 10 \
    --test-duration 60 \
    --completion-timeout-sec 60 \
    --request-timeout-sec 120 \
#    --max-num-seqs-range 256 1024 --max-num-seqs-steps 4 \
#    --max-num-batched-tokens-range 2048 8192 --max-num-batched-tokens-steps 4 \

   # Optional: set vLLM scheduler params (uncomment to use)
#   --max-num-seqs 256 \
#   --max-num-batched-tokens 4096 \
# Optional: sweep max_model_len (e.g. 1024, 2048, 4096)
#    --max-model-len-range 1024 4096 --max-model-len-steps 3 \
# Optional: sweep max_num_seqs / max_num_batched_tokens