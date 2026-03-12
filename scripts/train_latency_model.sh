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

# # With options
python -m resource_allocation.train_latency_model \
  --input performance_data_mistral_7b_final.json \
  --output resource_allocation/latency_model_mistral.pth \
  --epochs 500 \
  --lr 1e-3 \
  --n-folds 5 \
  --patience 50 \
  --metric p95_ttft \
  --loss  \
  --huber-delta 1.0 \
  --min-throughput-load-ratio 0.99

# python -m resource_allocation.train_stage_ttft_model -i performance_data_vicuna_13b_final.json
# python -m resource_allocation.train_queue_inspired_model -i performance_data_vicuna_13b_final.json