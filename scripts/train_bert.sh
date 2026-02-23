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

# python -m router_model.train_multi_head_regression \
#   --data router_model/datasets/routerbench_0shot_clean.pkl \
#   --output router_model/model_checkpoints \
#   --multi-gpu

python -m router_model.evaluate_multi_head_roc \
    --model router_model/model_checkpoints/multi_head_regression_best.pth \
    --data router_model/datasets/routerbench_0shot_clean.pkl \
    --batch-size 256