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

# python -m resource_allocation.train_latency_model \
#   --input performance_data_mistral_7b_final.json \
#   --metric tpot \
#   --loss huber \
#   --plot resource_allocation/latency_model_val_y_true_vs_y_pred.png \
#   --min-throughput-load-ratio 0.998

# python -m resource_allocation.train_capacity_latency_model \
#   --input performance_data_yi34b_final.json \
#   --output resource_allocation/capacity_latency_mistral.pth \
#   --metric p95_ttft \
#   --plot resource_allocation/val_y_true_vs_y_pred.png
#   --min-throughput-load-ratio 0.995

# python -m resource_allocation.plot_piecewise_latency \
#   --input performance_data_yi34b_final.json \
#   --tp 4 \
#   --threads 90 \
#   --metric p95_ttft \
#   --output plots/p95_ttft_tp4_threads9.png

# python -m resource_allocation.main \
#   --lambda-global 70.0 \
#   --beta 0.045 \
#   --tp 2 2 2 \
#   --threads 50 50 100 \
#   --metric p95_ttft \
#   --steps 200 \
#   --eta 0.1 \
#   --sample-frac 0.25 \
#   --tau 500 \
#   --momentum 0.9 \
#   --w-ema-decay 0.99 \
#   --patience 10 \
#   --obj-tol 1e-4 \

# python -m resource_allocation.main \
#   --optimize-beta \
#   --lambda-global 70.0 \
#   --beta 0.01 \
#   --tau 800 \
#   --tp 2 2 2 \
#   --threads 50 50 100 \
#   --metric p95_ttft \
#   --slack-tol 0.02 \
#   --steps 200 \
#   --eta 0.1 \
#   --sample-frac 0.25 \
#   --momentum 0.9 \
#   --w-ema-decay 0.99 \
#   --patience 10 \
#   --obj-tol 1e-4 \
#   --eta-beta-decay 0.98 \
#   --eta-beta 0.01 \
#   --eta-beta-min 1e-4

# python -m profiler.check_model_memory --model 01-ai/Yi-34B --tp 1 2 4 --util-min 0.2
# python -m profiler.check_model_memory --model mistralai/Mistral-7B-v0.1 --tp 1 2 4 --util-min 0.2
# python -m profiler.check_model_memory --model lmsys/vicuna-13b-v1.5 --tp 1 2 4 --util-min 0.2

python -m resource_allocation.resource_packing \
    --tp 2 2 2 \
    --threads 0.5 0.5 1 \
    --memory 0.5 0.5 0.9 \
    --gpus 4