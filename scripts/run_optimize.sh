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

# ---- Routing optimizer (fixed beta: optimize_fractions only) ----
LAMBDA_GLOBAL="${LAMBDA_GLOBAL:-10.0}"
BETA="${BETA:-10}"
TAU="${TAU:-500}"

TP_MISTRAL="${TP_MISTRAL:-2}"
TP_VICUNA="${TP_VICUNA:-2}"
TP_YI="${TP_YI:-4}"

TH_MISTRAL="${TH_MISTRAL:-50}"
TH_VICUNA="${TH_VICUNA:-50}"
TH_YI="${TH_YI:-50}"

METRIC="${METRIC:-p95_ttft}"  # one of: tpot, avg_latency_ms, p95_ttft

# optimize_fractions hyperparams
STEPS="${STEPS:-200}"
ETA="${ETA:-0.01}"
SEED="${SEED:-0}"
MOMENTUM="${MOMENTUM:-0.0}"
W_EMA_DECAY="${W_EMA_DECAY:-0.99}"
PATIENCE="${PATIENCE:-10}"
OBJ_TOL="${OBJ_TOL:-1e-4}"

# dual-prices hyperparams (passed into score_under_fractions_dual)
DUAL_MAX_ITER="${DUAL_MAX_ITER:-300}"
DUAL_ETA0="${DUAL_ETA0:-3e-5}"
DUAL_TOL="${DUAL_TOL:-1}"
DUAL_TIE_NOISE="${DUAL_TIE_NOISE:-1e-9}"

# optional: subsample fraction of prompts from routerbench scores
SAMPLE_FRAC="${SAMPLE_FRAC:-0.25}"  # if empty, don't pass --sample-frac

CMD=(
  python -m resource_allocation.main
  --lambda-global "$LAMBDA_GLOBAL"
  --beta "$BETA"
  --tau "$TAU"
  --tp "$TP_MISTRAL" "$TP_VICUNA" "$TP_YI"
  --threads "$TH_MISTRAL" "$TH_VICUNA" "$TH_YI"
  --metric "$METRIC"
  --steps "$STEPS"
  --eta "$ETA"
  --seed "$SEED"
  --momentum "$MOMENTUM"
  --w-ema-decay "$W_EMA_DECAY"
  --patience "$PATIENCE"
  --obj-tol "$OBJ_TOL"

  --dual-max-iter "$DUAL_MAX_ITER"
  --dual-eta0 "$DUAL_ETA0"
  --dual-tol "$DUAL_TOL"
  --dual-tie-noise "$DUAL_TIE_NOISE"
)

if [[ -n "$SAMPLE_FRAC" ]]; then
  CMD+=(--sample-frac "$SAMPLE_FRAC")
fi

echo "Running optimize_fractions (fixed beta) via resource_allocation.main"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
