#!/usr/bin/env bash
set -euo pipefail

# ---- Cluster env (same style as your other scripts) ----
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

# ---- User parameters ----
ROOT="${ROOT:-/data/gpfs/projects/punim2662/routing_system}"

NUM_GPUS="${NUM_GPUS:-4}"
LAMBDA_GLOBAL="${LAMBDA_GLOBAL:-70.0}"
TAU="${TAU:-500}"

# Output pickle
OUTPUT="${OUTPUT:-resource_allocation/brute_force_result_10.pkl}"

# Optional: subsample prompts from the scores matrix before optimizing.
# If empty, do not subsample (brute_force_setup --sample-frac remains None).
# Default subsampling fraction (faster dual-prices / brute-force runs).
SAMPLE_FRAC="${SAMPLE_FRAC:-0.25}"

# Metric for latency curves
METRIC="${METRIC:-p95_ttft}"   # one of: tpot, avg_latency_ms, p95_ttft, p95_topt

# ---- Search space (can widen/narrow) ----
TP_OPTIONS="${TP_OPTIONS:-1 2 4}"              # --tp-options
THREAD_OPTIONS="${THREAD_OPTIONS:-10 20 30 40 50 60 70 80 90 100}"  # --thread-options
MEMORY_SCALE_OPTIONS="${MEMORY_SCALE_OPTIONS:-1.0}"                   # --memory-scale-options

# ---- Feasibility / SLO tolerance ----
SLACK_TOL="${SLACK_TOL:-0.02}"
MIN_THREAD_SUM_RATIO="${MIN_THREAD_SUM_RATIO:-0.0}"

# ---- optimize_fractions hyperparams ----
BETA_INIT="${BETA_INIT:-0.01}"  # --beta-init (initial beta for optimize_beta)
STEPS="${STEPS:-200}"
ETA="${ETA:-0.1}"
DUAL_MAX_ITER="${DUAL_MAX_ITER:-300}"         # --dual-max-iter: dual/prices iterations (inner score solver)
DUAL_ETA0="${DUAL_ETA0:-1e-4}"               # --dual-eta0: base step size for dual prices
DUAL_TOL="${DUAL_TOL:-1}"                   # --dual-tol: stop when |count-c| <= dual_tol
DUAL_TIE_NOISE="${DUAL_TIE_NOISE:-1e-9}"     # --dual-tie-noise: noise for deterministic tie-breaking
MOMENTUM="${MOMENTUM:-0.0}"
W_EMA_DECAY="${W_EMA_DECAY:-0.99}"
PATIENCE="${PATIENCE:-10}"
OBJ_TOL="${OBJ_TOL:-1e-4}"

# ---- optimize_beta hyperparams ----
MAX_OUTER_STEPS="${MAX_OUTER_STEPS:-20}"
ETA_BETA="${ETA_BETA:-0.01}"
ETA_BETA_MIN="${ETA_BETA_MIN:-1e-4}"
ETA_BETA_DECAY="${ETA_BETA_DECAY:-0.98}"

# ---- Optional subset scenarios ----
# Set INCLUDE_SUBSET=1 to also test subset deployments
INCLUDE_SUBSET="${INCLUDE_SUBSET:-0}"  # default 0
SUBSET_SIZES="${SUBSET_SIZES:-2 3}"  # --subset-sizes

cd "$ROOT"

echo "Running brute-force setup search..."
echo "  NUM_GPUS=$NUM_GPUS"
echo "  LAMBDA_GLOBAL=$LAMBDA_GLOBAL"
echo "  TAU=$TAU"
echo "  METRIC=$METRIC"
echo "  OUTPUT=$OUTPUT"
if [[ -n "$SAMPLE_FRAC" ]]; then
  echo "  SAMPLE_FRAC=$SAMPLE_FRAC"
fi

CMD=(python -m resource_allocation.brute_force_setup
  --num-gpus "$NUM_GPUS"
  --lambda-global "$LAMBDA_GLOBAL"
  --tau "$TAU"
  --metric "$METRIC"
  --beta-init "$BETA_INIT"
  --slack-tol "$SLACK_TOL"
  --tp-options $TP_OPTIONS
  --thread-options $THREAD_OPTIONS
  --memory-scale-options $MEMORY_SCALE_OPTIONS
  --min-thread-sum-ratio "$MIN_THREAD_SUM_RATIO"

  --steps "$STEPS"
  --eta "$ETA"
  --dual-max-iter "$DUAL_MAX_ITER"
  --dual-eta0 "$DUAL_ETA0"
  --dual-tol "$DUAL_TOL"
  --dual-tie-noise "$DUAL_TIE_NOISE"
  --momentum "$MOMENTUM"
  --w-ema-decay "$W_EMA_DECAY"
  --patience "$PATIENCE"
  --obj-tol "$OBJ_TOL"

  --max-outer-steps "$MAX_OUTER_STEPS"
  --eta-beta "$ETA_BETA"
  --eta-beta-min "$ETA_BETA_MIN"
  --eta-beta-decay "$ETA_BETA_DECAY"

  -o "$OUTPUT"
)

if [[ -n "$SAMPLE_FRAC" ]]; then
  CMD+=(--sample-frac "$SAMPLE_FRAC")
fi

if [[ "$INCLUDE_SUBSET" == "1" ]]; then
  CMD+=(--include-subset-scenarios --subset-sizes $SUBSET_SIZES)
fi

echo "Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"