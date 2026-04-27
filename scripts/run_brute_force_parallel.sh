#!/usr/bin/env bash
# Run `run_brute_force_search.sh` in parallel for each (LAMBDA_GLOBAL, TAU, NUM_GPUS) row.
# Each run uses a unique OUTPUT path under resource_allocation/brute_force_results/:
#   brute_force_${gpus}g_lam..._tau....pkl
#
# Line format:  lambda:tau:num_gpus
# Two fields (lambda:tau) are still allowed; num_gpus defaults to env NUM_GPUS (default 4).
#
# Usage:
#   bash scripts/run_brute_force_parallel.sh
#   NUM_GPUS=4 bash scripts/run_brute_force_parallel.sh   # default gpus when line omits third field
#   MAX_PARALLEL=2 bash scripts/run_brute_force_parallel.sh   # cap concurrent jobs
#   MAX_PARALLEL=0 bash scripts/run_brute_force_parallel.sh   # no cap (all at once)
#   MIN_THREAD_SUM_RATIO=0.85 bash scripts/run_brute_force_parallel.sh   # passed to run_brute_force_search.sh
#
# Override pairs (one "lambda:tau:num_gpus" per line).
# Default grid (unless LAMBDA_TAU_PAIRS is set): λ ∈ {50,60,70,80,90}, τ ∈ {100..800} step 100, 4 GPUs.
#   LAMBDA_TAU_PAIRS=$'60:400:4
#   70:500:4' bash scripts/run_brute_force_parallel.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
NUM_GPUS="${NUM_GPUS:-4}"
# Passed through to run_brute_force_search.sh → brute_force_setup --min-thread-sum-ratio
MIN_THREAD_SUM_RATIO="${MIN_THREAD_SUM_RATIO:-1}"

DEFAULT_PAIRS=""
for lam in 70 80 90 100 ; do
  for tau in 100 200 300 400 500 600 700 800; do
    DEFAULT_PAIRS+="${lam}:${tau}:8"$'\n'
  done
done
PAIRS="${LAMBDA_TAU_PAIRS:-$DEFAULT_PAIRS}"

cd "$ROOT"

echo "run_brute_force_parallel: MIN_THREAD_SUM_RATIO=$MIN_THREAD_SUM_RATIO (override per run: MIN_THREAD_SUM_RATIO=0.85 bash ...)"

RESULTS_DIR="${ROOT}/resource_allocation/brute_force_resultss"
mkdir -p "$RESULTS_DIR"

pids=()
while IFS= read -r line || [[ -n "${line:-}" ]]; do
  line="${line//[$'\t\r']/}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" =~ ^# ]] && continue

  IFS=':' read -r lam tau num_gpus_line <<<"$line"
  lam="${lam#"${lam%%[![:space:]]*}"}"
  lam="${lam%"${lam##*[![:space:]]}"}"
  tau="${tau#"${tau%%[![:space:]]*}"}"
  tau="${tau%"${tau##*[![:space:]]}"}"
  num_gpus_line="${num_gpus_line#"${num_gpus_line%%[![:space:]]*}"}"
  num_gpus_line="${num_gpus_line%"${num_gpus_line##*[![:space:]]}"}"
  if [[ -z "${num_gpus_line//[[:space:]]/}" ]]; then
    num_gpus_line="$NUM_GPUS"
  fi
  if [[ ! "$num_gpus_line" =~ ^[1-9][0-9]*$ ]]; then
    echo "run_brute_force_parallel.sh: invalid num_gpus '${num_gpus_line}' in line: $line" >&2
    exit 1
  fi

  safelam="${lam//./_}"
  safetau="${tau//./_}"
  out="${RESULTS_DIR}/brute_force_${num_gpus_line}g_lam${safelam}_tau${safetau}.pkl"

  if [[ "$MAX_PARALLEL" -gt 0 ]]; then
    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
      wait -n 2>/dev/null || wait
    done
  fi

  echo "=== start NUM_GPUS=$num_gpus_line LAMBDA_GLOBAL=$lam TAU=$tau MIN_THREAD_SUM_RATIO=$MIN_THREAD_SUM_RATIO -> $out ==="
  (
    ROOT="$ROOT" OUTPUT="$out" NUM_GPUS="$num_gpus_line" LAMBDA_GLOBAL="$lam" TAU="$tau" \
      MIN_THREAD_SUM_RATIO="$MIN_THREAD_SUM_RATIO" \
      bash "$SCRIPT_DIR/run_brute_force_search.sh"
  ) &
  pids+=("$!")
done <<<"$PAIRS"

any_fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || any_fail=1
done

echo "Done (exit ${any_fail})."
exit "$any_fail"
