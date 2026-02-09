#!/usr/bin/env bash
# Collect routing performance at a fixed load.
# Start router + two models first: ./scripts/start_services.sh
# Then run this script (e.g. in another terminal or after backgrounding start_services).

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

module load GCCcore/11.3.0
module load Python/3.11.3
source ./env/bin/activate

python -m profiler.collect_routing_performance_data \
  --router-url "http://127.0.0.1:8000" \
  --load-rps 20 \
  --duration 60 \
  --output routing_performance.json \
  --prompts-path "datasets/lmsys_chat1m_prompts_100k_cleaned.pkl" \
  --request-timeout-sec 120 \
  --completion-timeout-sec 300 \
  "$@"
