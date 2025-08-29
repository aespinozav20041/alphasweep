#!/usr/bin/env bash
# Launch quant-pipeline in canary mode with minimal exposure
set -euo pipefail

export MAX_NOTIONAL_PER_ORDER=50
export MAX_TOTAL_NOTIONAL=200
export MAX_DD_DAILY=0.03
export LATENCY_THRESHOLD_MS=500

python -m quant_pipeline "$@"
