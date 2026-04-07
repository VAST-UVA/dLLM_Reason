#!/bin/bash
# Linear (left-to-right) unmasking.
# Unmasks positions sequentially from left to right, mimicking autoregressive order.
# Usage: bash scripts/runs/linear.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags linear \
    --output_dir "results/linear_$(date +%Y%m%d_%H%M%S)" \
    "$@"
