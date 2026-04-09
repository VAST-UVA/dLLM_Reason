#!/bin/bash
# Semi-autoregressive: process blocks left-to-right, confidence within each block.
# Bridges autoregressive and fully parallel diffusion generation.
# Usage: bash scripts/runs/semi_ar.sh [--num_samples 50] [--block_length 32] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags semi_ar \
    --output_dir "results/semi_ar_$(date +%Y%m%d_%H%M%S)" \
    "$@"
