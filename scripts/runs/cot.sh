#!/bin/bash
# Chain-of-Thought DAG unmasking.
# Partitions the generation into reasoning segments; each segment depends on all prior ones.
# --cot_steps controls the number of segments (default: 4 from config).
# Usage: bash scripts/runs/cot.sh [--cot_steps 6] [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags cot \
    --output_dir "results/cot_$(date +%Y%m%d_%H%M%S)" \
    "$@"
