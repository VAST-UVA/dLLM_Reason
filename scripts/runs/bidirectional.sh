#!/bin/bash
# Bidirectional DAG unmasking.
# Processes the sequence from both ends toward the center across 4 segments.
# Usage: bash scripts/runs/bidirectional.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags bidirectional \
    --output_dir "results/bidirectional_$(date +%Y%m%d_%H%M%S)" \
    "$@"
