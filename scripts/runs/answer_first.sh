#!/bin/bash
# Answer-First DAG unmasking.
# Unmasks the answer region (last 20% of tokens) first, then fills in reasoning.
# Usage: bash scripts/runs/answer_first.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags answer_first \
    --output_dir "results/answer_first_$(date +%Y%m%d_%H%M%S)" \
    "$@"
