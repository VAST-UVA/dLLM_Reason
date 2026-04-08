#!/bin/bash
# Run all DAG strategies in one shot for comparison.
# Results are written to a single output directory so summary.json covers all strategies.
# Usage: bash scripts/runs/all_strategies.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags confidence random entropy semi_ar maskgit_cosine critical_token_first curriculum linear cot skeleton bidirectional answer_first \
    --output_dir "results/all_strategies_$(date +%Y%m%d_%H%M%S)" \
    "$@"
