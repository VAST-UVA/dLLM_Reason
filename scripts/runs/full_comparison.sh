#!/bin/bash
# Full comparison: all strategies × selected benchmarks.
# Generates results/ directory with summary.json and per-run results.
# After completion, run: python scripts/generate_latex_table.py results/<dir>/summary.json
#
# Usage:
#   bash scripts/runs/full_comparison.sh
#   bash scripts/runs/full_comparison.sh --num_samples 100
#   bash scripts/runs/full_comparison.sh --benchmarks gsm8k math mbpp humaneval
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/full_comparison_${TIMESTAMP}"

exec python "$ROOT/scripts/eval_dags.py" \
    --dags confidence random entropy semi_ar maskgit_cosine critical_token_first curriculum linear cot skeleton bidirectional answer_first \
    --benchmarks gsm8k math mbpp humaneval arc mmlu \
    --save_outputs \
    --output_dir "$OUTPUT_DIR" \
    "$@"
