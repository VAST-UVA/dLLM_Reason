#!/bin/bash
# DAG Search: Greedy edge search.
# Iteratively adds/removes edges to maximize fitness on a given dataset.
#
# Usage:
#   bash scripts/runs/search_greedy.sh --checkpoint checkpoints/llada-instruct
#   bash scripts/runs/search_greedy.sh --checkpoint checkpoints/llada-instruct --dataset math --budget 200
#
# Key parameters (pass as CLI args to override):
#   --model        llada|mdlm|sedd|d3pm     (default: llada)
#   --checkpoint   path to model checkpoint  (REQUIRED)
#   --dataset      gsm8k|math|arc|prontoqa   (default: gsm8k)
#   --budget       number of DAG evaluations  (default: 100)
#   --fitness      accuracy|perplexity|combined (default: accuracy)
#   --init_dag     empty|cot|skeleton|linear  (default: empty)
#   --seq_len      generation length          (default: 256)
#   --fitness_samples  samples per evaluation (default: 50)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/search_dag.py" \
    --method greedy \
    --model llada \
    --dataset gsm8k \
    --budget 100 \
    --fitness accuracy \
    --fitness_samples 50 \
    --seq_len 256 \
    --output_dir "results/search_greedy_$(date +%Y%m%d_%H%M%S)" \
    "$@"
