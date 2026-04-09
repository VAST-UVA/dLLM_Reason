#!/bin/bash
# DAG Search: Differentiable (NOTEARS) search.
# Parameterizes DAG edges as continuous values, uses augmented Lagrangian
# with acyclicity constraint to learn optimal structure via gradient descent.
#
# Usage:
#   bash scripts/runs/search_differentiable.sh --checkpoint checkpoints/llada-instruct
#   bash scripts/runs/search_differentiable.sh --checkpoint checkpoints/llada-instruct --dataset math
#
# Key parameters (pass as CLI args to override):
#   --model        llada|mdlm|sedd|d3pm       (default: llada)
#   --checkpoint   path to model checkpoint    (REQUIRED)
#   --dataset      gsm8k|math|arc|prontoqa     (default: gsm8k)
#   --budget       number of DAG evaluations    (default: 200)
#   --fitness      accuracy|perplexity|combined (default: combined)
#   --seq_len      generation length            (default: 256)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/search_dag.py" \
    --method differentiable \
    --model llada \
    --dataset gsm8k \
    --budget 200 \
    --fitness combined \
    --fitness_samples 50 \
    --seq_len 256 \
    --output_dir "results/search_differentiable_$(date +%Y%m%d_%H%M%S)" \
    "$@"
