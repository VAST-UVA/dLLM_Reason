#!/bin/bash
# DAG Search: RL Policy search.
# Learns a policy network (Transformer) to construct DAGs edge-by-edge via REINFORCE.
#
# Usage:
#   bash scripts/runs/search_rl.sh --checkpoint checkpoints/llada-instruct
#   bash scripts/runs/search_rl.sh --checkpoint checkpoints/llada-instruct --budget 500
#
# Key parameters (pass as CLI args to override):
#   --model        llada|mdlm|sedd|d3pm       (default: llada)
#   --checkpoint   path to model checkpoint    (REQUIRED)
#   --dataset      gsm8k|math|arc|prontoqa     (default: gsm8k)
#   --budget       number of DAG evaluations    (default: 300)
#   --fitness      accuracy|perplexity|combined (default: accuracy)
#   --seq_len      generation length            (default: 256)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/search_dag.py" \
    --method rl_policy \
    --model llada \
    --dataset gsm8k \
    --budget 300 \
    --fitness accuracy \
    --fitness_samples 50 \
    --seq_len 256 \
    --output_dir "results/search_rl_$(date +%Y%m%d_%H%M%S)" \
    "$@"
