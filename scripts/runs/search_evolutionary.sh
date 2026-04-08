#!/bin/bash
# DAG Search: Evolutionary search with population-based optimization.
# Uses tournament selection, crossover, and mutation to evolve DAG structures.
#
# Usage:
#   bash scripts/runs/search_evolutionary.sh --checkpoint checkpoints/llada-instruct
#   bash scripts/runs/search_evolutionary.sh --checkpoint checkpoints/llada-instruct --dataset math
#
# Key parameters (pass as CLI args to override):
#   --model            llada|mdlm|sedd|d3pm       (default: llada)
#   --checkpoint       path to model checkpoint    (REQUIRED)
#   --dataset          gsm8k|math|arc|prontoqa     (default: gsm8k)
#   --budget           number of DAG evaluations    (default: 200)
#   --population_size  population size              (default: 20)
#   --mutation_rate    mutation probability          (default: 0.3)
#   --fitness          accuracy|perplexity|combined (default: accuracy)
#   --init_dag         empty|cot|skeleton|linear    (default: cot)
#   --seq_len          generation length            (default: 256)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/search_dag.py" \
    --method evolutionary \
    --model llada \
    --dataset gsm8k \
    --budget 200 \
    --population_size 20 \
    --mutation_rate 0.3 \
    --fitness accuracy \
    --init_dag cot \
    --fitness_samples 50 \
    --seq_len 256 \
    --output_dir "results/search_evolutionary_$(date +%Y%m%d_%H%M%S)" \
    "$@"
