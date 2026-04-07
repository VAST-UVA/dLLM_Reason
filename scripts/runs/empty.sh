#!/bin/bash
# Empty / unconstrained unmasking (pure random, no DAG).
# Standard LLaDA behavior with no structural constraint.
# Usage: bash scripts/runs/empty.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags empty \
    --output_dir "results/empty_$(date +%Y%m%d_%H%M%S)" \
    "$@"
