#!/bin/bash
# Skeleton-then-Detail DAG unmasking.
# Unmasks structural (every 3rd) tokens first, then fills in the detail tokens.
# Usage: bash scripts/runs/skeleton.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags skeleton \
    --output_dir "results/skeleton_$(date +%Y%m%d_%H%M%S)" \
    "$@"
