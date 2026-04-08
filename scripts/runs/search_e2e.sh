#!/bin/bash
# End-to-end DAG learning: jointly optimize DAG structure with task loss.
# Uses differentiable relaxation + NOTEARS acyclicity constraint.
#
# Usage:
#   bash scripts/runs/search_e2e.sh
#   bash scripts/runs/search_e2e.sh --budget 300 --benchmarks gsm8k
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/search_dag.py" \
    --method e2e \
    --budget 200 \
    --output_dir "results/search_e2e_$(date +%Y%m%d_%H%M%S)" \
    "$@"
