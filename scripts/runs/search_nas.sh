#!/bin/bash
# NAS-style DAG architecture search.
# Two modes: supernet (DARTS-like) or controller (ENAS-like).
#
# Usage:
#   bash scripts/runs/search_nas.sh                          # default: supernet
#   bash scripts/runs/search_nas.sh --nas_mode controller    # ENAS-style
#   bash scripts/runs/search_nas.sh --budget 300 --span_size 32
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/search_dag.py" \
    --method nas \
    --budget 200 \
    --output_dir "results/search_nas_$(date +%Y%m%d_%H%M%S)" \
    "$@"
