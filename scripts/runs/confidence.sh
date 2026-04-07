#!/bin/bash
# Confidence-based unmasking (LLaDA default).
# Unmasks highest-confidence tokens first at each diffusion step.
# Usage: bash scripts/runs/confidence.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags confidence \
    --output_dir "results/confidence_$(date +%Y%m%d_%H%M%S)" \
    "$@"
