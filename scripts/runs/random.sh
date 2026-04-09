#!/bin/bash
# Random unmasking (uniform baseline).
# Unmasks positions uniformly at random at each diffusion step.
# Usage: bash scripts/runs/random.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags random \
    --output_dir "results/random_$(date +%Y%m%d_%H%M%S)" \
    "$@"
