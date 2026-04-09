#!/bin/bash
# Entropy-based unmasking: unmask lowest-entropy (most certain) positions first.
# Similar to confidence but uses full distribution entropy instead of argmax prob.
# Usage: bash scripts/runs/entropy.sh [--num_samples 50] [...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/eval_dags.py" \
    --dags entropy \
    --output_dir "results/entropy_$(date +%Y%m%d_%H%M%S)" \
    "$@"
