#!/bin/bash
# ============================================================================
# Example Research Pipeline
# ============================================================================
#
# Three-stage pipeline: baseline eval -> DAG discovery -> DAG-aware training.
# All inference goes through the FastAPI server (serve.py); training runs locally.
#
# Prerequisites:
#   1. Start the server:  python scripts/serve.py --model_id GSAI-ML/LLaDA-8B-Instruct
#   2. Run this script:   bash scripts/runs/example_pipeline.sh
#
# ── Stages ──────────────────────────────────────────────────────────────────
#   Stage 1: Evaluate base model on benchmarks with multiple schedulers
#   Stage 2: For each prompt, find the best DAG template (unmasking order)
#   Stage 3: Fine-tune the model to internalise the best unmasking order
#
# ── Customisation ───────────────────────────────────────────────────────────
#   All arguments after the script are forwarded to run_research_pipeline.py.
#   Override any default below, e.g.:
#
#     bash scripts/runs/example_pipeline.sh --stages 1 --datasets math --num_samples 50
#     bash scripts/runs/example_pipeline.sh --stages 3 --s3_mode grpo --s3_dag_mode consensus
#
#   See: python scripts/run_research_pipeline.py --help
#   See: docs/ABLATION_SETTINGS.md
# ============================================================================
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python "$ROOT/scripts/run_research_pipeline.py" \
    --stages 1 2 3 \
    --datasets gsm8k \
    --num_samples 200 \
    --s1_schedulers confidence cot skeleton \
    --s2_strategies confidence cot skeleton bidirectional answer_first linear random \
    --s3_mode sft \
    --s3_dag_mode per_template \
    --resume \
    "$@"
