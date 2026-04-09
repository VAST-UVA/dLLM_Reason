#!/bin/bash
# LLaDA evaluation — all parameters live in configs/eval_default.yaml.
# CLI args override the config file. Examples:
#
#   bash scripts/run_eval.sh
#   bash scripts/run_eval.sh --benchmarks mbpp --num_samples 50
#   bash scripts/run_eval.sh --dags cot skeleton --num_steps 64
#   bash scripts/run_eval.sh --verbose_errors

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Environment check ──────────────────────────────────────────────────────────
python -c "import torch; print(\"PyTorch \$(python -c 'import torch; print(torch.__version__)')\") " 2>/dev/null || true
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {torch.cuda.device_count()}')"

python -c "import dllm_reason" 2>/dev/null || {
    echo "Installing dllm_reason..."
    pip install -e "$ROOT" --quiet
}

# ── Run ────────────────────────────────────────────────────────────────────────
# --output_dir gets a timestamp suffix; all other defaults come from configs/eval_default.yaml
exec python "$ROOT/scripts/eval_dags.py" \
    --output_dir "results/eval_$(date +%Y%m%d_%H%M%S)" \
    "$@"
