#!/bin/bash
# ============================================================
# LLaDA + DAG Unmasking Evaluation
# All parameters are set in the "Configuration" section below.
# Run:  bash scripts/run_eval.sh
# ============================================================

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
# Model
MODEL_ID="checkpoints/llada-instruct"
TORCH_DTYPE="bfloat16"          # bfloat16 | float16 | float32
DEVICE_MAP="auto"

# Inference
NUM_STEPS=128
BLOCK_LENGTH=32
TEMPERATURE=0.0                 # 0 = greedy argmax
CFG_SCALE=0.0                   # 0 = disabled
REMASKING="low_confidence"      # low_confidence | random
MAX_NEW_TOKENS=128              # must be divisible by BLOCK_LENGTH

# Benchmarks  (comment out entries to skip)
BENCHMARKS="mbpp humaneval"
NUM_SAMPLES=200                 # number of samples per benchmark; leave empty for all

# DAG strategies  (comment out entries to skip)
DAGS="confidence"
# DAGS="confidence empty linear cot skeleton bidirectional answer_first"
COT_STEPS=4
MMLU_SUBJECTS=""                # leave empty for default subset

# Output
OUTPUT_DIR="results/llada_eval_$(date +%Y%m%d_%H%M%S)"
RESUME=false                    # true = skip already-completed runs

# Logging
RUN_TESTS=true                  # false = skip code execution (inspect outputs only)
VERBOSE_ERRORS=false            # true = print per-sample stderr/error/timeout logs

# ── Environment check ──────────────────────────────────────────────────────────
echo "============================================================"
echo " Environment"
echo "============================================================"
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {torch.cuda.device_count()}')"

if ! python -c "import dllm_reason" 2>/dev/null; then
    echo "Installing dllm_reason..."
    pip install -e . --quiet
fi

if [ ! -d "$MODEL_ID" ]; then
    echo "[WARN] Model path '$MODEL_ID' not found. Set MODEL_ID or download first."
fi

# ── Build command ──────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

CMD="python scripts/eval_dags.py"
CMD="$CMD --model_id $MODEL_ID"
CMD="$CMD --torch_dtype $TORCH_DTYPE"
CMD="$CMD --device_map $DEVICE_MAP"
CMD="$CMD --benchmarks $BENCHMARKS"
CMD="$CMD --dags $DAGS"
CMD="$CMD --num_steps $NUM_STEPS"
CMD="$CMD --block_length $BLOCK_LENGTH"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --cfg_scale $CFG_SCALE"
CMD="$CMD --remasking $REMASKING"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --generation_len $MAX_NEW_TOKENS"
CMD="$CMD --cot_steps $COT_STEPS"
CMD="$CMD --output_dir $OUTPUT_DIR"

[ -n "$NUM_SAMPLES" ]    && CMD="$CMD --num_samples $NUM_SAMPLES"
[ -n "$MMLU_SUBJECTS" ]  && CMD="$CMD --mmlu_subjects $MMLU_SUBJECTS"
[ "$RESUME" = true ]     && CMD="$CMD --resume"
[ "$RUN_TESTS" = false ]     && CMD="$CMD --no_run_tests"
[ "$VERBOSE_ERRORS" = true ] && CMD="$CMD --verbose_errors"

# ── Run ────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Evaluation"
echo "============================================================"
echo "  Model       : $MODEL_ID"
echo "  Benchmarks  : $BENCHMARKS"
echo "  DAGs        : $DAGS"
echo "  Steps       : $NUM_STEPS  (block=$BLOCK_LENGTH)"
echo "  Gen tokens  : $MAX_NEW_TOKENS"
echo "  Temperature : $TEMPERATURE  CFG=$CFG_SCALE  remasking=$REMASKING"
echo "  Samples     : ${NUM_SAMPLES:-all}"
echo "  Output      : $OUTPUT_DIR"
echo "============================================================"
echo ""

eval $CMD 2>&1 | tee "$OUTPUT_DIR/eval.log"

echo ""
echo "============================================================"
echo " Done. Results in: $OUTPUT_DIR"
echo "============================================================"
