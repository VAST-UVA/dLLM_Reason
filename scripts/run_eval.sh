#!/bin/bash
# ============================================================
# LLaDA + DAG Unmasking Evaluation
# Upload this entire project to your server and run:
#   bash scripts/run_eval.sh
# ============================================================

set -e

# ── Configuration ────────────────────────────────────────────
MODEL_ID="GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_DIR="results/llada_dag_eval_$(date +%Y%m%d_%H%M%S)"
NUM_STEPS=128
BLOCK_LENGTH=32
NUM_SAMPLES=200          # Set to null to run on all samples
TEMPERATURE=0.0
CFG_SCALE=0.0
REMASKING="low_confidence"
MAX_NEW_TOKENS=128       # must be divisible by BLOCK_LENGTH
COT_STEPS=4
TORCH_DTYPE="bfloat16"

# Benchmarks to run (comment out any you don't want)
BENCHMARKS="mbpp humaneval hotpotqa mmlu"

# DAG strategies to test
DAGS="confidence empty linear cot skeleton bidirectional answer_first"

# MMLU subjects (leave empty to use default subset)
MMLU_SUBJECTS=""

# ── Environment setup ────────────────────────────────────────
echo "============================================================"
echo "Setting up environment"
echo "============================================================"

# Check CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Install package if not installed
if ! python -c "import dllm_reason" 2>/dev/null; then
    echo "Installing dllm_reason package..."
    pip install -e . --quiet
fi

# Check HuggingFace cache / login
echo "HuggingFace cache: ${HF_HOME:-~/.cache/huggingface}"

# ── Run evaluation ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "Starting evaluation"
echo "Model:      $MODEL_ID"
echo "Benchmarks: $BENCHMARKS"
echo "DAGs:       $DAGS"
echo "Steps:      $NUM_STEPS"
echo "Samples:    $NUM_SAMPLES"
echo "Output:     $OUTPUT_DIR"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Build the command
CMD="python scripts/eval_dags.py \
    --model_id $MODEL_ID \
    --benchmarks $BENCHMARKS \
    --dags $DAGS \
    --num_steps $NUM_STEPS \
    --block_length $BLOCK_LENGTH \
    --temperature $TEMPERATURE \
    --cfg_scale $CFG_SCALE \
    --remasking $REMASKING \
    --max_new_tokens $MAX_NEW_TOKENS \
    --generation_len $MAX_NEW_TOKENS \
    --cot_steps $COT_STEPS \
    --torch_dtype $TORCH_DTYPE \
    --output_dir $OUTPUT_DIR \
    --resume"

# Add num_samples if set
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# Add MMLU subjects if set
if [ -n "$MMLU_SUBJECTS" ]; then
    CMD="$CMD --mmlu_subjects $MMLU_SUBJECTS"
fi

# Run with logging
echo "Command: $CMD"
echo ""
eval $CMD 2>&1 | tee "$OUTPUT_DIR/eval.log"

echo ""
echo "============================================================"
echo "Evaluation complete. Results in: $OUTPUT_DIR"
echo "============================================================"
