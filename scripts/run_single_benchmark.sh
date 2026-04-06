#!/bin/bash
# Run a single benchmark with a specific DAG strategy.
# Usage: bash scripts/run_single_benchmark.sh <benchmark> <dag>
# Example: bash scripts/run_single_benchmark.sh mmlu cot

BENCHMARK=${1:-mmlu}
DAG=${2:-cot}
MODEL_ID=${MODEL_ID:-"GSAI-ML/LLaDA-8B-Instruct"}
NUM_SAMPLES=${NUM_SAMPLES:-100}
NUM_STEPS=${NUM_STEPS:-128}
OUTPUT_DIR=${OUTPUT_DIR:-"results/single_${BENCHMARK}_${DAG}"}

echo "Benchmark: $BENCHMARK | DAG: $DAG | Samples: $NUM_SAMPLES"

python scripts/eval_dags.py \
    --model_id "$MODEL_ID" \
    --benchmarks "$BENCHMARK" \
    --dags "$DAG" \
    --num_steps $NUM_STEPS \
    --num_samples $NUM_SAMPLES \
    --temperature 0.0 \
    --max_new_tokens 512 \
    --output_dir "$OUTPUT_DIR" \
    --resume \
    2>&1 | tee "${OUTPUT_DIR}_eval.log"
