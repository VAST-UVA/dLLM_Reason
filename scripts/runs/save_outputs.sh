#!/bin/bash
# ============================================================
# Save detailed per-sample outputs to JSON + Excel.
#
# What gets saved (all files land in OUTPUT_DIR):
#
#   {benchmark}_{dag}_samples.json   — per-sample records:
#       prompt, generated output, ground truth, pass/fail, etc.
#
#   {benchmark}_{dag}_samples.xlsx   — same data in spreadsheet form,
#       one row per sample, easy to open in Excel / Sheets.
#
#   {benchmark}_{dag}_trajectory.json  (only if RECORD_TRAJECTORY=true)
#       — per-step unmasking states decoded to text, e.g.:
#           step 0:  "<mask> <mask> <mask> ..."
#           step 10: "def add <mask> <mask> ..."
#           step 127: "def add(a, b): return a + b"
#       WARNING: large file (128 steps × N samples). Disable for big runs.
#
# What each flag controls:
#   --save_outputs         master switch (required to write any files)
#   --no_save_qa           omit prompt + generated answer from output
#   --no_save_ground_truth omit reference/canonical answers from output
#   --record_trajectory    also record per-step unmasking states
#   --output_formats       json xlsx  (default: both)
#
# Usage:
#   bash scripts/runs/save_outputs.sh
#   bash scripts/runs/save_outputs.sh --benchmarks mbpp --num_samples 50
#   bash scripts/runs/save_outputs.sh --record_trajectory --num_samples 10
# ============================================================

set -e

# ── What to save ──────────────────────────────────────────────────────────────
SAVE_QA=true             # save prompt + generated answer
SAVE_GROUND_TRUTH=true   # save reference answers
RECORD_TRAJECTORY=false  # per-step unmasking states (slow/large — keep false for big runs)
OUTPUT_FORMATS="json xlsx"  # file formats; space-separated

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CMD="python $ROOT/scripts/eval_dags.py"
CMD="$CMD --save_outputs"
CMD="$CMD --output_formats $OUTPUT_FORMATS"
CMD="$CMD --output_dir results/save_outputs_$(date +%Y%m%d_%H%M%S)"

[ "$SAVE_QA" = false ]           && CMD="$CMD --no_save_qa"
[ "$SAVE_GROUND_TRUTH" = false ] && CMD="$CMD --no_save_ground_truth"
[ "$RECORD_TRAJECTORY" = true ]  && CMD="$CMD --record_trajectory"

# Pass any extra CLI args through (e.g. --benchmarks mbpp --num_samples 50)
exec $CMD "$@"
