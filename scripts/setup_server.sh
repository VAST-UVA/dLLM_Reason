#!/bin/bash
# Server setup script. Run this once after uploading the project.
#
# Usage:
#   bash scripts/setup_server.sh                          # full setup (install + download all)
#   bash scripts/setup_server.sh --skip-downloads         # install only, skip data/model downloads
#   bash scripts/setup_server.sh --mirror https://hf-mirror.com   # use HF mirror
#
# Optional env vars:
#   HF_TOKEN        — HuggingFace access token
#   HF_MIRROR       — HuggingFace mirror URL (same as --mirror)
#   CUDA_VERSION    — PyTorch CUDA version (default: cu121)
#   DATASETS_DIR    — dataset save path (default: datasets/)
#   CHECKPOINTS_DIR — model save path (default: checkpoints/)

set -e

# ── Parse arguments ─────────────────────────────────────────
SKIP_DOWNLOADS=false
MIRROR="${HF_MIRROR:-}"
CUDA_VER="${CUDA_VERSION:-cu121}"
DATASETS_DIR="${DATASETS_DIR:-datasets}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-downloads) SKIP_DOWNLOADS=true; shift ;;
        --mirror) MIRROR="$2"; shift 2 ;;
        --cuda) CUDA_VER="$2"; shift 2 ;;
        --datasets-dir) DATASETS_DIR="$2"; shift 2 ;;
        --checkpoints-dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MIRROR_ARG=""
if [ -n "$MIRROR" ]; then
    MIRROR_ARG="--mirror $MIRROR"
    echo "Using HuggingFace mirror: $MIRROR"
fi

echo "============================================================"
echo "Setting up dLLM-Reason on server"
echo "============================================================"
echo "  CUDA version:    $CUDA_VER"
echo "  Datasets dir:    $DATASETS_DIR"
echo "  Checkpoints dir: $CHECKPOINTS_DIR"
echo "  Skip downloads:  $SKIP_DOWNLOADS"
echo "============================================================"

# ── [1/6] Python environment ───────────────────────────────────
echo ""
echo "[1/6] Checking Python..."
python --version
which python

# Create venv if conda not available
if ! command -v conda &>/dev/null; then
    echo "Creating venv..."
    python -m venv venv
    source venv/bin/activate
fi

# ── [2/6] Install dependencies ─────────────────────────────────
echo ""
echo "[2/6] Installing dependencies..."
pip install --upgrade pip

# PyTorch (adjust CUDA version via --cuda or CUDA_VERSION env var)
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VER}" --quiet

# Install project + dependencies (core + library + dev)
pip install -e ".[dev,library]" --quiet

echo "Installing additional eval dependencies..."
pip install evaluate human-eval --quiet

# ── [3/6] HuggingFace setup ────────────────────────────────────
echo ""
echo "[3/6] HuggingFace setup..."

# Cache to local disk if available
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/$(whoami)/.cache/huggingface"
    echo "Using HF cache: $HF_HOME"
fi

if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN detected."
else
    echo "Note: Set HF_TOKEN env var or run: huggingface-cli login"
fi

# ── [4/6] Download datasets ────────────────────────────────────
echo ""
if [ "$SKIP_DOWNLOADS" = false ]; then
    echo "[4/6] Downloading datasets..."
    python scripts/download_datasets.py --output_dir "$DATASETS_DIR" $MIRROR_ARG || {
        echo "Warning: some datasets failed to download. You can retry manually:"
        echo "  python scripts/download_datasets.py --output_dir $DATASETS_DIR $MIRROR_ARG"
    }
else
    echo "[4/6] Skipping dataset download (--skip-downloads)"
    echo "  Download manually: python scripts/download_datasets.py --output_dir $DATASETS_DIR $MIRROR_ARG"
fi

# ── [5/6] Download models ──────────────────────────────────────
echo ""
if [ "$SKIP_DOWNLOADS" = false ]; then
    echo "[5/6] Downloading model checkpoints..."
    python scripts/download_models.py --output_dir "$CHECKPOINTS_DIR" $MIRROR_ARG || {
        echo "Warning: some models failed to download. You can retry manually:"
        echo "  python scripts/download_models.py --output_dir $CHECKPOINTS_DIR $MIRROR_ARG"
    }
else
    echo "[5/6] Skipping model download (--skip-downloads)"
    echo "  Download manually: python scripts/download_models.py --output_dir $CHECKPOINTS_DIR $MIRROR_ARG"
fi

# ── [6/6] Verify installation ──────────────────────────────────
echo ""
echo "[6/6] Verifying installation..."
python -c "
import torch
import transformers
import datasets
import dllm_reason
from dllm_reason.graph.dag import TokenDAG
from dllm_reason.scheduler.dag_scheduler import DAGScheduler
from dllm_reason.library import LibraryConfig, DAGStore, DAGEntry

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:', torch.cuda.get_device_name(i))
print('✓ Transformers:', transformers.__version__)
print('✓ Datasets:', datasets.__version__)
print('✓ dllm_reason: OK')

# Quick TokenDAG test
dag = TokenDAG.linear_chain(10)
print('✓ TokenDAG: OK, depth =', dag.depth())
print('✓ DAG Library: OK')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Directory layout:"
echo "  checkpoints/    — model weights"
echo "  datasets/       — evaluation datasets"
echo ""
echo "Next steps:"
echo "  1. (Optional) Download only specific items:"
echo "     python scripts/download_models.py --models llada-instruct"
echo "     python scripts/download_datasets.py --datasets gsm8k mmlu"
echo "  2. Edit scripts/run_eval.sh to adjust NUM_SAMPLES, NUM_STEPS etc."
echo "  3. Run: bash scripts/run_eval.sh"
echo "  4. Or run single: bash scripts/run_single_benchmark.sh mmlu cot"
echo "============================================================"
