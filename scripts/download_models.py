"""Download model checkpoints to local checkpoints/ directory.

Usage:
    # Download all models
    python scripts/download_models.py

    # Download specific model
    python scripts/download_models.py --models llada-instruct

    # Use mirror (e.g. hf-mirror.com for China)
    python scripts/download_models.py --mirror https://hf-mirror.com

    # Specify output directory
    python scripts/download_models.py --output_dir /data/checkpoints

    # List available models
    python scripts/download_models.py --list
"""

import argparse
import os
import sys
from pathlib import Path

# ── Model registry ──────────────────────────────────────────────────────────

MODELS = {
    "llada-instruct": {
        "repo_id": "GSAI-ML/LLaDA-8B-Instruct",
        "description": "LLaDA 8B Instruct — main inference model",
        "size": "~16GB (bf16)",
    },
    "llada-base": {
        "repo_id": "GSAI-ML/LLaDA-8B-Base",
        "description": "LLaDA 8B Base — for fine-tuning",
        "size": "~16GB (bf16)",
    },
}

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to download. Available: {', '.join(MODELS.keys())}. Default: all.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT),
        help="Directory to save models (default: checkpoints/)",
    )
    parser.add_argument(
        "--mirror", type=str, default=None,
        help="HuggingFace mirror URL (e.g. https://hf-mirror.com)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token. Also reads HF_TOKEN env var.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and exit.",
    )
    return parser.parse_args()


def setup_mirror(mirror_url: str):
    """Configure HuggingFace to use a mirror endpoint."""
    os.environ["HF_ENDPOINT"] = mirror_url
    print(f"Using mirror: {mirror_url}")


def download_model(repo_id: str, local_dir: Path, token: str = None):
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"\nDownloading {repo_id} -> {local_dir}")
    print("This may take a while for large models...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        resume_download=True,
    )
    print(f"Done: {repo_id}")


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable models:")
        print("-" * 60)
        for name, info in MODELS.items():
            print(f"  {name:<20} {info['description']}")
            print(f"  {'':20} Repo: {info['repo_id']}")
            print(f"  {'':20} Size: {info['size']}")
            print()
        return

    if args.mirror:
        setup_mirror(args.mirror)

    token = args.token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = args.models if args.models else list(MODELS.keys())

    # Validate names
    for name in model_names:
        if name not in MODELS:
            print(f"Error: unknown model '{name}'. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    print(f"Output directory: {output_dir}")
    print(f"Models to download: {', '.join(model_names)}")
    print()

    for name in model_names:
        info = MODELS[name]
        local_dir = output_dir / name
        try:
            download_model(info["repo_id"], local_dir, token)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            print("You can retry with: python scripts/download_models.py "
                  f"--models {name}")
            continue

    print("\n" + "=" * 60)
    print("Model download complete!")
    print(f"Models saved to: {output_dir}")
    print("\nTo use local checkpoints, set model_id in configs:")
    for name in model_names:
        print(f"  {name}: {output_dir / name}")


if __name__ == "__main__":
    main()
