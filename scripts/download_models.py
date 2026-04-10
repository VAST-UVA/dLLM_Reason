"""Download model checkpoints to local directory.

Downloads models from HuggingFace Hub to ``checkpoints/<local_name>/``.
After downloading, all loading code (LLaDAWrapper, infer_llada, etc.)
will automatically use the local checkpoint — no code changes needed.

Usage:
    # Download all registered models
    python scripts/download_models.py

    # Download specific model(s)
    python scripts/download_models.py --models llada-instruct

    # Custom output directory
    python scripts/download_models.py --output_dir /data/checkpoints

    # Use mirror (mainland China)
    python scripts/download_models.py --mirror https://hf-mirror.com

    # List available models
    python scripts/download_models.py --list

Resource metadata is defined in dllm_reason.utils.resource_registry.
"""

import argparse
import os
import sys
from pathlib import Path

from dllm_reason.utils.resource_registry import (
    MODEL_REGISTRY,
    DEFAULT_CHECKPOINTS_DIR,
)


def parse_args():
    names = ", ".join(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Download model checkpoints from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to download. Available: {names}. Default: all.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help="Base directory to save models (default: checkpoints/)",
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
    """Download a model from HuggingFace Hub via snapshot_download."""
    from huggingface_hub import snapshot_download

    print(f"\n  Downloading {repo_id} -> {local_dir}")
    print("  This may take a while for large models...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        resume_download=True,
    )
    print(f"  Done: {repo_id}")


def main():
    args = parse_args()

    if args.list:
        print("\nRegistered models:")
        print("-" * 70)
        for name, entry in MODEL_REGISTRY.items():
            local_tag = f"  local: {entry.local_path}" if entry.local_path else ""
            print(f"  {name:<20} {entry.description}")
            print(f"  {'':20} repo:  {entry.repo_id}")
            print(f"  {'':20} size:  {entry.size}")
            if local_tag:
                print(f"  {'':20}{local_tag}")
            print()
        return

    if args.mirror:
        setup_mirror(args.mirror)
    elif os.environ.get("HF_MIRROR"):
        setup_mirror(os.environ["HF_MIRROR"])

    token = args.token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = args.models if args.models else list(MODEL_REGISTRY.keys())

    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"Error: unknown model '{name}'. "
                  f"Available: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    print(f"Output directory: {output_dir}")
    print(f"Models to download: {', '.join(model_names)}")

    for name in model_names:
        entry = MODEL_REGISTRY[name]
        local_dir = output_dir / entry.local_name
        try:
            download_model(entry.repo_id, local_dir, token)
        except Exception as e:
            print(f"\n  Error downloading {name}: {e}")
            print(f"  Retry: python scripts/download_models.py --models {name}")
            continue

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Models saved to: {output_dir}")
    print()
    print("Local paths (auto-detected by resolve_model_path):")
    for name in model_names:
        entry = MODEL_REGISTRY[name]
        print(f"  {entry.repo_id}")
        print(f"    -> {output_dir / entry.local_name}")


if __name__ == "__main__":
    main()
