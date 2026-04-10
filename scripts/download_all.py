"""Download all registered models and datasets to local directories.

One-command setup: downloads everything needed for offline usage.
After running this script, all loading code will use local data
automatically — no internet required.

Usage:
    # Download everything (models + datasets) to default dirs
    python scripts/download_all.py

    # Custom directories
    python scripts/download_all.py --checkpoints_dir /data/models \
                                   --datasets_dir /data/datasets

    # Use mirror (mainland China)
    python scripts/download_all.py --mirror https://hf-mirror.com
    # or:  HF_MIRROR=https://hf-mirror.com python scripts/download_all.py

    # Download only models or only datasets
    python scripts/download_all.py --models-only
    python scripts/download_all.py --datasets-only

    # Select specific resources
    python scripts/download_all.py --models llada-instruct \
                                   --datasets gsm8k math arc
"""

import argparse
import os
import sys
from pathlib import Path

from dllm_reason.utils.resource_registry import (
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_DATASETS_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download all models and datasets for offline usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # What to download
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to download (default: all). "
             f"Available: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Datasets to download (default: all). "
             f"Available: {', '.join(DATASET_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--models-only", action="store_true",
        help="Download models only, skip datasets.",
    )
    parser.add_argument(
        "--datasets-only", action="store_true",
        help="Download datasets only, skip models.",
    )

    # Where to save
    parser.add_argument(
        "--checkpoints_dir", type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help="Directory for model checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--datasets_dir", type=str, default=str(DEFAULT_DATASETS_DIR),
        help="Directory for datasets (default: datasets/)",
    )

    # Network
    parser.add_argument(
        "--mirror", type=str, default=None,
        help="HuggingFace mirror URL (e.g. https://hf-mirror.com)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token. Also reads HF_TOKEN env var.",
    )

    # Behavior
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if resources already exist locally.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all registered resources and exit.",
    )

    return parser.parse_args()


def setup_mirror(mirror_url: str):
    """Configure HuggingFace to use a mirror endpoint."""
    os.environ["HF_ENDPOINT"] = mirror_url
    print(f"Using mirror: {mirror_url}")


def list_all():
    """Print all registered models and datasets."""
    print("\n" + "=" * 70)
    print("REGISTERED MODELS")
    print("=" * 70)
    for name, entry in MODEL_REGISTRY.items():
        print(f"  {name:<20} {entry.repo_id:<40} {entry.size}")
    print()
    print("=" * 70)
    print("REGISTERED DATASETS")
    print("=" * 70)
    for name, entry in DATASET_REGISTRY.items():
        splits = ", ".join(entry.splits)
        print(f"  {name:<12} {entry.repo_id:<40} [{splits}]  {entry.size}")
    print()


def download_models(model_names: list[str], output_dir: Path,
                    token: str = None, force: bool = False):
    """Download selected models."""
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"Downloading {len(model_names)} model(s) -> {output_dir}")
    print(f"{'─' * 60}")

    for name in model_names:
        entry = MODEL_REGISTRY[name]
        local_dir = output_dir / entry.local_name

        # Skip if exists and not forced
        if not force and local_dir.is_dir() and any(local_dir.iterdir()):
            print(f"  [skip] {name}  (already at {local_dir})")
            continue

        print(f"\n  [{name}] {entry.repo_id} ({entry.size})")
        try:
            snapshot_download(
                repo_id=entry.repo_id,
                local_dir=str(local_dir),
                token=token,
                resume_download=True,
            )
            print(f"  [{name}] Done -> {local_dir}")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")


def download_datasets(dataset_names: list[str], output_dir: Path,
                      token: str = None, force: bool = False):
    """Download selected datasets."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"Downloading {len(dataset_names)} dataset(s) -> {output_dir}")
    print(f"{'─' * 60}")

    for name in dataset_names:
        entry = DATASET_REGISTRY[name]
        save_dir = output_dir / entry.local_name

        splits_needed = []
        for split in entry.splits:
            split_dir = save_dir / split
            if not force and (split_dir / "dataset_info.json").exists():
                print(f"  [skip] {name}/{split}")
            else:
                splits_needed.append(split)

        if not splits_needed:
            continue

        print(f"\n  [{name}] {entry.repo_id}  splits: {', '.join(splits_needed)}")

        for split in splits_needed:
            try:
                if entry.config:
                    ds = load_dataset(entry.repo_id, entry.config, split=split)
                else:
                    ds = load_dataset(entry.repo_id, split=split)

                split_dir = save_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)
                ds.save_to_disk(str(split_dir))
                print(f"    {split}: {len(ds)} examples")
            except Exception as e:
                print(f"    {split}: FAILED — {e}")


def main():
    args = parse_args()

    if args.list:
        list_all()
        return

    # Mirror setup
    mirror = args.mirror or os.environ.get("HF_MIRROR")
    if mirror:
        setup_mirror(mirror)

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    ckpt_dir = Path(args.checkpoints_dir)
    ds_dir = Path(args.datasets_dir)

    # Determine what to download
    do_models = not args.datasets_only
    do_datasets = not args.models_only

    model_names = args.models if args.models else list(MODEL_REGISTRY.keys())
    dataset_names = args.datasets if args.datasets else list(DATASET_REGISTRY.keys())

    # Validate
    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"Error: unknown model '{name}'")
            sys.exit(1)
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            print(f"Error: unknown dataset '{name}'")
            sys.exit(1)

    print("=" * 60)
    print("dLLM-Reason: Download All Resources")
    print("=" * 60)
    if do_models:
        print(f"  Models:   {', '.join(model_names)}  -> {ckpt_dir}")
    if do_datasets:
        print(f"  Datasets: {', '.join(dataset_names)}  -> {ds_dir}")

    if do_models:
        download_models(model_names, ckpt_dir, token, args.force)
    if do_datasets:
        download_datasets(dataset_names, ds_dir, token, args.force)

    print("\n" + "=" * 60)
    print("All downloads complete!")
    print()
    if do_models:
        print(f"Models at:   {ckpt_dir}")
    if do_datasets:
        print(f"Datasets at: {ds_dir}")
    print()
    print("These paths are auto-detected at runtime. No config changes needed.")


if __name__ == "__main__":
    main()
