"""Download datasets to local datasets/ directory.

Usage:
    # Download all datasets
    python scripts/download_datasets.py

    # Download specific datasets
    python scripts/download_datasets.py --datasets gsm8k mmlu

    # Use mirror
    python scripts/download_datasets.py --mirror https://hf-mirror.com

    # Specify output directory
    python scripts/download_datasets.py --output_dir /data/datasets

    # List available datasets
    python scripts/download_datasets.py --list

Resource metadata (repo_id, config, splits, etc.) is defined once in
``dllm_reason.utils.resource_registry.DATASET_REGISTRY``.
"""

import argparse
import os
import sys
from pathlib import Path

from dllm_reason.utils.resource_registry import (
    DATASET_REGISTRY,
    DatasetEntry,
    DEFAULT_DATASETS_DIR,
)


def parse_args():
    names = ", ".join(DATASET_REGISTRY.keys())
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Datasets to download. Available: {names}. Default: all.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_DATASETS_DIR),
        help="Directory to save datasets (default: datasets/)",
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
        help="List available datasets and exit.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the dataset split already exists on disk.",
    )
    return parser.parse_args()


def setup_mirror(mirror_url: str):
    """Configure HuggingFace to use a mirror endpoint."""
    os.environ["HF_ENDPOINT"] = mirror_url
    print(f"Using mirror: {mirror_url}")


def split_exists(split_dir: Path) -> bool:
    """Return True if a split was already saved by save_to_disk()."""
    return (split_dir / "dataset_info.json").exists()


def download_dataset(entry: DatasetEntry, output_dir: Path,
                     token: str = None, force: bool = False):
    """Download a dataset via HuggingFace datasets library.

    Skips any split whose target directory already contains a valid
    save_to_disk() snapshot (i.e. has dataset_info.json).
    Pass force=True to re-download regardless.
    """
    from datasets import load_dataset

    save_dir = output_dir / entry.local_name
    splits_needed = []
    for split in entry.splits:
        split_dir = save_dir / split
        if not force and split_exists(split_dir):
            print(f"  [skip] {entry.local_name}/{split} already exists at {split_dir}")
        else:
            splits_needed.append(split)

    if not splits_needed:
        print(f"[skip] {entry.local_name} — all splits already downloaded")
        return

    print(f"\nDownloading {entry.local_name} ({entry.repo_id}) "
          f"— splits: {', '.join(splits_needed)} ...")

    for split in splits_needed:
        try:
            kwargs = {"split": split}
            if entry.config:
                ds = load_dataset(entry.repo_id, entry.config, **kwargs)
            else:
                ds = load_dataset(entry.repo_id, **kwargs)

            split_dir = save_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(split_dir))
            print(f"  {split}: {len(ds)} examples -> {split_dir}")

        except Exception as e:
            print(f"  Warning: could not download split '{split}': {e}")
            continue

    print(f"Done: {entry.local_name}")


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable datasets:")
        print("-" * 70)
        for name, entry in DATASET_REGISTRY.items():
            print(f"  {name:<12} {entry.description}")
            print(f"  {'':12} Repo: {entry.repo_id}, Size: {entry.size}")
            print(f"  {'':12} Splits: {', '.join(entry.splits)}")
            print()
        return

    if args.mirror:
        setup_mirror(args.mirror)

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = args.datasets if args.datasets else list(DATASET_REGISTRY.keys())

    # Validate names
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            print(f"Error: unknown dataset '{name}'. "
                  f"Available: {', '.join(DATASET_REGISTRY.keys())}")
            sys.exit(1)

    print(f"Output directory: {output_dir}")
    print(f"Datasets to download: {', '.join(dataset_names)}")
    print()

    success, failed = [], []
    for name in dataset_names:
        try:
            download_dataset(DATASET_REGISTRY[name], output_dir, token,
                             force=args.force)
            success.append(name)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            failed.append(name)
            continue

    print("\n" + "=" * 60)
    print(f"Download complete! Success: {len(success)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print("Retry with: python scripts/download_datasets.py "
              f"--datasets {' '.join(failed)}")
    print(f"\nDatasets saved to: {output_dir}")


if __name__ == "__main__":
    main()
