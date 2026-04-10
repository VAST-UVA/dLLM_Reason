"""Download datasets to local directory.

Downloads datasets from HuggingFace Hub to ``datasets/<local_name>/<split>/``
using ``save_to_disk()`` format.  After downloading, all loading code
(reasoning_datasets.py, benchmarks.py, etc.) will automatically use the
local copy — no code changes needed.

Usage:
    # Download all registered datasets
    python scripts/download_datasets.py

    # Download specific dataset(s)
    python scripts/download_datasets.py --datasets gsm8k math

    # Custom output directory
    python scripts/download_datasets.py --output_dir /data/datasets

    # Use mirror (mainland China)
    python scripts/download_datasets.py --mirror https://hf-mirror.com

    # Re-download even if already present
    python scripts/download_datasets.py --force

    # List available datasets
    python scripts/download_datasets.py --list

Resource metadata is defined in dllm_reason.utils.resource_registry.
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
    parser = argparse.ArgumentParser(
        description="Download datasets from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Datasets to download. Available: {names}. Default: all.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_DATASETS_DIR),
        help="Base directory to save datasets (default: datasets/)",
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
        help="Re-download even if the split already exists on disk.",
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
    """Download a single dataset (all splits) via HuggingFace datasets."""
    from datasets import load_dataset

    save_dir = output_dir / entry.local_name
    splits_needed = []
    for split in entry.splits:
        split_dir = save_dir / split
        if not force and split_exists(split_dir):
            print(f"  [skip] {entry.local_name}/{split}  (already at {split_dir})")
        else:
            splits_needed.append(split)

    if not splits_needed:
        print(f"  [skip] {entry.local_name} — all splits present")
        return

    print(f"\n  Downloading {entry.local_name} ({entry.repo_id}) "
          f"— splits: {', '.join(splits_needed)}")

    for split in splits_needed:
        try:
            if entry.config:
                ds = load_dataset(entry.repo_id, entry.config, split=split)
            else:
                ds = load_dataset(entry.repo_id, split=split)

            split_dir = save_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(split_dir))
            print(f"    {split}: {len(ds)} examples -> {split_dir}")

        except Exception as e:
            print(f"    Warning: could not download split '{split}': {e}")
            continue

    print(f"  Done: {entry.local_name}")


def main():
    args = parse_args()

    if args.list:
        print("\nRegistered datasets:")
        print("-" * 70)
        for name, entry in DATASET_REGISTRY.items():
            local_tag = f"  local: {entry.local_path}" if entry.local_path else ""
            print(f"  {name:<12} {entry.description}")
            print(f"  {'':12} repo:    {entry.repo_id}")
            print(f"  {'':12} config:  {entry.config or '(none)'}")
            print(f"  {'':12} splits:  {', '.join(entry.splits)}")
            print(f"  {'':12} size:    {entry.size}")
            if local_tag:
                print(f"  {'':12}{local_tag}")
            print()
        return

    if args.mirror:
        setup_mirror(args.mirror)
    elif os.environ.get("HF_MIRROR"):
        setup_mirror(os.environ["HF_MIRROR"])

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = args.datasets if args.datasets else list(DATASET_REGISTRY.keys())

    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            print(f"Error: unknown dataset '{name}'. "
                  f"Available: {', '.join(DATASET_REGISTRY.keys())}")
            sys.exit(1)

    print(f"Output directory: {output_dir}")
    print(f"Datasets to download: {', '.join(dataset_names)}")

    success, failed = [], []
    for name in dataset_names:
        try:
            download_dataset(DATASET_REGISTRY[name], output_dir, token,
                             force=args.force)
            success.append(name)
        except Exception as e:
            print(f"\n  Error downloading {name}: {e}")
            failed.append(name)
            continue

    print("\n" + "=" * 60)
    print(f"Download complete! Success: {len(success)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print(f"Retry: python scripts/download_datasets.py "
              f"--datasets {' '.join(failed)}")
    print(f"\nDatasets saved to: {output_dir}")
    print()
    print("Local paths (auto-detected by resolve_dataset):")
    for name in success:
        entry = DATASET_REGISTRY[name]
        print(f"  {entry.repo_id}")
        print(f"    -> {output_dir / entry.local_name}/{{split}}/")


if __name__ == "__main__":
    main()
