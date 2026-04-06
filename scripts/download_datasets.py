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
"""

import argparse
import os
import sys
from pathlib import Path

# ── Dataset registry ────────────────────────────────────────────────────────

DATASETS = {
    # Training / reasoning evaluation
    "gsm8k": {
        "repo_id": "openai/gsm8k",
        "config": "main",
        "splits": ["train", "test"],
        "description": "Grade School Math 8K — arithmetic reasoning",
        "size": "~7MB",
    },
    "math": {
        "repo_id": "hendrycks/competition_math",
        "config": None,
        "splits": ["train", "test"],
        "description": "MATH — competition-level math problems",
        "size": "~50MB",
    },
    "arc": {
        "repo_id": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "splits": ["train", "test", "validation"],
        "description": "ARC-Challenge — science reasoning (multiple choice)",
        "size": "~1MB",
    },
    "prontoqa": {
        "repo_id": "renma/ProntoQA",
        "config": None,
        "splits": ["train"],
        "description": "ProntoQA — logical reasoning",
        "size": "~2MB",
    },
    # Benchmark evaluation
    "mbpp": {
        "repo_id": "google-research-datasets/mbpp",
        "config": "sanitized",
        "splits": ["test", "train", "prompt"],
        "description": "MBPP — basic Python programming problems",
        "size": "~2MB",
    },
    "humaneval": {
        "repo_id": "openai/openai_humaneval",
        "config": None,
        "splits": ["test"],
        "description": "HumanEval — Python code generation",
        "size": "~1MB",
    },
    "hotpotqa": {
        "repo_id": "hotpot_qa",
        "config": "distractor",
        "splits": ["train", "validation"],
        "description": "HotpotQA — multi-hop question answering",
        "size": "~600MB",
    },
    "mmlu": {
        "repo_id": "cais/mmlu",
        "config": "all",
        "splits": ["test", "validation"],
        "description": "MMLU — massive multitask language understanding",
        "size": "~4MB",
    },
}

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "datasets"


def parse_args():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Datasets to download. Available: {', '.join(DATASETS.keys())}. Default: all.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT),
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


def download_dataset(name: str, info: dict, output_dir: Path,
                     token: str = None, force: bool = False):
    """Download a dataset via HuggingFace datasets library.

    Skips any split whose target directory already contains a valid
    save_to_disk() snapshot (i.e. has dataset_info.json).
    Pass force=True to re-download regardless.
    """
    from datasets import load_dataset

    save_dir = output_dir / name
    splits_needed = []
    for split in info["splits"]:
        split_dir = save_dir / split
        if not force and split_exists(split_dir):
            print(f"  [skip] {name}/{split} already exists at {split_dir}")
        else:
            splits_needed.append(split)

    if not splits_needed:
        print(f"[skip] {name} — all splits already downloaded")
        return

    print(f"\nDownloading {name} ({info['repo_id']}) — splits: {', '.join(splits_needed)} ...")

    for split in splits_needed:
        try:
            kwargs = {"split": split}
            if info.get("config"):
                ds = load_dataset(info["repo_id"], info["config"], **kwargs)
            else:
                ds = load_dataset(info["repo_id"], **kwargs)

            split_dir = save_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(split_dir))
            print(f"  {split}: {len(ds)} examples -> {split_dir}")

        except Exception as e:
            print(f"  Warning: could not download split '{split}': {e}")
            continue

    print(f"Done: {name}")


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable datasets:")
        print("-" * 70)
        for name, info in DATASETS.items():
            print(f"  {name:<12} {info['description']}")
            print(f"  {'':12} Repo: {info['repo_id']}, Size: {info['size']}")
            print(f"  {'':12} Splits: {', '.join(info['splits'])}")
            print()
        return

    if args.mirror:
        setup_mirror(args.mirror)

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = args.datasets if args.datasets else list(DATASETS.keys())

    # Validate names
    for name in dataset_names:
        if name not in DATASETS:
            print(f"Error: unknown dataset '{name}'. Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    print(f"Output directory: {output_dir}")
    print(f"Datasets to download: {', '.join(dataset_names)}")
    print()

    success, failed = [], []
    for name in dataset_names:
        try:
            download_dataset(name, DATASETS[name], output_dir, token, force=args.force)
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
