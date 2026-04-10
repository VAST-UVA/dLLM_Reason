"""Local-first resolution for models and datasets.

Resolution order (for both models and datasets):
  1. ``entry.local_path``  — explicit user-specified path (highest priority)
  2. ``<default_dir>/<local_name>/``  — project-relative convention
     (checkpoints/<name> for models, datasets/<name>/<split> for datasets)
  3. HuggingFace download  — remote fallback (lowest priority)

Resource metadata is defined in ``resource_registry.py``.

Mirror support:
  Set ``HF_MIRROR`` (e.g. ``https://hf-mirror.com``) to auto-set
  ``HF_ENDPOINT`` for all HuggingFace downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

from dllm_reason.utils.logging import get_logger
from dllm_reason.utils.resource_registry import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_DATASETS_DIR,
    REPO_TO_MODEL_KEY,
    REPO_TO_DATASET_KEY,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
)

logger = get_logger(__name__)


# ── Mirror setup ──────────────────────────────────────────────────────────────

def setup_hf_mirror(mirror_url: str | None = None) -> None:
    """Configure HuggingFace endpoint to use a mirror.

    Priority:
      1. Explicit ``mirror_url`` argument
      2. ``HF_MIRROR`` environment variable
      3. ``HF_ENDPOINT`` environment variable (already set by user)
      4. No action (use default huggingface.co)
    """
    if mirror_url:
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info(f"HuggingFace mirror set: {mirror_url}")
        return

    hf_mirror = os.environ.get("HF_MIRROR")
    if hf_mirror:
        os.environ["HF_ENDPOINT"] = hf_mirror
        logger.info(f"HuggingFace mirror set from HF_MIRROR: {hf_mirror}")
        return

    if os.environ.get("HF_ENDPOINT"):
        logger.debug(f"HF_ENDPOINT already set: {os.environ['HF_ENDPOINT']}")


def _ensure_mirror() -> None:
    """Auto-apply mirror if HF_MIRROR is set but HF_ENDPOINT is not."""
    hf_mirror = os.environ.get("HF_MIRROR")
    if hf_mirror and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = hf_mirror
        logger.info(f"Auto-applied HF_MIRROR: {hf_mirror}")


# ── Model resolution ─────────────────────────────────────────────────────────

def resolve_model_path(
    model_id: str,
    checkpoints_dir: str | Path | None = None,
) -> str:
    """Resolve a model identifier to a loadable path.

    Resolution order:
      1. ``model_id`` is already a local directory → use directly.
      2. Registry ``local_path`` is set and exists → use it.
      3. ``checkpoints/<local_name>/`` exists (non-empty) → use it.
      4. Return ``model_id`` unchanged (triggers HuggingFace download).

    Args:
        model_id: HuggingFace repo ID or local path.
        checkpoints_dir: Override for the checkpoints base directory.

    Returns:
        Local path string if available, otherwise the original model_id.
    """
    # 1. Already a local path that exists
    if Path(model_id).is_dir():
        logger.info(f"Model path is already local: {model_id}")
        return model_id

    # Look up registry entry
    key = REPO_TO_MODEL_KEY.get(model_id)
    entry = MODEL_REGISTRY.get(key) if key else None

    # 2. Explicit local_path from registry
    if entry and entry.local_path:
        p = Path(entry.local_path)
        if p.is_dir() and any(p.iterdir()):
            logger.info(f"Using registry local_path: {p}")
            return str(p)
        else:
            logger.warning(f"Registry local_path not found: {p}, trying defaults")

    # 3. Default checkpoints/<local_name>/
    local_name = entry.local_name if entry else None
    if local_name is None and "/" in model_id:
        local_name = model_id.split("/")[-1].lower().replace("_", "-")

    if local_name:
        ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else DEFAULT_CHECKPOINTS_DIR
        local_path = ckpt_dir / local_name
        if local_path.is_dir() and any(local_path.iterdir()):
            logger.info(f"Found local checkpoint: {local_path}")
            return str(local_path)

    # 4. Fallback: will download from HuggingFace
    _ensure_mirror()
    logger.info(f"No local model found, will download from {model_id}")
    return model_id


# ── Dataset resolution ────────────────────────────────────────────────────────

def resolve_dataset(
    repo_id: str,
    config: str | None = None,
    split: str = "train",
    datasets_dir: str | Path | None = None,
):
    """Load a dataset, checking local paths before downloading.

    Resolution order:
      1. Registry ``local_path/<split>/`` has ``dataset_info.json`` → load_from_disk.
      2. ``datasets/<local_name>/<split>/`` has ``dataset_info.json`` → load_from_disk.
      3. HuggingFace ``load_dataset()`` (remote fallback).

    Args:
        repo_id:  HuggingFace dataset repo ID.
        config:   Dataset config / subset name.
        split:    Split to load.
        datasets_dir: Override for the datasets base directory.

    Returns:
        A HuggingFace ``Dataset`` object.
    """
    from datasets import load_dataset, load_from_disk

    # Look up registry entry
    key = REPO_TO_DATASET_KEY.get(repo_id)
    entry = DATASET_REGISTRY.get(key) if key else None

    ds_dir = Path(datasets_dir) if datasets_dir else DEFAULT_DATASETS_DIR

    # 1. Explicit local_path from registry
    if entry and entry.local_path:
        split_dir = Path(entry.local_path) / split
        if (split_dir / "dataset_info.json").exists():
            logger.info(f"Loading dataset from registry local_path: {split_dir}")
            return load_from_disk(str(split_dir))
        else:
            logger.debug(f"Registry local_path split not found: {split_dir}")

    # 2. Default datasets/<local_name>/<split>/
    local_name = entry.local_name if entry else None
    if local_name:
        split_dir = ds_dir / local_name / split
        if (split_dir / "dataset_info.json").exists():
            logger.info(f"Loading dataset from local: {split_dir}")
            return load_from_disk(str(split_dir))

    # 3. Fallback: download from HuggingFace
    _ensure_mirror()
    logger.info(f"Local dataset not found, downloading: {repo_id} (split={split})")
    if config:
        return load_dataset(repo_id, config, split=split)
    else:
        return load_dataset(repo_id, split=split)
