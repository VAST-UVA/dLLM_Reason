"""Local-first resolution for models and datasets.

Before fetching from HuggingFace Hub, check whether the resource has
already been downloaded to the project's local directories:

  - Models:   checkpoints/<name>/   (by download_models.py)
  - Datasets: datasets/<name>/<split>/  (by download_datasets.py, save_to_disk format)

This avoids unnecessary network access when offline or when data is
already available locally.

Resource metadata (repo_id, local_name, etc.) is defined once in
``resource_registry.py`` — not duplicated here.

Mirror support:
  Set ``HF_MIRROR`` (e.g. ``https://hf-mirror.com``) to route all
  HuggingFace downloads through a mirror.  Alternatively, set
  ``HF_ENDPOINT`` directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dllm_reason.utils.logging import get_logger
from dllm_reason.utils.resource_registry import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_DATASETS_DIR,
    REPO_TO_MODEL_LOCAL,
    REPO_TO_DATASET_LOCAL,
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
    """Return a local checkpoint path if it exists, otherwise return *model_id* as-is.

    Lookup order:
      1. *model_id* is already a local directory → use directly.
      2. ``checkpoints/<local_name>/`` exists and is non-empty → use it.
      3. Return *model_id* unchanged (triggers HuggingFace download).
    """
    # Already a local path
    if Path(model_id).is_dir():
        logger.info(f"Model path is already local: {model_id}")
        return model_id

    # Resolve via registry (repo_id → local_name)
    local_name = REPO_TO_MODEL_LOCAL.get(model_id)
    if local_name is None:
        # Heuristic: "org/model-name" → try "model-name"
        if "/" in model_id:
            local_name = model_id.split("/")[-1].lower().replace("_", "-")
        else:
            return model_id

    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else DEFAULT_CHECKPOINTS_DIR
    local_path = ckpt_dir / local_name

    if local_path.is_dir() and any(local_path.iterdir()):
        logger.info(f"Found local checkpoint: {local_path}  (skipping download)")
        return str(local_path)

    _ensure_mirror()
    logger.info(f"No local checkpoint at {local_path}, will download from {model_id}")
    return model_id


# ── Dataset resolution ────────────────────────────────────────────────────────

def resolve_dataset(
    repo_id: str,
    config: str | None = None,
    split: str = "train",
    datasets_dir: str | Path | None = None,
):
    """Load a dataset locally if available, otherwise fall back to HuggingFace.

    Checks ``datasets/<local_name>/<split>/`` for a ``save_to_disk()``
    snapshot.  If found, loads with ``load_from_disk()``.  Otherwise
    downloads via ``load_dataset()`` (mirror auto-applied if configured).
    """
    from datasets import load_dataset, load_from_disk

    local_name = REPO_TO_DATASET_LOCAL.get(repo_id)
    ds_dir = Path(datasets_dir) if datasets_dir else DEFAULT_DATASETS_DIR

    if local_name:
        split_dir = ds_dir / local_name / split
        if (split_dir / "dataset_info.json").exists():
            logger.info(f"Loading dataset from local: {split_dir}")
            return load_from_disk(str(split_dir))

    # Fallback: download from HuggingFace
    _ensure_mirror()
    logger.info(f"Local dataset not found, downloading: {repo_id} (split={split})")
    if config:
        return load_dataset(repo_id, config, split=split)
    else:
        return load_dataset(repo_id, split=split)
