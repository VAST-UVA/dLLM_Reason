"""Local-first resolution for models and datasets.

Before fetching from HuggingFace Hub, check whether the resource has
already been downloaded to the project's local directories:

  - Models:   checkpoints/<name>/   (by download_models.py)
  - Datasets: datasets/<name>/<split>/  (by download_datasets.py, save_to_disk format)

This avoids unnecessary network access when offline or when data is
already available locally.

Mirror support:
  Set the environment variable ``HF_MIRROR`` (e.g. ``https://hf-mirror.com``)
  to route all HuggingFace downloads through a mirror.  This is useful in
  mainland China where huggingface.co is not directly accessible.
  Alternatively, set ``HF_ENDPOINT`` directly (standard HuggingFace env var).
"""

from __future__ import annotations

import os
from pathlib import Path

from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)

# Project root — two levels up from src/dllm_reason/utils/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default local directories (same as download_models.py / download_datasets.py)
DEFAULT_CHECKPOINTS_DIR = _PROJECT_ROOT / "checkpoints"
DEFAULT_DATASETS_DIR = _PROJECT_ROOT / "datasets"

# Default HuggingFace mirror for China mainland
DEFAULT_HF_MIRROR = "https://hf-mirror.com"


# ── Mirror setup ──────────────────────────────────────────────────────────────

def setup_hf_mirror(mirror_url: str | None = None) -> None:
    """Configure HuggingFace endpoint to use a mirror.

    Priority:
      1. Explicit ``mirror_url`` argument
      2. ``HF_MIRROR`` environment variable
      3. ``HF_ENDPOINT`` environment variable (already set by user)
      4. No action (use default huggingface.co)

    The mirror is set via ``HF_ENDPOINT``, which is respected by both
    ``huggingface_hub`` and the ``datasets`` library.
    """
    if mirror_url:
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info(f"HuggingFace mirror set: {mirror_url}")
        return

    # Check HF_MIRROR env var
    hf_mirror = os.environ.get("HF_MIRROR")
    if hf_mirror:
        os.environ["HF_ENDPOINT"] = hf_mirror
        logger.info(f"HuggingFace mirror set from HF_MIRROR: {hf_mirror}")
        return

    # HF_ENDPOINT already set — respect it
    if os.environ.get("HF_ENDPOINT"):
        logger.debug(f"HF_ENDPOINT already set: {os.environ['HF_ENDPOINT']}")


def _ensure_mirror() -> None:
    """Auto-apply mirror if HF_MIRROR is set but HF_ENDPOINT is not."""
    hf_mirror = os.environ.get("HF_MIRROR")
    if hf_mirror and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = hf_mirror
        logger.info(f"Auto-applied HF_MIRROR: {hf_mirror}")


# ── Model resolution ─────────────────────────────────────────────────────────

# HuggingFace repo_id → local directory name (mirrors MODELS in download_models.py)
_MODEL_LOCAL_NAMES: dict[str, str] = {
    "GSAI-ML/LLaDA-8B-Instruct": "llada-instruct",
    "GSAI-ML/LLaDA-8B-Base": "llada-base",
}


def resolve_model_path(
    model_id: str,
    checkpoints_dir: str | Path | None = None,
) -> str:
    """Return a local checkpoint path if it exists, otherwise return model_id as-is.

    Checks the following in order:
      1. ``model_id`` is already a local directory → use it directly.
      2. ``checkpoints/<local_name>/`` exists (non-empty) → use it.
      3. Return ``model_id`` unchanged (will trigger HuggingFace download).

    If falling back to HuggingFace, ``HF_MIRROR`` / ``HF_ENDPOINT`` is
    respected automatically.

    Args:
        model_id: HuggingFace repo ID (e.g. "GSAI-ML/LLaDA-8B-Instruct")
                  or an already-local path (e.g. "checkpoints/llada-instruct").
        checkpoints_dir: Override for the checkpoints directory.

    Returns:
        Local path string if found, otherwise the original model_id.
    """
    # Already a local path that exists — use it directly
    if Path(model_id).is_dir():
        logger.info(f"Model path is already local: {model_id}")
        return model_id

    # Try resolving from known HF repo → local name mapping
    local_name = _MODEL_LOCAL_NAMES.get(model_id)
    if local_name is None:
        # If model_id looks like "org/name", try the part after "/"
        if "/" in model_id:
            local_name = model_id.split("/")[-1].lower().replace("_", "-")
        else:
            return model_id

    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else DEFAULT_CHECKPOINTS_DIR
    local_path = ckpt_dir / local_name

    if local_path.is_dir() and any(local_path.iterdir()):
        logger.info(f"Found local checkpoint: {local_path}  (skipping HuggingFace download)")
        return str(local_path)

    # Will download — ensure mirror is applied
    _ensure_mirror()
    logger.info(f"No local checkpoint at {local_path}, will download from {model_id}")
    return model_id


# ── Dataset resolution ────────────────────────────────────────────────────────

# HuggingFace repo_id → local directory name (mirrors DATASETS in download_datasets.py)
_DATASET_LOCAL_NAMES: dict[str, str] = {
    "openai/gsm8k": "gsm8k",
    "hendrycks/competition_math": "math",
    "allenai/ai2_arc": "arc",
    "renma/ProntoQA": "prontoqa",
    "google-research-datasets/mbpp": "mbpp",
    "openai/openai_humaneval": "humaneval",
    "hotpot_qa": "hotpotqa",
    "cais/mmlu": "mmlu",
    "Idavidrein/gpqa": "gpqa",
    "AI-MO/aimo-validation-aime": "aime",
    "Maxwell-Jia/AIME_2024": "aime",
}


def resolve_dataset(
    repo_id: str,
    config: str | None = None,
    split: str = "train",
    datasets_dir: str | Path | None = None,
):
    """Load a dataset locally if available, otherwise fall back to HuggingFace.

    Checks ``datasets/<local_name>/<split>/`` for a ``save_to_disk()``
    snapshot (contains ``dataset_info.json``).  If found, loads with
    ``datasets.load_from_disk()``.  Otherwise downloads via
    ``datasets.load_dataset()`` (mirror is auto-applied if configured).

    Args:
        repo_id:  HuggingFace dataset repo ID.
        config:   Dataset config / subset name (e.g. "main", "sanitized").
        split:    Split to load ("train", "test", etc.).
        datasets_dir: Override for the datasets directory.

    Returns:
        A HuggingFace ``Dataset`` object.
    """
    from datasets import load_dataset, load_from_disk

    local_name = _DATASET_LOCAL_NAMES.get(repo_id)
    ds_dir = Path(datasets_dir) if datasets_dir else DEFAULT_DATASETS_DIR

    if local_name:
        split_dir = ds_dir / local_name / split
        if (split_dir / "dataset_info.json").exists():
            logger.info(f"Loading dataset from local: {split_dir}")
            return load_from_disk(str(split_dir))

    # Fallback: download from HuggingFace (auto-apply mirror)
    _ensure_mirror()
    logger.info(f"Local dataset not found, downloading: {repo_id} (split={split})")
    if config:
        return load_dataset(repo_id, config, split=split)
    else:
        return load_dataset(repo_id, split=split)
