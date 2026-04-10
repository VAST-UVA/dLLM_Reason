"""Centralized registry of models and datasets.

Single source of truth for all HuggingFace resource metadata.
Both the download scripts (``scripts/download_models.py``,
``scripts/download_datasets.py``) and runtime loading
(``local_resolve.py``) import from here.

Adding a new resource
---------------------
1.  Models  — add an entry to ``MODEL_REGISTRY``.
2.  Datasets — add an entry to ``DATASET_REGISTRY``.
That's it.  The download scripts, local-first resolver, and all
loading sites will pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Project root & default directories ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"


# ── Model entry ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelEntry:
    """Metadata for a downloadable model checkpoint.

    Attributes:
        local_name:  Directory name under ``checkpoints/`` (e.g. ``llada-instruct``).
        repo_id:     HuggingFace repo ID (e.g. ``GSAI-ML/LLaDA-8B-Instruct``).
        description: Short human-readable description.
        size:        Approximate download size string.
    """
    local_name: str
    repo_id: str
    description: str = ""
    size: str = ""


# ── Dataset entry ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetEntry:
    """Metadata for a downloadable dataset.

    Attributes:
        local_name:  Directory name under ``datasets/`` (e.g. ``gsm8k``).
        repo_id:     HuggingFace dataset repo ID.
        config:      HF dataset config / subset (e.g. ``"main"``, ``"ARC-Challenge"``).
                     None if the dataset has no configs.
        splits:      List of splits to download (e.g. ``["train", "test"]``).
        description: Short human-readable description.
        size:        Approximate download size string.
    """
    local_name: str
    repo_id: str
    config: Optional[str] = None
    splits: list[str] = field(default_factory=lambda: ["train", "test"])
    description: str = ""
    size: str = ""


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY — add new models here
# ═════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, ModelEntry] = {
    "llada-instruct": ModelEntry(
        local_name="llada-instruct",
        repo_id="GSAI-ML/LLaDA-8B-Instruct",
        description="LLaDA 8B Instruct — main inference model",
        size="~16GB (bf16)",
    ),
    "llada-base": ModelEntry(
        local_name="llada-base",
        repo_id="GSAI-ML/LLaDA-8B-Base",
        description="LLaDA 8B Base — for fine-tuning",
        size="~16GB (bf16)",
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET REGISTRY — add new datasets here
# ═════════════════════════════════════════════════════════════════════════════

DATASET_REGISTRY: dict[str, DatasetEntry] = {
    # ── Training / reasoning evaluation ──
    "gsm8k": DatasetEntry(
        local_name="gsm8k",
        repo_id="openai/gsm8k",
        config="main",
        splits=["train", "test"],
        description="Grade School Math 8K — arithmetic reasoning",
        size="~7MB",
    ),
    "math": DatasetEntry(
        local_name="math",
        repo_id="hendrycks/competition_math",
        config=None,
        splits=["train", "test"],
        description="MATH — competition-level math problems",
        size="~50MB",
    ),
    "arc": DatasetEntry(
        local_name="arc",
        repo_id="allenai/ai2_arc",
        config="ARC-Challenge",
        splits=["train", "test", "validation"],
        description="ARC-Challenge — science reasoning (multiple choice)",
        size="~1MB",
    ),
    "prontoqa": DatasetEntry(
        local_name="prontoqa",
        repo_id="renma/ProntoQA",
        config=None,
        splits=["train"],
        description="ProntoQA — logical reasoning",
        size="~2MB",
    ),
    # ── Benchmark evaluation ──
    "mbpp": DatasetEntry(
        local_name="mbpp",
        repo_id="google-research-datasets/mbpp",
        config="sanitized",
        splits=["test", "train", "prompt"],
        description="MBPP — basic Python programming problems",
        size="~2MB",
    ),
    "humaneval": DatasetEntry(
        local_name="humaneval",
        repo_id="openai/openai_humaneval",
        config=None,
        splits=["test"],
        description="HumanEval — Python code generation",
        size="~1MB",
    ),
    "hotpotqa": DatasetEntry(
        local_name="hotpotqa",
        repo_id="hotpot_qa",
        config="distractor",
        splits=["train", "validation"],
        description="HotpotQA — multi-hop question answering",
        size="~600MB",
    ),
    "mmlu": DatasetEntry(
        local_name="mmlu",
        repo_id="cais/mmlu",
        config="all",
        splits=["test", "validation"],
        description="MMLU — massive multitask language understanding",
        size="~4MB",
    ),
    "gpqa": DatasetEntry(
        local_name="gpqa",
        repo_id="Idavidrein/gpqa",
        config="gpqa_diamond",
        splits=["train"],
        description="GPQA Diamond — graduate-level science QA",
        size="~1MB",
    ),
    "aime": DatasetEntry(
        local_name="aime",
        repo_id="AI-MO/aimo-validation-aime",
        config=None,
        splits=["train"],
        description="AIME — competition math (integer answers 000-999)",
        size="~1MB",
    ),
}


# ── Lookup helpers ────────────────────────────────────────────────────────────

def _build_repo_to_local(
    registry: dict[str, ModelEntry | DatasetEntry],
) -> dict[str, str]:
    """Build a reverse mapping: repo_id → local_name."""
    return {entry.repo_id: entry.local_name for entry in registry.values()}


# Pre-built reverse maps (repo_id → local_name)
REPO_TO_MODEL_LOCAL: dict[str, str] = _build_repo_to_local(MODEL_REGISTRY)
REPO_TO_DATASET_LOCAL: dict[str, str] = _build_repo_to_local(DATASET_REGISTRY)


def get_model(name: str) -> ModelEntry:
    """Lookup a model by local name. Raises KeyError if not found."""
    return MODEL_REGISTRY[name]


def get_dataset(name: str) -> DatasetEntry:
    """Lookup a dataset by local name. Raises KeyError if not found."""
    return DATASET_REGISTRY[name]


def find_model_by_repo(repo_id: str) -> ModelEntry | None:
    """Find a model entry by HuggingFace repo_id."""
    local_name = REPO_TO_MODEL_LOCAL.get(repo_id)
    return MODEL_REGISTRY.get(local_name) if local_name else None


def find_dataset_by_repo(repo_id: str) -> DatasetEntry | None:
    """Find a dataset entry by HuggingFace repo_id."""
    local_name = REPO_TO_DATASET_LOCAL.get(repo_id)
    return DATASET_REGISTRY.get(local_name) if local_name else None


def list_models() -> list[str]:
    """Return all registered model local names."""
    return list(MODEL_REGISTRY.keys())


def list_datasets() -> list[str]:
    """Return all registered dataset local names."""
    return list(DATASET_REGISTRY.keys())
