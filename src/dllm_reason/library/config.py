"""Master configuration for DAG Library with ablation toggles.

Every component is independently toggleable so experiments can isolate variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums for strategy selection ─────────────────────────────────────────────

class RetrievalMode(Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    PERFORMANCE = "performance"


class FusionStrategy(Enum):
    WEIGHTED = "weighted"
    RRF = "rrf"           # Reciprocal Rank Fusion
    MAX = "max"
    VOTING = "voting"
    NONE = "none"         # skip fusion, use single channel


class ConstraintMode(Enum):
    HARD = "hard"         # strict DAG enforcement
    SOFT = "soft"         # penalty-based relaxation
    NONE = "none"         # no constraints from library


class FeedbackSource(Enum):
    AUTO = "auto"         # benchmark accuracy
    HUMAN = "human"       # human ratings
    ELO = "elo"           # Elo tournament rating


class MergeStrategy(Enum):
    UNION = "union"
    INTERSECTION = "intersection"
    WEIGHTED = "weighted"


# ── Component configs ────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    """Which retrieval channels to activate and how."""
    enabled: bool = True
    channels: list[RetrievalMode] = field(
        default_factory=lambda: [RetrievalMode.SEMANTIC]
    )
    top_k: int = 5
    semantic_model: str = "all-MiniLM-L6-v2"
    structural_metric: str = "edit_distance"  # edit_distance | spectral
    performance_weight_decay: float = 0.95    # recent results weighted more


@dataclass
class FusionConfig:
    """How to fuse results from multiple retrieval channels."""
    enabled: bool = True
    strategy: FusionStrategy = FusionStrategy.WEIGHTED
    channel_weights: dict[str, float] = field(
        default_factory=lambda: {"semantic": 0.5, "structural": 0.3, "performance": 0.2}
    )
    rrf_k: int = 60  # RRF constant


@dataclass
class FeedbackConfig:
    """Which feedback signals to collect and use."""
    enabled: bool = True
    sources: list[FeedbackSource] = field(
        default_factory=lambda: [FeedbackSource.AUTO]
    )
    auto_metric: str = "accuracy"     # accuracy | f1 | pass@1
    elo_initial: float = 1500.0
    elo_k: float = 32.0
    human_scale: tuple[float, float] = (1.0, 5.0)


@dataclass
class ConstraintConfig:
    """How retrieved DAGs constrain generation."""
    mode: ConstraintMode = ConstraintMode.HARD
    soft_penalty_weight: float = 0.1


@dataclass
class MergeConfig:
    """How to merge multiple retrieved DAGs into one."""
    enabled: bool = True
    strategy: MergeStrategy = MergeStrategy.UNION
    weighted_threshold: float = 0.5  # edge kept if weight > threshold


@dataclass
class FitnessConfig:
    """Composite fitness weights — each signal independently toggleable."""
    accuracy_weight: float = 0.5
    human_weight: float = 0.3
    structural_weight: float = 0.2
    # Graceful degradation: if a signal source is disabled,
    # remaining weights are renormalized automatically.


@dataclass
class StoreConfig:
    """Storage backend settings."""
    db_path: str = "dag_library.db"
    faiss_index_path: str = "dag_library.faiss"
    embedding_dim: int = 384  # matches all-MiniLM-L6-v2


# ── Master config ────────────────────────────────────────────────────────────

@dataclass
class LibraryConfig:
    """Top-level config aggregating all sub-configs.

    Set any sub-config's ``enabled`` to False to disable that component.
    This is the single entry point for ablation experiments.
    """
    enabled: bool = True              # kill-switch for entire library
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)

    def active_feedback_sources(self) -> list[FeedbackSource]:
        if not self.feedback.enabled:
            return []
        return list(self.feedback.sources)

    def active_retrieval_channels(self) -> list[RetrievalMode]:
        if not self.retrieval.enabled:
            return []
        return list(self.retrieval.channels)

    def normalized_fitness_weights(self) -> dict[str, float]:
        """Return fitness weights renormalized over enabled sources only."""
        w = {}
        if FeedbackSource.AUTO in self.active_feedback_sources():
            w["accuracy"] = self.fitness.accuracy_weight
        if FeedbackSource.HUMAN in self.active_feedback_sources():
            w["human"] = self.fitness.human_weight
        # structural prior is always available if retrieval is on
        if self.retrieval.enabled:
            w["structural"] = self.fitness.structural_weight
        total = sum(w.values()) or 1.0
        return {k: v / total for k, v in w.items()}
