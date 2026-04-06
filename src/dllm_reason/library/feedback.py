"""Feedback handlers: update DAG entries with evaluation signals.

Three independent sources: auto (benchmark), human, Elo tournament.
Each can be enabled/disabled independently via config.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Optional

from dllm_reason.library.config import FeedbackConfig, FeedbackSource
from dllm_reason.library.entry import DAGEntry
from dllm_reason.library.store import DAGStore


class FeedbackHandler(ABC):
    """Process a feedback signal and update the relevant DAGEntry."""

    @property
    @abstractmethod
    def source(self) -> FeedbackSource:
        ...

    @abstractmethod
    def update(self, entry: DAGEntry, store: DAGStore, **kwargs) -> DAGEntry:
        """Apply feedback and persist. Returns updated entry."""


# ── Auto (benchmark) ─────────────────────────────────────────────────────────

class AutoFeedback(FeedbackHandler):
    """Update entry with benchmark evaluation scores."""

    def __init__(self, metric: str = "accuracy"):
        self._metric = metric

    @property
    def source(self) -> FeedbackSource:
        return FeedbackSource.AUTO

    def update(
        self,
        entry: DAGEntry,
        store: DAGStore,
        benchmark: str = "",
        metrics: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> DAGEntry:
        if not benchmark or metrics is None:
            raise ValueError("AutoFeedback requires benchmark name and metrics dict.")
        entry.add_benchmark_score(benchmark, metrics)
        store.update(entry)
        return entry


# ── Human ────────────────────────────────────────────────────────────────────

class HumanFeedback(FeedbackHandler):
    """Record a human quality rating for a DAG entry."""

    @property
    def source(self) -> FeedbackSource:
        return FeedbackSource.HUMAN

    def update(
        self,
        entry: DAGEntry,
        store: DAGStore,
        rater_id: str = "anonymous",
        score: float = 3.0,
        **kwargs,
    ) -> DAGEntry:
        entry.add_human_rating(rater_id, score)
        store.update(entry)
        return entry


# ── Elo tournament ───────────────────────────────────────────────────────────

class EloFeedback(FeedbackHandler):
    """Elo rating system: update ratings after a head-to-head comparison."""

    def __init__(self, k: float = 32.0, initial: float = 1500.0):
        self._k = k
        self._initial = initial

    @property
    def source(self) -> FeedbackSource:
        return FeedbackSource.ELO

    def update(
        self,
        entry: DAGEntry,
        store: DAGStore,
        opponent: Optional[DAGEntry] = None,
        outcome: float = 1.0,  # 1.0=win, 0.5=draw, 0.0=loss
        **kwargs,
    ) -> DAGEntry:
        if opponent is None:
            raise ValueError("EloFeedback requires an opponent entry.")
        self.update_pair(entry, opponent, outcome, store)
        return entry

    def update_pair(
        self,
        entry_a: DAGEntry,
        entry_b: DAGEntry,
        outcome_a: float,
        store: DAGStore,
    ) -> tuple[float, float]:
        """Update both entries' Elo ratings. Returns (new_a, new_b)."""
        ra, rb = entry_a.elo_rating, entry_b.elo_rating
        ea = 1.0 / (1.0 + math.pow(10, (rb - ra) / 400.0))
        eb = 1.0 - ea

        entry_a.elo_rating = ra + self._k * (outcome_a - ea)
        entry_b.elo_rating = rb + self._k * ((1.0 - outcome_a) - eb)

        entry_a.updated_at = time.time()
        entry_b.updated_at = time.time()

        store.update(entry_a)
        store.update(entry_b)
        return entry_a.elo_rating, entry_b.elo_rating

    def run_tournament(
        self,
        entries: list[DAGEntry],
        outcomes: list[tuple[int, int, float]],
        store: DAGStore,
    ) -> list[DAGEntry]:
        """Run a batch of matchups. outcomes: [(idx_a, idx_b, outcome_a), ...]"""
        for idx_a, idx_b, outcome_a in outcomes:
            self.update_pair(entries[idx_a], entries[idx_b], outcome_a, store)
        return entries


# ── Factory ──────────────────────────────────────────────────────────────────

def create_feedback_handler(
    source: FeedbackSource, config: FeedbackConfig
) -> FeedbackHandler:
    if source == FeedbackSource.AUTO:
        return AutoFeedback(metric=config.auto_metric)
    elif source == FeedbackSource.HUMAN:
        return HumanFeedback()
    elif source == FeedbackSource.ELO:
        return EloFeedback(k=config.elo_k, initial=config.elo_initial)
    else:
        raise ValueError(f"Unknown feedback source: {source}")
