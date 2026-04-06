"""Retrieval channels: each channel scores library entries by a different signal.

Abstract interface + three concrete channels (semantic, structural, performance).
Channels are independently toggleable via config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from dllm_reason.library.config import RetrievalConfig, RetrievalMode
from dllm_reason.library.embedder import TaskEmbedder
from dllm_reason.library.entry import DAGEntry
from dllm_reason.library.store import DAGStore


class RetrievalChannel(ABC):
    """Scores a set of candidate DAGEntries for a query."""

    @property
    @abstractmethod
    def mode(self) -> RetrievalMode:
        ...

    @abstractmethod
    def retrieve(
        self, query: RetrievalQuery, store: DAGStore, top_k: int = 5
    ) -> list[tuple[DAGEntry, float]]:
        """Return (entry, relevance_score) sorted descending."""


class RetrievalQuery:
    """Encapsulates a retrieval request with all possible query signals."""

    def __init__(
        self,
        task_description: str = "",
        task_embedding: Optional[np.ndarray] = None,
        reference_dag: Optional[DAGEntry] = None,
        target_seq_len: Optional[int] = None,
        target_benchmark: Optional[str] = None,
        target_metric: str = "accuracy",
    ):
        self.task_description = task_description
        self.task_embedding = task_embedding
        self.reference_dag = reference_dag
        self.target_seq_len = target_seq_len
        self.target_benchmark = target_benchmark
        self.target_metric = target_metric


# ── Semantic channel ─────────────────────────────────────────────────────────

class SemanticRetrieval(RetrievalChannel):
    """Retrieve DAGs whose task descriptions are semantically similar."""

    def __init__(self, embedder: TaskEmbedder):
        self._embedder = embedder

    @property
    def mode(self) -> RetrievalMode:
        return RetrievalMode.SEMANTIC

    def retrieve(
        self, query: RetrievalQuery, store: DAGStore, top_k: int = 5
    ) -> list[tuple[DAGEntry, float]]:
        if query.task_embedding is not None:
            qvec = query.task_embedding
        elif query.task_description:
            qvec = self._embedder.embed(query.task_description)
        else:
            return []

        # Ensure FAISS index is built
        if store._faiss_index is None:
            store.build_faiss_index()

        return store.search_by_embedding(qvec, top_k=top_k)


# ── Structural channel ──────────────────────────────────────────────────────

class StructuralRetrieval(RetrievalChannel):
    """Retrieve DAGs structurally similar to a reference DAG.

    Metrics: edge edit distance, spectral distance.
    """

    def __init__(self, metric: str = "edit_distance"):
        self._metric = metric

    @property
    def mode(self) -> RetrievalMode:
        return RetrievalMode.STRUCTURAL

    def retrieve(
        self, query: RetrievalQuery, store: DAGStore, top_k: int = 5
    ) -> list[tuple[DAGEntry, float]]:
        if query.reference_dag is None:
            return []

        ref = query.reference_dag
        candidates = store.list_all(limit=5000)
        # Filter to same seq_len for meaningful comparison
        candidates = [c for c in candidates if c.seq_len == ref.seq_len]

        scored = []
        ref_adj = np.array(ref.adjacency, dtype=np.float32)
        for c in candidates:
            c_adj = np.array(c.adjacency, dtype=np.float32)
            if len(c_adj) != len(ref_adj):
                continue
            dist = self._compute_distance(ref_adj, c_adj, ref.seq_len)
            # Convert distance to similarity (higher = better)
            sim = 1.0 / (1.0 + dist)
            scored.append((c, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _compute_distance(
        self, adj_a: np.ndarray, adj_b: np.ndarray, seq_len: int
    ) -> float:
        if self._metric == "edit_distance":
            return float(np.sum(np.abs(adj_a - adj_b)))
        elif self._metric == "spectral":
            return self._spectral_distance(adj_a, adj_b, seq_len)
        else:
            raise ValueError(f"Unknown structural metric: {self._metric}")

    @staticmethod
    def _spectral_distance(adj_a: np.ndarray, adj_b: np.ndarray, n: int) -> float:
        ma = adj_a.reshape(n, n)
        mb = adj_b.reshape(n, n)
        try:
            ea = np.sort(np.abs(np.linalg.eigvals(ma)))
            eb = np.sort(np.abs(np.linalg.eigvals(mb)))
            return float(np.linalg.norm(ea - eb))
        except np.linalg.LinAlgError:
            return float(np.sum(np.abs(adj_a - adj_b)))


# ── Performance channel ──────────────────────────────────────────────────────

class PerformanceRetrieval(RetrievalChannel):
    """Retrieve top-performing DAGs on a target benchmark/metric."""

    def __init__(self, weight_decay: float = 0.95):
        self._weight_decay = weight_decay

    @property
    def mode(self) -> RetrievalMode:
        return RetrievalMode.PERFORMANCE

    def retrieve(
        self, query: RetrievalQuery, store: DAGStore, top_k: int = 5
    ) -> list[tuple[DAGEntry, float]]:
        candidates = store.list_all(limit=5000)

        scored = []
        for c in candidates:
            score = self._score_entry(c, query.target_benchmark, query.target_metric)
            if score > 0:
                scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _score_entry(
        self, entry: DAGEntry, benchmark: Optional[str], metric: str
    ) -> float:
        if benchmark and benchmark in entry.benchmark_scores:
            return entry.benchmark_scores[benchmark].get(metric, 0.0)
        # Fallback: average across all benchmarks
        if not entry.benchmark_scores:
            return 0.0
        values = [
            scores.get(metric, 0.0) for scores in entry.benchmark_scores.values()
        ]
        return sum(values) / len(values) if values else 0.0


# ── Factory ──────────────────────────────────────────────────────────────────

def create_retrieval_channel(
    mode: RetrievalMode,
    config: RetrievalConfig,
    embedder: Optional[TaskEmbedder] = None,
) -> RetrievalChannel:
    if mode == RetrievalMode.SEMANTIC:
        if embedder is None:
            raise ValueError("SemanticRetrieval requires a TaskEmbedder.")
        return SemanticRetrieval(embedder)
    elif mode == RetrievalMode.STRUCTURAL:
        return StructuralRetrieval(metric=config.structural_metric)
    elif mode == RetrievalMode.PERFORMANCE:
        return PerformanceRetrieval(weight_decay=config.performance_weight_decay)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
