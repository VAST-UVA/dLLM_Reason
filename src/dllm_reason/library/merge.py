"""DAG merge strategies: combine multiple retrieved DAGs into one.

Each strategy is independently selectable. Set merge.enabled=False to skip
merging and use the top-1 retrieved DAG directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from dllm_reason.library.config import MergeConfig, MergeStrategy
from dllm_reason.library.entry import DAGEntry


class DAGMerger(ABC):
    """Combine multiple DAG adjacency matrices into one."""

    @abstractmethod
    def merge(self, entries: list[DAGEntry], scores: list[float]) -> torch.Tensor:
        """Return a boolean adjacency matrix (seq_len, seq_len).

        Args:
            entries: DAGEntries to merge (must share same seq_len).
            scores: Retrieval relevance scores, same length as entries.
        """


class UnionMerger(DAGMerger):
    """OR of all adjacencies — maximally constrained."""

    def merge(self, entries: list[DAGEntry], scores: list[float]) -> torch.Tensor:
        if not entries:
            raise ValueError("No entries to merge.")
        seq_len = entries[0].seq_len
        result = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for e in entries:
            adj = torch.tensor(e.adjacency, dtype=torch.bool).reshape(seq_len, seq_len)
            result |= adj
        # Ensure acyclicity
        return _enforce_acyclicity(result)


class IntersectionMerger(DAGMerger):
    """AND of all adjacencies — only consensus edges."""

    def merge(self, entries: list[DAGEntry], scores: list[float]) -> torch.Tensor:
        if not entries:
            raise ValueError("No entries to merge.")
        seq_len = entries[0].seq_len
        result = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for e in entries:
            adj = torch.tensor(e.adjacency, dtype=torch.bool).reshape(seq_len, seq_len)
            result &= adj
        return result  # intersection of DAGs is always acyclic if inputs are


class WeightedMerger(DAGMerger):
    """Weighted soft vote: edge kept if weighted sum > threshold."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold

    def merge(self, entries: list[DAGEntry], scores: list[float]) -> torch.Tensor:
        if not entries:
            raise ValueError("No entries to merge.")
        seq_len = entries[0].seq_len

        # Normalise scores
        total = sum(scores) or 1.0
        weights = [s / total for s in scores]

        accumulator = torch.zeros(seq_len, seq_len, dtype=torch.float32)
        for e, w in zip(entries, weights):
            adj = torch.tensor(e.adjacency, dtype=torch.float32).reshape(seq_len, seq_len)
            accumulator += w * adj

        result = accumulator > self._threshold
        return _enforce_acyclicity(result)


# ── Acyclicity enforcement ───────────────────────────────────────────────────

def _enforce_acyclicity(adj: torch.Tensor) -> torch.Tensor:
    """Remove back edges via DFS to guarantee a valid DAG."""
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    result = adj.clone()

    def dfs(u: int) -> None:
        color[u] = GRAY
        for v in range(n):
            if not result[u, v]:
                continue
            if color[v] == GRAY:
                result[u, v] = False  # back edge — remove
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for u in range(n):
        if color[u] == WHITE:
            dfs(u)

    return result


# ── Factory ──────────────────────────────────────────────────────────────────

def create_merger(config: MergeConfig) -> DAGMerger:
    if config.strategy == MergeStrategy.UNION:
        return UnionMerger()
    elif config.strategy == MergeStrategy.INTERSECTION:
        return IntersectionMerger()
    elif config.strategy == MergeStrategy.WEIGHTED:
        return WeightedMerger(threshold=config.weighted_threshold)
    else:
        raise ValueError(f"Unknown merge strategy: {config.strategy}")
