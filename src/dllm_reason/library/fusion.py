"""Fusion strategies: combine rankings from multiple retrieval channels.

Each strategy is independently selectable via config. Set fusion.enabled=False
to skip fusion entirely and use a single channel's results directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from dllm_reason.library.config import FusionConfig, FusionStrategy
from dllm_reason.library.entry import DAGEntry


class FusionMethod(ABC):
    """Merge ranked lists from multiple channels into one."""

    @abstractmethod
    def fuse(
        self,
        channel_results: dict[str, list[tuple[DAGEntry, float]]],
        top_k: int = 5,
    ) -> list[tuple[DAGEntry, float]]:
        """channel_results: {channel_name: [(entry, score), ...]}"""


class WeightedFusion(FusionMethod):
    """Weighted sum of normalised scores across channels."""

    def __init__(self, weights: dict[str, float]):
        self._weights = weights

    def fuse(
        self,
        channel_results: dict[str, list[tuple[DAGEntry, float]]],
        top_k: int = 5,
    ) -> list[tuple[DAGEntry, float]]:
        combined: dict[str, tuple[DAGEntry, float]] = {}

        for channel, results in channel_results.items():
            w = self._weights.get(channel, 1.0)
            # Min-max normalise scores within this channel
            if not results:
                continue
            scores = [s for _, s in results]
            lo, hi = min(scores), max(scores)
            span = hi - lo if hi > lo else 1.0

            for entry, score in results:
                norm_score = (score - lo) / span
                eid = entry.entry_id
                if eid in combined:
                    combined[eid] = (entry, combined[eid][1] + w * norm_score)
                else:
                    combined[eid] = (entry, w * norm_score)

        ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class RRFFusion(FusionMethod):
    """Reciprocal Rank Fusion — robust to score scale differences."""

    def __init__(self, k: int = 60):
        self._k = k

    def fuse(
        self,
        channel_results: dict[str, list[tuple[DAGEntry, float]]],
        top_k: int = 5,
    ) -> list[tuple[DAGEntry, float]]:
        rrf_scores: dict[str, tuple[DAGEntry, float]] = {}

        for _channel, results in channel_results.items():
            for rank, (entry, _score) in enumerate(results):
                eid = entry.entry_id
                rrf = 1.0 / (self._k + rank + 1)
                if eid in rrf_scores:
                    rrf_scores[eid] = (entry, rrf_scores[eid][1] + rrf)
                else:
                    rrf_scores[eid] = (entry, rrf)

        ranked = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class MaxFusion(FusionMethod):
    """Take max score across channels for each entry."""

    def fuse(
        self,
        channel_results: dict[str, list[tuple[DAGEntry, float]]],
        top_k: int = 5,
    ) -> list[tuple[DAGEntry, float]]:
        best: dict[str, tuple[DAGEntry, float]] = {}

        for _channel, results in channel_results.items():
            for entry, score in results:
                eid = entry.entry_id
                if eid not in best or score > best[eid][1]:
                    best[eid] = (entry, score)

        ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class VotingFusion(FusionMethod):
    """Borda count voting — each channel votes by rank position."""

    def fuse(
        self,
        channel_results: dict[str, list[tuple[DAGEntry, float]]],
        top_k: int = 5,
    ) -> list[tuple[DAGEntry, float]]:
        votes: dict[str, tuple[DAGEntry, float]] = {}

        for _channel, results in channel_results.items():
            n = len(results)
            for rank, (entry, _score) in enumerate(results):
                borda = n - rank
                eid = entry.entry_id
                if eid in votes:
                    votes[eid] = (entry, votes[eid][1] + borda)
                else:
                    votes[eid] = (entry, float(borda))

        ranked = sorted(votes.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ── Factory ──────────────────────────────────────────────────────────────────

def create_fusion(config: FusionConfig) -> FusionMethod:
    if config.strategy == FusionStrategy.WEIGHTED:
        return WeightedFusion(config.channel_weights)
    elif config.strategy == FusionStrategy.RRF:
        return RRFFusion(k=config.rrf_k)
    elif config.strategy == FusionStrategy.MAX:
        return MaxFusion()
    elif config.strategy == FusionStrategy.VOTING:
        return VotingFusion()
    elif config.strategy == FusionStrategy.NONE:
        # Pass-through: return first channel's results
        return MaxFusion()  # Effectively same when single channel
    else:
        raise ValueError(f"Unknown fusion strategy: {config.strategy}")
