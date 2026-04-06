"""Composite fitness: aggregates multiple evaluation signals with toggleable weights.

Graceful degradation: if a signal source is disabled, remaining weights are
renormalized automatically. This supports ablation experiments where individual
signals are removed to measure their contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dllm_reason.library.config import FitnessConfig, LibraryConfig
from dllm_reason.library.entry import DAGEntry


@dataclass
class FitnessBreakdown:
    """Detailed fitness result for interpretability."""
    total: float
    components: dict[str, float]       # signal_name -> weighted_contribution
    raw_scores: dict[str, float]       # signal_name -> raw_score
    weights: dict[str, float]          # signal_name -> normalised_weight


class CompositeFitness:
    """Compute a scalar fitness for a DAGEntry by combining multiple signals.

    Each signal can be independently disabled. Weights auto-renormalize.
    """

    def __init__(self, config: LibraryConfig):
        self._config = config
        self._weights = config.normalized_fitness_weights()

    def evaluate(
        self,
        entry: DAGEntry,
        benchmark: Optional[str] = None,
        metric: str = "accuracy",
    ) -> FitnessBreakdown:
        raw: dict[str, float] = {}
        components: dict[str, float] = {}

        # Accuracy signal (auto feedback)
        if "accuracy" in self._weights:
            if benchmark and benchmark in entry.benchmark_scores:
                raw["accuracy"] = entry.benchmark_scores[benchmark].get(metric, 0.0)
            else:
                raw["accuracy"] = entry.best_score(metric)
            components["accuracy"] = self._weights["accuracy"] * raw["accuracy"]

        # Human feedback signal
        if "human" in self._weights:
            avg = entry.avg_human_rating()
            # Normalize human ratings from [1,5] to [0,1]
            lo, hi = self._config.feedback.human_scale
            raw["human"] = (avg - lo) / (hi - lo) if avg > 0 else 0.0
            components["human"] = self._weights["human"] * raw["human"]

        # Structural prior signal
        if "structural" in self._weights:
            raw["structural"] = self._structural_score(entry)
            components["structural"] = self._weights["structural"] * raw["structural"]

        total = sum(components.values())
        return FitnessBreakdown(
            total=total,
            components=components,
            raw_scores=raw,
            weights=self._weights,
        )

    def score(
        self,
        entry: DAGEntry,
        benchmark: Optional[str] = None,
        metric: str = "accuracy",
    ) -> float:
        """Convenience: return scalar fitness only."""
        return self.evaluate(entry, benchmark, metric).total

    @staticmethod
    def _structural_score(entry: DAGEntry) -> float:
        """Heuristic structural quality score in [0, 1].

        Prefers DAGs with moderate depth and edge count (not too sparse, not too dense).
        """
        if entry.seq_len == 0:
            return 0.0
        max_edges = entry.seq_len * (entry.seq_len - 1) / 2
        if max_edges == 0:
            return 0.0
        density = entry.num_edges / max_edges
        # Bell curve centred at density=0.1 (sparse but structured)
        density_score = 4.0 * density * (1.0 - density)
        # Depth bonus — prefer deep over flat
        depth_ratio = min(entry.depth / max(entry.seq_len * 0.5, 1), 1.0)
        return 0.5 * density_score + 0.5 * depth_ratio
