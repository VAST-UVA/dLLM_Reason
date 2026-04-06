"""Abstract base class for DAG structure search methods."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.models.base import DiffusionLM


@dataclass
class SearchResult:
    """Result of a DAG search run."""
    best_dag: TokenDAG
    best_fitness: float
    history: list[dict]  # [{dag, fitness, step}, ...]
    metadata: dict


class DAGSearcher(abc.ABC):
    """Abstract interface for DAG structure optimization.

    All search methods find a TokenDAG that maximizes a fitness function
    (typically downstream task accuracy) when used to guide unmasking.
    """

    @abc.abstractmethod
    def search(
        self,
        model: DiffusionLM,
        eval_fn: callable,
        seq_len: int,
        budget: int = 100,
        **kwargs,
    ) -> SearchResult:
        """Run the search.

        Args:
            model: the dLLM to evaluate with
            eval_fn: callable(model, dag) -> float, fitness score
            seq_len: sequence length for the DAG
            budget: computational budget (number of evaluations)

        Returns:
            SearchResult with the best DAG found
        """
        ...
