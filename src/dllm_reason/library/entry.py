"""DAGEntry: the unit of storage in the DAG Library.

Each entry wraps a TokenDAG with rich metadata for retrieval and feedback.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class DAGEntry:
    """A single DAG record in the library.

    Attributes:
        entry_id: Unique identifier.
        adjacency: Flattened adjacency matrix (seq_len * seq_len) as list[int].
        seq_len: Sequence length this DAG was built for.
        task_description: Free-text description of the task / prompt.
        task_embedding: Dense vector from a sentence encoder (populated lazily).
        source: How this DAG was created (template / search / manual / merged).
        template_name: If source == "template", which template.
        search_method: If source == "search", which search algorithm.
        benchmark_scores: {benchmark_name: {metric: value}}.
        human_ratings: List of (rater_id, score, timestamp).
        elo_rating: Current Elo rating.
        num_edges: Cached edge count.
        depth: Cached topological depth.
        tags: Free-form tags for filtering.
        created_at: Unix timestamp.
        updated_at: Unix timestamp.
        metadata: Arbitrary extra data.
    """

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    adjacency: list[int] = field(default_factory=list)
    seq_len: int = 0
    task_description: str = ""
    task_embedding: Optional[list[float]] = None
    source: str = "unknown"           # template | search | manual | merged
    template_name: Optional[str] = None
    search_method: Optional[str] = None
    benchmark_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    human_ratings: list[tuple[str, float, float]] = field(default_factory=list)
    elo_rating: float = 1500.0
    num_edges: int = 0
    depth: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Conversions ──────────────────────────────────────────────────────

    @classmethod
    def from_token_dag(
        cls,
        dag: Any,  # TokenDAG — avoid circular import
        task_description: str = "",
        source: str = "unknown",
        **kwargs: Any,
    ) -> DAGEntry:
        """Create an entry from a TokenDAG object."""
        adj_flat = dag.adjacency.cpu().flatten().int().tolist()
        from dllm_reason.eval.dag_analysis import analyze_dag
        stats = analyze_dag(dag)
        return cls(
            adjacency=adj_flat,
            seq_len=dag.seq_len,
            task_description=task_description,
            source=source,
            num_edges=stats.num_edges,
            depth=stats.depth,
            **kwargs,
        )

    def to_token_dag(self, device: str = "cpu") -> Any:
        """Reconstruct a TokenDAG from this entry."""
        from dllm_reason.graph.dag import TokenDAG
        adj = torch.tensor(self.adjacency, dtype=torch.bool, device=device)
        adj = adj.reshape(self.seq_len, self.seq_len)
        return TokenDAG(adj)

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "adjacency": self.adjacency,
            "seq_len": self.seq_len,
            "task_description": self.task_description,
            "task_embedding": self.task_embedding,
            "source": self.source,
            "template_name": self.template_name,
            "search_method": self.search_method,
            "benchmark_scores": self.benchmark_scores,
            "human_ratings": self.human_ratings,
            "elo_rating": self.elo_rating,
            "num_edges": self.num_edges,
            "depth": self.depth,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DAGEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> DAGEntry:
        return cls.from_dict(json.loads(s))

    # ── Feedback helpers ─────────────────────────────────────────────────

    def add_benchmark_score(self, benchmark: str, metrics: dict[str, float]) -> None:
        self.benchmark_scores[benchmark] = metrics
        self.updated_at = time.time()

    def add_human_rating(self, rater_id: str, score: float) -> None:
        self.human_ratings.append((rater_id, score, time.time()))
        self.updated_at = time.time()

    def avg_human_rating(self) -> float:
        if not self.human_ratings:
            return 0.0
        return sum(r[1] for r in self.human_ratings) / len(self.human_ratings)

    def best_score(self, metric: str = "accuracy") -> float:
        """Best score across all benchmarks for a given metric."""
        best = 0.0
        for scores in self.benchmark_scores.values():
            if metric in scores:
                best = max(best, scores[metric])
        return best
