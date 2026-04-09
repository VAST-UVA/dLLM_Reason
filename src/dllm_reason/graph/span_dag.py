"""Span-level DAG — coarse-grained DAG operating on token spans (chunks).

Instead of building a DAG over individual tokens (O(n^2) adjacency),
SpanDAG groups consecutive tokens into spans of ``span_size`` and defines
dependencies between spans.  This reduces the search space by span_size^2
while preserving the core constraint: a span can only be unmasked after all
its parent spans are fully committed.

The SpanDAG internally stores a small (num_spans x num_spans) adjacency
matrix and can be expanded to a full TokenDAG for use with existing
schedulers via `to_token_dag()`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from dllm_reason.graph.dag import TokenDAG


class SpanDAG:
    """Coarse-grained DAG over token spans.

    Attributes:
        span_size:     number of tokens per span
        num_spans:     total number of spans
        seq_len:       original sequence length
        adjacency:     (num_spans, num_spans) bool — span-level edges
    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        span_size: int,
        seq_len: int,
        prompt_len: int = 0,
    ) -> None:
        self.adjacency = adjacency.bool()
        self.span_size = span_size
        self.seq_len = seq_len
        self.prompt_len = prompt_len
        self.num_spans = adjacency.shape[0]

    # ── Factories ──────────────────────────────────────────────────────────

    @classmethod
    def from_token_dag(
        cls,
        dag: TokenDAG,
        span_size: int,
        prompt_len: int = 0,
    ) -> "SpanDAG":
        """Compress a TokenDAG into a SpanDAG by grouping tokens."""
        seq_len = dag.seq_len
        gen_len = seq_len - prompt_len
        num_spans = math.ceil(gen_len / span_size)

        span_adj = torch.zeros(num_spans, num_spans, dtype=torch.bool,
                               device=dag.adjacency.device)

        token_adj = dag.adjacency[prompt_len:, prompt_len:]
        for si in range(num_spans):
            for sj in range(num_spans):
                if si == sj:
                    continue
                i_start, i_end = si * span_size, min((si + 1) * span_size, gen_len)
                j_start, j_end = sj * span_size, min((sj + 1) * span_size, gen_len)
                # Edge exists if ANY token in span_i has an edge to ANY token in span_j
                if token_adj[i_start:i_end, j_start:j_end].any():
                    span_adj[si, sj] = True

        return cls(span_adj, span_size, seq_len, prompt_len)

    @classmethod
    def from_levels(
        cls,
        span_levels: list[list[int]],
        num_spans: int,
        span_size: int,
        seq_len: int,
        prompt_len: int = 0,
        device: torch.device | str = "cpu",
    ) -> "SpanDAG":
        """Create a SpanDAG from ordered levels of span indices.

        span_levels: [[span_indices_level_0], [span_indices_level_1], ...]
        Each level depends on all previous levels.
        """
        adj = torch.zeros(num_spans, num_spans, dtype=torch.bool, device=device)
        prev_spans: list[int] = []
        for level in span_levels:
            for parent in prev_spans:
                for child in level:
                    if parent != child:
                        adj[parent, child] = True
            prev_spans.extend(level)
        return cls(adj, span_size, seq_len, prompt_len)

    @classmethod
    def empty(
        cls,
        seq_len: int,
        span_size: int = 8,
        prompt_len: int = 0,
        device: torch.device | str = "cpu",
    ) -> "SpanDAG":
        """Empty SpanDAG with no edges."""
        gen_len = seq_len - prompt_len
        num_spans = math.ceil(gen_len / span_size)
        adj = torch.zeros(num_spans, num_spans, dtype=torch.bool, device=device)
        return cls(adj, span_size, seq_len, prompt_len)

    @classmethod
    def linear_chain(
        cls,
        seq_len: int,
        span_size: int = 8,
        prompt_len: int = 0,
        device: torch.device | str = "cpu",
    ) -> "SpanDAG":
        """Linear left-to-right chain at span level."""
        gen_len = seq_len - prompt_len
        num_spans = math.ceil(gen_len / span_size)
        adj = torch.zeros(num_spans, num_spans, dtype=torch.bool, device=device)
        for i in range(num_spans - 1):
            adj[i, i + 1] = True
        return cls(adj, span_size, seq_len, prompt_len)

    @classmethod
    def cot(
        cls,
        seq_len: int,
        span_size: int = 8,
        num_cot_levels: int = 4,
        prompt_len: int = 0,
        device: torch.device | str = "cpu",
    ) -> "SpanDAG":
        """Chain-of-Thought at span level: partition spans into levels."""
        gen_len = seq_len - prompt_len
        num_spans = math.ceil(gen_len / span_size)
        spans_per_level = num_spans // num_cot_levels
        remainder = num_spans % num_cot_levels

        levels: list[list[int]] = []
        idx = 0
        for lv in range(num_cot_levels):
            n = spans_per_level + (1 if lv < remainder else 0)
            levels.append(list(range(idx, idx + n)))
            idx += n

        return cls.from_levels(levels, num_spans, span_size, seq_len, prompt_len, device)

    # ── Conversion ─────────────────────────────────────────────────────────

    def to_token_dag(self) -> TokenDAG:
        """Expand SpanDAG back to a full TokenDAG."""
        device = self.adjacency.device
        token_adj = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool, device=device)

        # Prompt → all generation positions
        if self.prompt_len > 0:
            token_adj[:self.prompt_len, self.prompt_len:] = True

        gen_len = self.seq_len - self.prompt_len
        for si in range(self.num_spans):
            for sj in range(self.num_spans):
                if not self.adjacency[si, sj]:
                    continue
                i_start = self.prompt_len + si * self.span_size
                i_end = min(i_start + self.span_size, self.seq_len)
                j_start = self.prompt_len + sj * self.span_size
                j_end = min(j_start + self.span_size, self.seq_len)
                # All tokens in span_i are parents of all tokens in span_j
                token_adj[i_start:i_end, j_start:j_end] = True

        return TokenDAG(token_adj)

    # ── Mutation operators (for search) ────────────────────────────────────

    def add_edge(self, src: int, dst: int) -> "SpanDAG":
        """Add a span-level edge, checking acyclicity."""
        new_adj = self.adjacency.clone()
        new_adj[src, dst] = True
        new_dag = SpanDAG(new_adj, self.span_size, self.seq_len, self.prompt_len)
        if not new_dag.is_valid():
            raise ValueError(f"Adding edge ({src}, {dst}) would create a cycle")
        return new_dag

    def remove_edge(self, src: int, dst: int) -> "SpanDAG":
        """Remove a span-level edge."""
        new_adj = self.adjacency.clone()
        new_adj[src, dst] = False
        return SpanDAG(new_adj, self.span_size, self.seq_len, self.prompt_len)

    def is_valid(self) -> bool:
        """Check if the span-level graph is a DAG (no cycles)."""
        # Use topological sort; if it fails, there's a cycle
        n = self.num_spans
        in_degree = self.adjacency.sum(dim=0).long()
        queue = (in_degree == 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        visited = 0
        adj = self.adjacency.clone()
        while queue:
            node = queue.pop(0)
            visited += 1
            for child in range(n):
                if adj[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return visited == n

    def num_edges(self) -> int:
        return int(self.adjacency.sum().item())

    def __repr__(self) -> str:
        return (
            f"SpanDAG(num_spans={self.num_spans}, span_size={self.span_size}, "
            f"edges={self.num_edges()}, seq_len={self.seq_len})"
        )
