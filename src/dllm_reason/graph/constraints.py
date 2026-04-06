"""DAG constraint validation and enforcement utilities."""

from __future__ import annotations

import torch

from dllm_reason.graph.dag import TokenDAG


def is_acyclic(adjacency: torch.Tensor) -> bool:
    """Check if an adjacency matrix represents a DAG (no cycles).

    Uses topological sort attempt via Kahn's algorithm.
    """
    try:
        dag = TokenDAG(adjacency)
        dag.topological_levels()
        return True
    except ValueError:
        return False


def enforce_acyclicity(adjacency: torch.Tensor) -> torch.Tensor:
    """Remove minimum edges to make the graph acyclic.

    Uses DFS-based cycle detection and removes back edges.
    """
    n = adjacency.shape[0]
    adj = adjacency.clone().bool()

    visited = torch.zeros(n, dtype=torch.bool)
    in_stack = torch.zeros(n, dtype=torch.bool)

    def dfs(node: int) -> list[tuple[int, int]]:
        """Returns back edges (edges that create cycles)."""
        back_edges = []
        visited[node] = True
        in_stack[node] = True

        for neighbor in adj[node].nonzero(as_tuple=False).squeeze(-1).tolist():
            if isinstance(neighbor, int):
                if not visited[neighbor]:
                    back_edges.extend(dfs(neighbor))
                elif in_stack[neighbor]:
                    back_edges.append((node, neighbor))

        in_stack[node] = False
        return back_edges

    all_back_edges = []
    for node in range(n):
        if not visited[node]:
            all_back_edges.extend(dfs(node))

    # Remove back edges
    for src, dst in all_back_edges:
        adj[src, dst] = False

    return adj


def topological_mutation(
    dag: TokenDAG,
    num_add: int = 1,
    num_remove: int = 1,
) -> TokenDAG:
    """Mutate a DAG while guaranteeing acyclicity by construction.

    Add edges only consistent with current topological ordering
    (from lower to higher level). Remove edges randomly.
    """
    levels = dag.topological_levels()
    # Build level map: position -> level index
    level_map = {}
    for level_idx, positions in enumerate(levels):
        for pos in positions:
            level_map[pos] = level_idx

    adj = dag.adjacency.clone()
    seq_len = dag.seq_len

    # Remove random existing edges
    existing_edges = adj.nonzero(as_tuple=False)
    if len(existing_edges) > 0 and num_remove > 0:
        remove_idx = torch.randperm(len(existing_edges))[:num_remove]
        for idx in remove_idx:
            src, dst = existing_edges[idx].tolist()
            adj[src, dst] = False

    # Add random edges respecting topological order
    added = 0
    attempts = 0
    max_attempts = num_add * 10
    while added < num_add and attempts < max_attempts:
        src = torch.randint(0, seq_len, (1,)).item()
        dst = torch.randint(0, seq_len, (1,)).item()
        if src != dst and not adj[src, dst] and level_map.get(src, 0) < level_map.get(dst, 0):
            adj[src, dst] = True
            added += 1
        attempts += 1

    return TokenDAG(adj)
