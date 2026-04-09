"""TokenDAG: Core data structure for DAG-guided unmasking.

The DAG operates over token positions (0..seq_len-1). Each directed edge
(i, j) means "position i must be unmasked BEFORE position j."

This is the foundational abstraction of the entire project. The DAG
defines reasoning dependencies between token positions, and the
DAGScheduler uses it to control the unmasking order during inference.
"""

from __future__ import annotations

from typing import Sequence

import torch
import networkx as nx


class TokenDAG:
    """Directed Acyclic Graph over token positions.

    Internally stored as a boolean adjacency matrix on GPU for efficient
    batch operations. adjacency[i][j] = True means edge i -> j
    (i must be unmasked before j).

    The critical operation `ready_positions` reduces to a single batched
    matrix operation, making it efficient for use inside the sampling loop.
    """

    def __init__(self, adjacency: torch.Tensor):
        """
        Args:
            adjacency: (seq_len, seq_len) boolean tensor.
                       adjacency[i][j] = True means i -> j (i before j).
        """
        assert adjacency.dim() == 2
        assert adjacency.shape[0] == adjacency.shape[1]
        self._adjacency = adjacency.bool()
        self._seq_len = adjacency.shape[0]

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def adjacency(self) -> torch.Tensor:
        return self._adjacency

    @property
    def device(self) -> torch.device:
        return self._adjacency.device

    def to(self, device: torch.device | str) -> TokenDAG:
        return TokenDAG(self._adjacency.to(device))

    @classmethod
    def empty(cls, seq_len: int, device: torch.device | str = "cpu") -> TokenDAG:
        """No edges — all positions are independent (fully parallel unmasking)."""
        adj = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        return cls(adj)

    @classmethod
    def linear_chain(cls, seq_len: int, device: torch.device | str = "cpu") -> TokenDAG:
        """Left-to-right chain: 0 -> 1 -> 2 -> ... -> seq_len-1.

        This mimics autoregressive generation order.
        """
        adj = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len - 1):
            adj[i, i + 1] = True
        return cls(adj)

    @classmethod
    def from_edges(
        cls,
        seq_len: int,
        edges: Sequence[tuple[int, int]],
        device: torch.device | str = "cpu",
    ) -> TokenDAG:
        """Construct from explicit edge list.

        Args:
            seq_len: total number of positions
            edges: list of (src, dst) tuples, meaning src must be unmasked before dst
        """
        adj = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for src, dst in edges:
            assert 0 <= src < seq_len and 0 <= dst < seq_len, f"Edge ({src}, {dst}) out of range"
            adj[src, dst] = True
        dag = cls(adj)
        assert dag.is_valid(), "Provided edges contain a cycle!"
        return dag

    @classmethod
    def from_levels(
        cls,
        levels: list[list[int]],
        seq_len: int | None = None,
        device: torch.device | str = "cpu",
    ) -> TokenDAG:
        """Construct from topological levels.

        Level 0 positions have no parents. Level k positions depend on
        all level k-1 positions.

        Args:
            levels: list of lists, levels[k] = positions at level k
            seq_len: total positions (inferred if None)
        """
        if seq_len is None:
            seq_len = max(pos for level in levels for pos in level) + 1

        adj = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for k in range(1, len(levels)):
            for parent in levels[k - 1]:
                for child in levels[k]:
                    adj[parent, child] = True
        return cls(adj)

    @classmethod
    def from_networkx(cls, G: nx.DiGraph, device: torch.device | str = "cpu") -> TokenDAG:
        """Construct from a NetworkX directed graph."""
        assert nx.is_directed_acyclic_graph(G), "Graph must be a DAG"
        seq_len = G.number_of_nodes()
        adj = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for src, dst in G.edges():
            adj[src, dst] = True
        return cls(adj)

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis/visualization."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self._seq_len))
        edges = self._adjacency.nonzero(as_tuple=False).tolist()
        G.add_edges_from(edges)
        return G

    def ready_positions(self, is_unmasked: torch.Tensor) -> torch.Tensor:
        """Determine which positions are ready to be unmasked.

        A position j is ready if ALL its parents (positions i where
        adjacency[i][j] = True) are already unmasked.

        This is the critical operation used inside the sampling loop.
        Reduces to a single batched matrix operation on GPU.

        Args:
            is_unmasked: (batch, seq_len) or (seq_len,) boolean tensor

        Returns:
            (batch, seq_len) or (seq_len,) boolean tensor, True where ready
        """
        was_1d = is_unmasked.dim() == 1
        if was_1d:
            is_unmasked = is_unmasked.unsqueeze(0)

        # Move adjacency to same device
        adj = self._adjacency.to(is_unmasked.device)

        # For each position j: check if all parents are unmasked
        # adj[i, j] = True means i is parent of j
        # We need: for all i where adj[i, j], is_unmasked[:, i] is True
        # Equivalent: (~adj[i, j]) | is_unmasked[:, i] should be True for all i
        # Shape: adj is (L, L), is_unmasked is (B, L)
        # (~adj).T is (L, L): (~adj).T[j, i] = ~adj[i, j]
        # We need: for each j, all( (~adj[:, j]) | is_unmasked[b, :] )

        # Expand: adj (L, L) -> (1, L, L), is_unmasked (B, L) -> (B, L, 1)
        # Check: (B, L_parent, L_child)
        condition = (~adj).unsqueeze(0) | is_unmasked.unsqueeze(-1)  # (B, L, L)
        ready = condition.all(dim=1)  # (B, L) — all parents satisfied

        if was_1d:
            ready = ready.squeeze(0)

        return ready

    def topological_levels(self) -> list[list[int]]:
        """Compute topological levels using Kahn's algorithm.

        Level 0 = roots (no parents).
        Level k = nodes whose all parents are in levels < k.

        Returns:
            List of lists, where levels[k] contains position indices at level k.
        """
        adj = self._adjacency.cpu()
        in_degree = adj.sum(dim=0).long()  # (L,) number of parents per node
        remaining = torch.ones(self._seq_len, dtype=torch.bool)

        levels = []
        while remaining.any():
            # Find nodes with in_degree 0 among remaining
            ready = (in_degree == 0) & remaining
            if not ready.any():
                raise ValueError("DAG contains a cycle!")
            level = ready.nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(level, int):
                level = [level]
            levels.append(level)
            # Remove these nodes: decrement in_degree of their children
            for node in level:
                remaining[node] = False
                children = adj[node].nonzero(as_tuple=False).squeeze(-1)
                if children.dim() == 0:
                    children = children.unsqueeze(0)
                for child in children.tolist():
                    in_degree[child] -= 1

        return levels

    def to_mask_schedule(self, num_steps: int) -> list[list[int]]:
        """Distribute topological levels across T diffusion steps.

        If num_levels <= num_steps: some steps handle one level, others are
        refinement steps (no new unmasking).
        If num_levels > num_steps: merge adjacent levels to fit.

        Args:
            num_steps: number of diffusion steps T

        Returns:
            List of length num_steps. schedule[step] = positions to unmask at step.
        """
        levels = self.topological_levels()
        num_levels = len(levels)

        if num_levels <= num_steps:
            # Map levels to steps, with empty steps for refinement
            schedule = [[] for _ in range(num_steps)]
            # Spread levels evenly across steps
            step_indices = torch.linspace(0, num_steps - 1, num_levels).long().tolist()
            for level_idx, step_idx in enumerate(step_indices):
                schedule[step_idx] = levels[level_idx]
        else:
            # Merge levels to fit into num_steps
            schedule = [[] for _ in range(num_steps)]
            levels_per_step = num_levels / num_steps
            for level_idx, level in enumerate(levels):
                step_idx = min(int(level_idx / levels_per_step), num_steps - 1)
                schedule[step_idx].extend(level)

        return schedule

    def is_valid(self) -> bool:
        """Check if the graph is a valid DAG (no cycles)."""
        try:
            self.topological_levels()
            return True
        except ValueError:
            return False

    def num_edges(self) -> int:
        return self._adjacency.sum().item()

    def depth(self) -> int:
        """Number of topological levels (longest path + 1)."""
        return len(self.topological_levels())

    def add_edges(self, edges: Sequence[tuple[int, int]]) -> TokenDAG:
        """Return a new DAG with additional edges (validates acyclicity)."""
        new_adj = self._adjacency.clone()
        for src, dst in edges:
            new_adj[src, dst] = True
        new_dag = TokenDAG(new_adj)
        if not new_dag.is_valid():
            raise ValueError("Adding these edges would create a cycle!")
        return new_dag

    def remove_edges(self, edges: Sequence[tuple[int, int]]) -> TokenDAG:
        """Return a new DAG with edges removed."""
        new_adj = self._adjacency.clone()
        for src, dst in edges:
            new_adj[src, dst] = False
        return TokenDAG(new_adj)

    def mutate(
        self,
        add: Sequence[tuple[int, int]] = (),
        remove: Sequence[tuple[int, int]] = (),
    ) -> TokenDAG:
        """Return a new validated DAG with edges added/removed."""
        new_adj = self._adjacency.clone()
        for src, dst in remove:
            new_adj[src, dst] = False
        for src, dst in add:
            new_adj[src, dst] = True
        new_dag = TokenDAG(new_adj)
        if not new_dag.is_valid():
            raise ValueError("Mutation would create a cycle!")
        return new_dag

    def subgraph(self, positions: Sequence[int]) -> TokenDAG:
        """Extract subgraph over a subset of positions."""
        positions = list(positions)
        n = len(positions)
        adj = torch.zeros(n, n, dtype=torch.bool, device=self.device)
        for i, pi in enumerate(positions):
            for j, pj in enumerate(positions):
                adj[i, j] = self._adjacency[pi, pj]
        return TokenDAG(adj)

    def transitive_closure(self) -> TokenDAG:
        """Compute transitive closure (all reachability edges)."""
        # Use matrix exponentiation: reach = (I + A)^n > 0
        adj_float = self._adjacency.float()
        identity = torch.eye(self._seq_len, device=self.device)
        reach = identity + adj_float
        for _ in range(self._seq_len.bit_length()):
            reach = torch.mm(reach, reach)
        # Threshold and remove self-loops
        closure = (reach > 0) & ~torch.eye(self._seq_len, dtype=torch.bool, device=self.device)
        return TokenDAG(closure)

    def to_visual_str(self, max_cols: int = 64) -> str:
        """Return a visual string representation of this DAG for terminal display.

        Output sections
        ---------------
        1. Header  — n / edges / depth / density.
        2. Level bar-chart  — one bar per topological level; bar width is
           proportional to the number of token positions at that level.
           Reading top-to-bottom gives the unmask order.
        3. Adjacency heatmap  — rows are src positions, columns are dst
           positions.  '#' = edge present, '.' = no edge.  When seq_len
           exceeds *max_cols* the grid is uniformly down-sampled so it
           always fits in roughly *max_cols* characters wide.

        Example output for ``TokenDAG.linear_chain(8)``::

            ========================================================
              TokenDAG  n=8  edges=7  depth=8  density=0.125
            ========================================================
              Topological levels  (level 0 unmasked first)
              --------------------------------------------------------
              Lv  0 |#####...................................|    1  [0]
              Lv  1 |#####...................................|    1  [1]
              ...
              Adjacency matrix  (# = edge i->j,  . = no edge)
              --------------------------------------------------------
                   0       (display column index)
                   01234567
                   --------
               0  |.#......
               1  |..#.....
               ...
            ========================================================
        """
        n      = self._seq_len
        edges  = self.num_edges()
        adj_cpu = self._adjacency.cpu()

        try:
            levels = self.topological_levels()
            depth  = len(levels)
        except ValueError:
            return f"TokenDAG(n={n}, edges={edges}, INVALID - contains a cycle)"

        density = edges / max(n * (n - 1), 1)
        W    = max(max_cols + 12, 56)   # total display width
        SEP  = "=" * W
        THIN = "-" * W
        lines = [SEP]

        # ── 1. Header ────────────────────────────────────────────────────
        lines.append(
            f"  TokenDAG  n={n}  edges={edges}  depth={depth}"
            f"  density={density:.3f}"
        )
        lines.append(SEP)

        # ── 2. Level bar-chart ───────────────────────────────────────────
        lines.append("  Topological levels  (level 0 unmasked first)")
        lines.append("  " + THIN)

        bar_max = max(W - 34, 8)      # chars available for the bar body
        for i, level in enumerate(levels):
            k       = len(level)
            # Clamp so bar_fill is always in [1, bar_max]
            bar_fill = min(bar_max, max(1, round(bar_max * k / n)))
            bar      = "#" * bar_fill + "." * (bar_max - bar_fill)

            # Compact position summary
            if k <= 8:
                pos_s = "[" + " ".join(str(p) for p in level) + "]"
            elif k <= 16:
                head  = " ".join(str(p) for p in level[:4])
                pos_s = "[" + head + " .. " + str(level[-1]) + "]"
            else:
                pos_s = f"[{level[0]}..{level[-1]}, x{k}]"

            lines.append(f"  Lv {i:2d} |{bar}|  {k:4d}  {pos_s}")

        # ── 3. Adjacency heatmap ─────────────────────────────────────────
        lines.append("")
        lines.append("  Adjacency matrix  (# = edge i->j,  . = no edge)")
        lines.append("  " + THIN)

        # Down-sample uniformly so grid fits in max_cols columns
        stride = max(1, -(-n // max_cols))   # == ceil(n / max_cols)
        pts    = list(range(0, n, stride))
        m      = len(pts)

        # Width of the row-label field, e.g. "  255 |" -> label_w=3
        label_w = len(str(pts[-1])) if pts else 1
        # How many chars appear before the cell data on every row line:
        #   "  " + label_w chars + " |" = label_w + 4
        row_prefix_w = label_w + 4
        indent = " " * row_prefix_w   # same indent for ruler lines

        # Ruler: show the *display* column index (not the position value)
        # so the header always lines up exactly with the cell characters.
        ruler_tens = "".join(
            str((ci // 10) % 10) if ci % 10 == 0 else " "
            for ci in range(m)
        )
        ruler_ones = "".join(str(ci % 10) for ci in range(m))

        lines.append(f"{indent}{ruler_tens}  (display column index)")
        lines.append(f"{indent}{ruler_ones}")
        lines.append(f"{indent}{'-' * m}")

        for r in pts:
            cells = "".join(
                "#" if adj_cpu[r, c].item() else "."
                for c in pts
            )
            lines.append(f"  {r:{label_w}d} |{cells}")

        if stride > 1:
            lines.append(
                f"  (down-sampled: each cell = {stride} positions;"
                f" {n} total, {m} shown)"
            )

        lines.append(SEP)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_visual_str()

    def __repr__(self) -> str:
        return f"TokenDAG(seq_len={self._seq_len}, edges={self.num_edges()}, depth={self.depth()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenDAG):
            return False
        return torch.equal(self._adjacency, other._adjacency)
