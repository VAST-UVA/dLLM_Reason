"""Visualization utilities for TokenDAG structures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch

from dllm_reason.graph.dag import TokenDAG


def draw_dag(
    dag: TokenDAG,
    tokens: list[str] | None = None,
    title: str = "TokenDAG",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | Path | None = None,
    highlight_levels: bool = True,
    max_display: int = 50,
) -> plt.Figure:
    """Visualize a TokenDAG as a layered graph.

    Args:
        dag: the TokenDAG to visualize
        tokens: optional token strings to label nodes
        title: plot title
        figsize: figure size
        save_path: if given, save to this path
        highlight_levels: color nodes by topological level
        max_display: max positions to display (truncate for large sequences)
    """
    G = dag.to_networkx()

    # Truncate for large graphs
    if dag.seq_len > max_display:
        nodes_to_keep = list(range(max_display))
        G = G.subgraph(nodes_to_keep).copy()

    levels = dag.topological_levels()

    fig, ax = plt.subplots(figsize=figsize)

    # Use dot layout (layered)
    try:
        pos = nx.multipartite_layout(G, subset_key="level")
    except Exception:
        # Assign level attribute for layout
        for level_idx, positions in enumerate(levels):
            for p in positions:
                if p in G.nodes:
                    G.nodes[p]["level"] = level_idx
        try:
            pos = nx.multipartite_layout(G, subset_key="level")
        except Exception:
            pos = nx.spring_layout(G)

    # Color by level
    cmap = plt.cm.Set3
    colors = []
    for node in G.nodes:
        for level_idx, level_positions in enumerate(levels):
            if node in level_positions:
                colors.append(cmap(level_idx / max(1, len(levels) - 1)))
                break
        else:
            colors.append("lightgray")

    # Labels
    if tokens:
        labels = {i: f"{i}:{tokens[i]}" for i in G.nodes if i < len(tokens)}
    else:
        labels = {i: str(i) for i in G.nodes}

    nx.draw_networkx(
        G, pos, ax=ax,
        labels=labels,
        node_color=colors,
        node_size=500,
        font_size=8,
        arrows=True,
        arrowsize=15,
        edge_color="gray",
        alpha=0.9,
    )

    # Legend
    if highlight_levels:
        patches = [
            mpatches.Patch(color=cmap(i / max(1, len(levels) - 1)), label=f"Level {i}")
            for i in range(min(len(levels), 10))
        ]
        ax.legend(handles=patches, loc="upper left", fontsize=8)

    ax.set_title(f"{title} (nodes={dag.seq_len}, edges={dag.num_edges()}, depth={dag.depth()})")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def draw_unmasking_timeline(
    dag: TokenDAG,
    num_steps: int,
    tokens: list[str] | None = None,
    figsize: tuple[int, int] = (14, 4),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualize the unmasking schedule as a timeline heatmap.

    Shows which positions are unmasked at each diffusion step.
    """
    schedule = dag.to_mask_schedule(num_steps)

    # Build a matrix: (num_steps, seq_len), value = step when unmasked
    seq_len = dag.seq_len
    timeline = -torch.ones(seq_len, dtype=torch.long)

    for step, positions in enumerate(schedule):
        for pos in positions:
            if timeline[pos] == -1:
                timeline[pos] = step

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    matrix = timeline.unsqueeze(0).float()
    matrix[matrix == -1] = float("nan")

    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("")
    ax.set_yticks([])

    if tokens and len(tokens) <= 30:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)

    plt.colorbar(im, ax=ax, label="Unmasking Step")
    ax.set_title(f"Unmasking Timeline ({num_steps} steps)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
