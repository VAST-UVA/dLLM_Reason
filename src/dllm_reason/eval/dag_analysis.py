"""Analysis tools for understanding DAG structure and its effect on reasoning.

Provides tools to:
1. Analyze properties of discovered/learned DAGs
2. Visualize the relationship between DAG structure and performance
3. Compare DAG topology across different tasks
4. Ablate DAG components (depth, width, density)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DAGStats:
    """Structural statistics of a TokenDAG."""
    seq_len: int
    num_edges: int
    depth: int                   # Number of topological levels
    density: float               # Edges / possible edges
    avg_in_degree: float
    avg_out_degree: float
    max_in_degree: int
    max_out_degree: int
    num_roots: int               # Positions with no parents (level 0)
    num_leaves: int              # Positions with no children
    avg_path_length: float       # Average shortest path between connected nodes
    level_sizes: list[int]       # Number of positions per topological level

    def to_dict(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "num_edges": self.num_edges,
            "depth": self.depth,
            "density": self.density,
            "avg_in_degree": self.avg_in_degree,
            "avg_out_degree": self.avg_out_degree,
            "max_in_degree": self.max_in_degree,
            "max_out_degree": self.max_out_degree,
            "num_roots": self.num_roots,
            "num_leaves": self.num_leaves,
            "avg_path_length": self.avg_path_length,
            "level_sizes": self.level_sizes,
        }

    def __str__(self) -> str:
        return (
            f"DAGStats(n={self.seq_len}, edges={self.num_edges}, "
            f"depth={self.depth}, density={self.density:.3f}, "
            f"roots={self.num_roots}, leaves={self.num_leaves})"
        )


def analyze_dag(dag: TokenDAG) -> DAGStats:
    """Compute structural statistics of a DAG."""
    G = dag.to_networkx()
    adj = dag.adjacency.cpu()

    seq_len = dag.seq_len
    num_edges = dag.num_edges()
    possible_edges = seq_len * (seq_len - 1)  # Directed, no self-loops
    density = num_edges / max(possible_edges, 1)

    in_degrees = adj.sum(dim=0).float()
    out_degrees = adj.sum(dim=1).float()

    levels = dag.topological_levels()
    level_sizes = [len(level) for level in levels]
    num_roots = level_sizes[0] if level_sizes else 0
    num_leaves = sum(1 for node in G.nodes if G.out_degree(node) == 0)

    # Average shortest path length (only for connected pairs)
    try:
        path_lengths = []
        for source in list(G.nodes)[:min(50, seq_len)]:  # Sample for speed
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, length in lengths.items():
                if target != source:
                    path_lengths.append(length)
        avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
    except Exception:
        avg_path_length = 0.0

    return DAGStats(
        seq_len=seq_len,
        num_edges=num_edges,
        depth=len(levels),
        density=density,
        avg_in_degree=in_degrees.mean().item(),
        avg_out_degree=out_degrees.mean().item(),
        max_in_degree=int(in_degrees.max().item()),
        max_out_degree=int(out_degrees.max().item()),
        num_roots=num_roots,
        num_leaves=num_leaves,
        avg_path_length=avg_path_length,
        level_sizes=level_sizes,
    )


def compare_dags(
    dags: dict[str, TokenDAG],
    save_path: str | Path | None = None,
) -> dict[str, DAGStats]:
    """Compare structural properties of multiple DAGs.

    Prints a comparison table and returns stats.
    """
    stats = {name: analyze_dag(dag) for name, dag in dags.items()}

    # Print table
    print(f"\n{'='*80}")
    print("DAG Structure Comparison")
    print("=" * 80)
    print(f"{'Name':<20} {'Edges':>6} {'Depth':>6} {'Density':>8} {'Roots':>6} {'Leaves':>7} {'AvgPath':>8}")
    print("-" * 80)
    for name, s in stats.items():
        print(
            f"{name:<20} {s.num_edges:>6} {s.depth:>6} "
            f"{s.density:>8.4f} {s.num_roots:>6} {s.num_leaves:>7} "
            f"{s.avg_path_length:>8.2f}"
        )
    print("=" * 80)

    if save_path:
        import json
        with open(save_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in stats.items()}, f, indent=2)

    return stats


def plot_dag_stats_vs_performance(
    stats_list: list[DAGStats],
    accuracies: list[float],
    names: list[str],
    figsize: tuple = (12, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plots of DAG structural properties vs task accuracy.

    Helps understand which structural properties correlate with better reasoning.
    """
    properties = [
        ("depth", "Topological Depth"),
        ("density", "Edge Density"),
        ("avg_in_degree", "Avg In-Degree"),
        ("num_roots", "Number of Roots"),
        ("avg_path_length", "Avg Path Length"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, (prop, prop_label) in enumerate(properties):
        ax = axes[idx]
        xs = [getattr(s, prop) for s in stats_list]
        ys = accuracies

        ax.scatter(xs, ys, alpha=0.7, s=80)
        for x, y, name in zip(xs, ys, names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

        # Trend line
        if len(xs) >= 3:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5)

        ax.set_xlabel(prop_label)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs {prop_label}")
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    if len(properties) < len(axes):
        for ax in axes[len(properties):]:
            ax.set_visible(False)

    plt.suptitle("DAG Structural Properties vs Reasoning Accuracy", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_level_distribution(
    dags: dict[str, TokenDAG],
    figsize: tuple = (12, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the distribution of tokens across topological levels for each DAG."""
    fig, axes = plt.subplots(1, len(dags), figsize=figsize, sharey=False)
    if len(dags) == 1:
        axes = [axes]

    for ax, (name, dag) in zip(axes, dags.items()):
        levels = dag.topological_levels()
        level_sizes = [len(level) for level in levels]
        level_indices = list(range(len(level_sizes)))

        ax.bar(level_indices, level_sizes, color="steelblue", alpha=0.8)
        ax.set_xlabel("Topological Level")
        ax.set_ylabel("# Positions")
        ax.set_title(f"{name}\n(depth={len(levels)}, edges={dag.num_edges()})")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Token Distribution Across Topological Levels", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_unmasking_heatmap(
    dag: TokenDAG,
    num_steps: int = 32,
    title: str = "",
    figsize: tuple = (14, 3),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap showing when each token position is unmasked across steps."""
    schedule = dag.to_mask_schedule(num_steps)
    seq_len = dag.seq_len

    step_map = torch.full((seq_len,), -1, dtype=torch.long)
    for step, positions in enumerate(schedule):
        for pos in positions:
            if pos < seq_len and step_map[pos] == -1:
                step_map[pos] = step

    # Matrix: 1 row, seq_len columns; value = step when unmasked
    matrix = step_map.float().unsqueeze(0).numpy()
    matrix[matrix == -1] = float("nan")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="plasma", interpolation="nearest",
                   vmin=0, vmax=num_steps)
    ax.set_xlabel("Token Position")
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label="Unmasking Step", orientation="horizontal", fraction=0.02)
    ax.set_title(f"Unmasking Timeline: {title} ({num_steps} steps)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def search_history_plot(
    history: list[dict],
    title: str = "DAG Search Progress",
    figsize: tuple = (10, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot fitness vs evaluation steps during DAG search."""
    steps = [h.get("step", i) for i, h in enumerate(history)]
    fitnesses = [h["fitness"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Fitness over steps
    axes[0].plot(steps, fitnesses, "b-", linewidth=1.5)
    axes[0].fill_between(steps, fitnesses, alpha=0.1, color="blue")
    axes[0].set_xlabel("Evaluation Step")
    axes[0].set_ylabel("Fitness (Accuracy)")
    axes[0].set_title("Search Progress")
    axes[0].grid(True, alpha=0.3)

    # Edge count if available
    if "edges" in history[0]:
        edges = [h["edges"] for h in history]
        axes[1].scatter(edges, fitnesses, alpha=0.5, s=20)
        axes[1].set_xlabel("Number of Edges")
        axes[1].set_ylabel("Fitness (Accuracy)")
        axes[1].set_title("Edges vs Fitness")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
