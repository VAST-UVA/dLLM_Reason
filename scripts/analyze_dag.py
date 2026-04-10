"""Analyze and visualize DAG structure from a file or named template.

Computes structural statistics (depth, density, edge count, level sizes) and
generates visualizations: adjacency heatmap, unmasking timeline, and level
distribution.

Usage
-----
# Analyze a saved adjacency tensor
python scripts/analyze_dag.py \\
    --adjacency_pt results/dag_search/best_dag_adjacency.pt \\
    --output_dir results/dag_analysis

# Analyze a named template
python scripts/analyze_dag.py --template cot --seq_len 256

# Compare multiple templates
python scripts/analyze_dag.py \\
    --templates cot skeleton bidirectional answer_first \\
    --seq_len 128 \\
    --output_dir results/template_comparison

# Plot unmasking timeline for a specific DAG
python scripts/analyze_dag.py \\
    --adjacency_pt results/best_dag_adjacency.pt \\
    --plot_timeline --num_steps 32

# Analyze a DAG from the library
python scripts/analyze_dag.py \\
    --library_db library/dags.db \\
    --entry_id abc123def456
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dllm_reason.eval.dag_analysis import (
    analyze_dag,
    compare_dags,
    plot_unmasking_heatmap,
    plot_level_distribution,
)


def load_dag_from_pt(path: str) -> "TokenDAG":
    from dllm_reason.graph.dag import TokenDAG
    adj = torch.load(path, map_location="cpu")
    if adj.dim() == 1:
        n = int(adj.numel() ** 0.5)
        adj = adj.reshape(n, n)
    return TokenDAG(adj.bool())


def load_dag_from_template(name: str, seq_len: int) -> "TokenDAG":
    from dllm_reason.graph.templates import build_template
    return build_template(name, seq_len)


def load_dag_from_library(db_path: str, entry_id: str) -> "TokenDAG":
    from dllm_reason.library.store import DAGStore
    from dllm_reason.library.config import StoreConfig
    store = DAGStore(config=StoreConfig(db_path=db_path))
    entry = store.get(entry_id)
    if entry is None:
        raise ValueError(f"Entry not found in library: {entry_id}")
    return entry.to_token_dag()


def print_stats(name: str, stats) -> None:
    print(f"\n  ┌─ {name} {'─'*(50 - len(name))}")
    print(f"  │  seq_len        : {stats.seq_len}")
    print(f"  │  edges          : {stats.num_edges}")
    print(f"  │  depth          : {stats.depth}  (topological levels)")
    print(f"  │  density        : {stats.density:.4f}")
    print(f"  │  avg in-degree  : {stats.avg_in_degree:.2f}")
    print(f"  │  avg out-degree : {stats.avg_out_degree:.2f}")
    print(f"  │  roots/leaves   : {stats.num_roots} / {stats.num_leaves}")
    print(f"  │  avg path len   : {stats.avg_path_length:.2f}")
    print(f"  │  level sizes    : {stats.level_sizes[:8]}"
          + ("…" if len(stats.level_sizes) > 8 else ""))
    print(f"  └{'─'*53}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze and visualize DAG structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--adjacency_pt", help="Path to .pt adjacency tensor")
    src.add_argument("--template",     help="Single named template")
    src.add_argument("--library_db",   help="Library DB path (use with --entry_id)")

    p.add_argument("--entry_id",   help="Entry ID when using --library_db")
    p.add_argument("--seq_len",    type=int, default=256,
                   help="Sequence length for template generation")

    # Multi-template comparison
    p.add_argument("--templates", nargs="+", default=None,
                   help="Multiple template names to compare")

    # Output
    p.add_argument("--output_dir", default="results/dag_analysis")
    p.add_argument("--no_plots",   action="store_true",
                   help="Skip matplotlib figures")

    # Timeline options
    p.add_argument("--plot_timeline", action="store_true",
                   help="Generate unmasking timeline plot")
    p.add_argument("--num_steps", type=int, default=32,
                   help="Steps for timeline plot")

    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Multi-template comparison ──────────────────────────────────────────
    if args.templates:
        from dllm_reason.graph.templates import build_template
        dags = {}
        for name in args.templates:
            try:
                dags[name] = build_template(name, args.seq_len)
            except Exception as e:
                print(f"[WARN] Could not build template {name!r}: {e}")

        all_stats = compare_dags(dags, save_path=out_dir / "comparison_table.txt")
        for name, stats in all_stats.items():
            print_stats(name, stats)

        if not args.no_plots:
            fig = plot_level_distribution(
                dags, save_path=out_dir / "level_distribution.png"
            )
            print(f"\nLevel distribution → {out_dir}/level_distribution.png")

        stats_export = {name: s.to_dict() for name, s in all_stats.items()}
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats_export, f, indent=2)
        print(f"Stats JSON       → {out_dir}/stats.json")
        return

    # ── Single DAG ────────────────────────────────────────────────────────
    if args.adjacency_pt:
        dag  = load_dag_from_pt(args.adjacency_pt)
        name = Path(args.adjacency_pt).stem
    elif args.template:
        dag  = load_dag_from_template(args.template, args.seq_len)
        name = args.template
    elif args.library_db:
        if not args.entry_id:
            print("--entry_id is required with --library_db"); sys.exit(1)
        dag  = load_dag_from_library(args.library_db, args.entry_id)
        name = args.entry_id[:12]
    else:
        print("Provide --adjacency_pt, --template, --templates, or --library_db.")
        sys.exit(1)

    stats = analyze_dag(dag)
    print_stats(name, stats)

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    if not args.no_plots:
        # Adjacency heatmap
        try:
            from dllm_reason.graph.viz import draw_dag
            fig = draw_dag(dag, title=name,
                           save_path=out_dir / f"{name}_dag.png")
            print(f"DAG plot         → {out_dir}/{name}_dag.png")
        except Exception as e:
            print(f"[WARN] DAG plot failed: {e}")

        # Unmasking timeline
        if args.plot_timeline:
            try:
                fig = plot_unmasking_heatmap(
                    dag, num_steps=args.num_steps,
                    title=name,
                    save_path=out_dir / f"{name}_timeline.png",
                )
                print(f"Timeline plot    → {out_dir}/{name}_timeline.png")
            except Exception as e:
                print(f"[WARN] Timeline plot failed: {e}")

    print(f"\nStats saved → {out_dir}/stats.json")


if __name__ == "__main__":
    main()
