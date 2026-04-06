"""DAG visualization and analysis script.

Visualizes TokenDAG structures, compares their properties,
and generates figures suitable for inclusion in a paper.

Usage:
    # Visualize all template DAGs:
    python scripts/visualize_dag.py \
        --mode templates \
        --seq_len 32 \
        --output_dir figures/dags

    # Visualize a learned DAG from search:
    python scripts/visualize_dag.py \
        --mode learned \
        --adjacency results/dag_search/best_dag_adjacency.pt \
        --output_dir figures/dags

    # Compare DAGs by structure:
    python scripts/visualize_dag.py \
        --mode compare \
        --seq_len 32 \
        --output_dir figures/dags

    # Generate unmasking timeline figures:
    python scripts/visualize_dag.py \
        --mode timeline \
        --seq_len 32 \
        --num_steps 32
"""

import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DAG structures")
    parser.add_argument("--mode", type=str, default="templates",
                        choices=["templates", "learned", "compare", "timeline"])
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--cot_steps", type=int, default=4)
    parser.add_argument("--adjacency", type=str, default=None,
                        help="Path to saved adjacency .pt file (for --mode learned)")
    parser.add_argument("--output_dir", type=str, default="figures/dags")
    return parser.parse_args()


def build_all_template_dags(seq_len: int, cot_steps: int):
    """Build all template DAGs for comparison."""
    from dllm_reason.graph.dag import TokenDAG
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag, interleaved_dag,
    )

    dags = {
        "Empty (Random)": TokenDAG.empty(seq_len),
        "Linear Chain": TokenDAG.linear_chain(seq_len),
        f"CoT ({cot_steps} steps)": chain_of_thought_dag(seq_len, cot_steps),
        "Skeleton-then-Detail": skeleton_then_detail_dag(
            seq_len,
            skeleton_positions=list(range(0, seq_len, 3)),
            detail_positions=list(range(1, seq_len, 3)),
        ),
        "Bidirectional": bidirectional_dag(seq_len, num_segments=4),
        "Interleaved": interleaved_dag(seq_len, num_groups=3),
    }
    return dags


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from dllm_reason.graph.viz import draw_dag, draw_unmasking_timeline
    from dllm_reason.eval.dag_analysis import (
        compare_dags, plot_level_distribution, plot_unmasking_heatmap,
    )

    if args.mode == "templates":
        dags = build_all_template_dags(args.seq_len, args.cot_steps)

        # Draw each DAG structure
        for name, dag in dags.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            fig = draw_dag(dag, title=name, figsize=(10, 6))
            fig.savefig(output_dir / f"dag_{safe_name}.png", dpi=150, bbox_inches="tight")
            print(f"Saved: dag_{safe_name}.png")
            import matplotlib.pyplot as plt
            plt.close(fig)

        # Compare all stats
        compare_dags(dags, save_path=output_dir / "dag_comparison.json")

        # Level distribution
        fig = plot_level_distribution(dags, figsize=(14, 4))
        fig.savefig(output_dir / "level_distributions.png", dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("Saved: level_distributions.png")

    elif args.mode == "timeline":
        dags = build_all_template_dags(args.seq_len, args.cot_steps)
        import matplotlib.pyplot as plt
        for name, dag in dags.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            fig = plot_unmasking_heatmap(dag, args.num_steps, title=name)
            fig.savefig(output_dir / f"timeline_{safe_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: timeline_{safe_name}.png")

    elif args.mode == "learned":
        if args.adjacency is None:
            print("ERROR: --adjacency required for --mode learned")
            return
        import matplotlib.pyplot as plt
        adj = torch.load(args.adjacency, map_location="cpu")
        from dllm_reason.graph.dag import TokenDAG
        dag = TokenDAG(adj)

        fig = draw_dag(dag, title="Learned DAG")
        fig.savefig(output_dir / "learned_dag.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_unmasking_heatmap(dag, args.num_steps, title="Learned DAG")
        fig.savefig(output_dir / "learned_dag_timeline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        from dllm_reason.eval.dag_analysis import analyze_dag
        stats = analyze_dag(dag)
        print(f"\nLearned DAG stats:\n{stats}")

    elif args.mode == "compare":
        # Comparison with learned DAG if provided
        dags = build_all_template_dags(args.seq_len, args.cot_steps)
        if args.adjacency:
            adj = torch.load(args.adjacency, map_location="cpu")
            from dllm_reason.graph.dag import TokenDAG
            dags["Learned (Search)"] = TokenDAG(adj)

        stats = compare_dags(dags, save_path=output_dir / "dag_comparison.json")
        print(f"\nAll stats saved to {output_dir}/dag_comparison.json")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
