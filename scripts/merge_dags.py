"""Merge multiple DAG adjacency tensors into a single consensus DAG.

Three merge strategies
-----------------------
  union         Include any edge present in at least one input DAG
  intersection  Include only edges present in ALL input DAGs
  weighted      Weighted-vote: include edges weighted-sum ≥ threshold

Input DAGs can be provided as:
  - .pt files (adjacency tensors)
  - Named templates
  - DAG Library entry IDs

Usage
-----
# Union of two search results
python scripts/merge_dags.py \\
    --inputs results/run1/best_dag.pt results/run2/best_dag.pt \\
    --method union \\
    --output merged_dag.pt

# Weighted merge with custom weights
python scripts/merge_dags.py \\
    --inputs results/run1/best_dag.pt results/run2/best_dag.pt \\
    --method weighted --weights 0.7 0.3 --threshold 0.5 \\
    --output merged_dag.pt

# Merge named templates
python scripts/merge_dags.py \\
    --templates cot skeleton \\
    --seq_len 256 \\
    --method union \\
    --output data/merged_cot_skeleton.pt

# Merge library entries (by ID)
python scripts/merge_dags.py \\
    --library_db library/dags.db \\
    --entry_ids abc123 def456 ghi789 \\
    --method weighted \\
    --output merged.pt \\
    --add_to_library
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def load_dag_adj(path: str) -> torch.Tensor:
    from dllm_reason.graph.dag import TokenDAG
    adj = torch.load(path, map_location="cpu")
    if adj.dim() == 1:
        n = int(adj.numel() ** 0.5)
        adj = adj.reshape(n, n)
    return adj.float()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge multiple DAG adjacency tensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs",    nargs="+", help=".pt adjacency files")
    src.add_argument("--templates", nargs="+", help="Named templates to merge")
    src.add_argument("--entry_ids", nargs="+", help="Library entry IDs to merge")

    p.add_argument("--seq_len",  type=int, default=256,
                   help="Sequence length (for template generation)")
    p.add_argument("--method",   default="weighted",
                   choices=["union", "intersection", "weighted"])
    p.add_argument("--weights",  nargs="+", type=float, default=None,
                   help="Per-DAG weights for weighted merge (default: uniform)")
    p.add_argument("--threshold",type=float, default=0.5,
                   help="Edge inclusion threshold for weighted merge")

    p.add_argument("--output",   required=True, help="Output .pt file path")
    p.add_argument("--visualize",action="store_true", help="Save a PNG of the result")

    p.add_argument("--library_db",     default="library/dags.db",
                   help="Library DB (required for --entry_ids)")
    p.add_argument("--add_to_library", action="store_true",
                   help="Save merged DAG to the library")
    p.add_argument("--task_description", default="Merged DAG")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load input adjacency tensors ──────────────────────────────────────
    adjs: list[torch.Tensor] = []

    if args.inputs:
        for path in args.inputs:
            adjs.append(load_dag_adj(path))

    elif args.templates:
        from dllm_reason.graph.templates import build_template
        for name in args.templates:
            dag = build_template(name, args.seq_len)
            adjs.append(dag.adjacency.float())

    elif args.entry_ids:
        from dllm_reason.library.store import DAGStore
        from dllm_reason.library.config import StoreConfig
        store = DAGStore(config=StoreConfig(db_path=args.library_db))
        for eid in args.entry_ids:
            entry = store.get(eid)
            if entry is None:
                print(f"Entry not found: {eid}"); sys.exit(1)
            dag = entry.to_token_dag()
            adjs.append(dag.adjacency.float())

    if not adjs:
        print("No input DAGs provided."); sys.exit(1)

    # Ensure all same size
    sizes = [a.shape for a in adjs]
    if len(set(sizes)) > 1:
        print(f"Adjacency size mismatch: {sizes}"); sys.exit(1)

    # ── Build DAGEntry list and scores for merger ─────────────────────────
    from dllm_reason.library.entry import DAGEntry
    from dllm_reason.graph.dag import TokenDAG

    entries = [
        DAGEntry.from_token_dag(TokenDAG(adj.bool()), source="manual")
        for adj in adjs
    ]
    weights = args.weights or [1.0 / len(adjs)] * len(adjs)
    if len(weights) != len(adjs):
        print(f"--weights length ({len(weights)}) != number of inputs ({len(adjs)})")
        sys.exit(1)

    # ── Merge ─────────────────────────────────────────────────────────────
    from dllm_reason.library.merge import UnionMerger, IntersectionMerger, WeightedMerger

    if args.method == "union":
        merger = UnionMerger()
    elif args.method == "intersection":
        merger = IntersectionMerger()
    else:
        merger = WeightedMerger(threshold=args.threshold)

    merged_adj = merger.merge(entries, weights)  # (N, N) bool tensor

    # ── Print stats ───────────────────────────────────────────────────────
    from dllm_reason.eval.dag_analysis import analyze_dag
    merged_dag = TokenDAG(merged_adj)
    stats = analyze_dag(merged_dag)
    print(f"\n  Method     : {args.method}")
    print(f"  Inputs     : {len(adjs)}")
    print(f"  Seq len    : {stats.seq_len}")
    print(f"  Edges      : {stats.num_edges}")
    print(f"  Depth      : {stats.depth}")
    print(f"  Density    : {stats.density:.4f}\n")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_adj.cpu(), out)
    print(f"Saved merged adjacency → {out}")

    if args.visualize:
        try:
            from dllm_reason.graph.viz import draw_dag
            png = out.with_suffix(".png")
            draw_dag(merged_dag, title=f"Merged ({args.method})", save_path=png)
            print(f"Visualization        → {png}")
        except Exception as e:
            print(f"[WARN] Visualization failed: {e}")

    if args.add_to_library:
        from dllm_reason.library.store import DAGStore
        from dllm_reason.library.config import StoreConfig
        store = DAGStore(config=StoreConfig(db_path=args.library_db))
        entry = DAGEntry.from_token_dag(
            merged_dag,
            task_description=args.task_description,
            source="merged",
        )
        store.add(entry)
        print(f"Added to library     → {entry.entry_id}")


if __name__ == "__main__":
    main()
