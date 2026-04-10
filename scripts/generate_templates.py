"""Generate, visualize, and save named DAG templates.

Builds one or more TokenDAG templates by name and saves the adjacency
tensors to disk.  Optionally renders visualizations and prints structural
statistics for each template.

Available templates
-------------------
  cot            Chain-of-Thought: sequential reasoning blocks
  answer_first   Answer region first, then fill reasoning
  skeleton        Skeleton-then-Detail: structure → details
  bidirectional   Unmask from both ends toward center
  interleaved     Alternating reasoning / answer tokens
  linear          Strict left-to-right (autoregressive simulation)
  random_low      Random sparse DAG (~5 % density)
  random_high     Random dense DAG (~20 % density)

Usage
-----
# Save all templates at seq_len=256
python scripts/generate_templates.py \\
    --seq_len 256 \\
    --output_dir data/templates

# Save specific templates only
python scripts/generate_templates.py \\
    --templates cot skeleton bidirectional \\
    --seq_len 128 \\
    --output_dir data/templates

# Generate + visualize
python scripts/generate_templates.py \\
    --seq_len 128 \\
    --visualize \\
    --output_dir data/templates

# Print stats only, no files saved
python scripts/generate_templates.py --seq_len 64 --stats_only

# Add all generated templates to the DAG Library
python scripts/generate_templates.py \\
    --seq_len 256 \\
    --add_to_library \\
    --library_db library/dags.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dllm_reason.graph.templates import build_all_templates, TEMPLATE_NAMES
from dllm_reason.eval.dag_analysis import analyze_dag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate and save named DAG templates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--templates", nargs="+", default=None,
                   choices=TEMPLATE_NAMES,
                   help=f"Templates to generate (default: all {len(TEMPLATE_NAMES)})")
    p.add_argument("--seq_len",    type=int, default=256)
    p.add_argument("--output_dir", default="data/templates",
                   help="Directory for .pt adjacency tensors")

    p.add_argument("--visualize",  action="store_true",
                   help="Save PNG visualizations for each template")
    p.add_argument("--stats_only", action="store_true",
                   help="Print statistics only; do not save files")

    p.add_argument("--add_to_library", action="store_true",
                   help="Insert generated templates into the DAG Library")
    p.add_argument("--library_db", default="library/dags.db",
                   help="DAGStore path (used with --add_to_library)")

    return p.parse_args()


def main() -> None:
    args     = parse_args()
    names    = args.templates or TEMPLATE_NAMES
    out_dir  = Path(args.output_dir)

    if not args.stats_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    templates = build_all_templates(seq_len=args.seq_len, names=names)

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n  {'TEMPLATE':<20}  {'EDGES':>7}  {'DEPTH':>6}  "
          f"{'DENSITY':>8}  {'ROOTS':>6}  {'LEAVES':>6}")
    print(f"  {'─'*20}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}")

    store = None
    if args.add_to_library:
        from dllm_reason.library.store import DAGStore
        from dllm_reason.library.config import StoreConfig
        from dllm_reason.library.entry import DAGEntry
        store = DAGStore(config=StoreConfig(db_path=args.library_db))

    for name, dag in templates.items():
        stats = analyze_dag(dag)
        print(f"  {name:<20}  {stats.num_edges:>7}  {stats.depth:>6}  "
              f"{stats.density:>8.4f}  {stats.num_roots:>6}  {stats.num_leaves:>6}")

        if args.stats_only:
            continue

        # Save adjacency tensor
        pt_path = out_dir / f"{name}_seq{args.seq_len}.pt"
        torch.save(dag.adjacency.cpu(), pt_path)

        # Save visualization
        if args.visualize:
            try:
                from dllm_reason.graph.viz import draw_dag
                draw_dag(dag, title=name,
                         save_path=out_dir / f"{name}_seq{args.seq_len}.png")
            except Exception as e:
                print(f"    [WARN] Visualization failed for {name}: {e}")

        # Add to library
        if store is not None:
            entry = DAGEntry.from_token_dag(
                dag,
                task_description=f"Named template: {name}  (seq_len={args.seq_len})",
                source="template",
                template_name=name,
            )
            store.add(entry)

    if args.stats_only:
        print()
        return

    print(f"\n  Saved {len(templates)} template(s) → {out_dir}/")
    if args.visualize:
        print(f"  PNG visualizations saved in same directory.")
    if args.add_to_library:
        print(f"  Templates added to DAG Library: {args.library_db}")
    print()


if __name__ == "__main__":
    main()
