"""DAG Library management: add, list, retrieve, delete, stats.

The DAG Library (DAGStore) is a SQLite-backed persistent registry of
TokenDAG structures with metadata, benchmark scores, and Elo ratings.

Usage
-----
# Show library statistics
python scripts/manage_library.py stats --db_path library/dags.db

# List top-10 entries by Elo rating
python scripts/manage_library.py list --db_path library/dags.db --sort elo --top 10

# Add a named template DAG to the library
python scripts/manage_library.py add \\
    --db_path library/dags.db \\
    --source template \\
    --template_name cot \\
    --seq_len 256 \\
    --task_description "Chain-of-thought reasoning for math problems"

# Add a DAG from a saved adjacency tensor
python scripts/manage_library.py add \\
    --db_path library/dags.db \\
    --adjacency_pt results/dag_search/best_dag_adjacency.pt \\
    --source search \\
    --search_method evolutionary \\
    --task_description "Evolved DAG for GSM8K"

# Get a specific entry by ID
python scripts/manage_library.py get --db_path library/dags.db --entry_id abc123

# Delete an entry
python scripts/manage_library.py delete --db_path library/dags.db --entry_id abc123

# Export all entries to JSON
python scripts/manage_library.py export \\
    --db_path library/dags.db \\
    --out library/export.json

# Retrieve entries by query (semantic / structural / performance)
python scripts/manage_library.py retrieve \\
    --db_path library/dags.db \\
    --mode semantic \\
    --query "multi-hop reasoning over structured knowledge" \\
    --top_k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _open_store(db_path: str):
    from dllm_reason.library.store import DAGStore
    from dllm_reason.library.config import StoreConfig
    cfg = StoreConfig(db_path=db_path)
    return DAGStore(config=cfg)


def _print_entry(entry, verbose: bool = False) -> None:
    from dllm_reason.graph.dag import TokenDAG
    print(f"  id          : {entry.entry_id}")
    print(f"  source      : {entry.source}"
          + (f" / {entry.template_name}" if entry.template_name else "")
          + (f" / {entry.search_method}"  if entry.search_method  else ""))
    print(f"  seq_len     : {entry.seq_len}")
    print(f"  edges       : {entry.num_edges}  depth={entry.depth}")
    print(f"  elo         : {entry.elo_rating:.1f}")
    print(f"  description : {entry.task_description[:80] or '—'}")
    if entry.benchmark_scores:
        for bm, metrics in entry.benchmark_scores.items():
            m_str = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            print(f"  benchmark   : [{bm}]  {m_str}")
    if entry.tags:
        print(f"  tags        : {', '.join(entry.tags)}")
    if verbose:
        dag_preview = entry.adjacency[:16]
        print(f"  adjacency   : {dag_preview}{'...' if len(entry.adjacency) > 16 else ''}")


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_stats(args: argparse.Namespace) -> None:
    store = _open_store(args.db_path)
    entries = store.list_all(limit=100_000)
    n = len(entries)
    if n == 0:
        print("Library is empty.")
        return

    elos       = [e.elo_rating for e in entries]
    edges_list = [e.num_edges   for e in entries]
    sources    = {}
    for e in entries:
        sources[e.source] = sources.get(e.source, 0) + 1

    print(f"\n{'─'*48}")
    print(f"  DAG Library  —  {args.db_path}")
    print(f"{'─'*48}")
    print(f"  entries      : {n}")
    print(f"  avg elo      : {sum(elos)/n:.1f}")
    print(f"  avg edges    : {sum(edges_list)/n:.1f}")
    print(f"  max elo      : {max(elos):.1f}")
    print(f"  sources      : {json.dumps(sources)}")
    print(f"{'─'*48}\n")


def cmd_list(args: argparse.Namespace) -> None:
    store = _open_store(args.db_path)

    if args.sort == "elo":
        entries = store.top_by_elo(k=args.top)
    else:
        entries = store.list_all(limit=args.top)
        if args.sort == "edges":
            entries.sort(key=lambda e: e.num_edges, reverse=True)
        elif args.sort == "date":
            entries.sort(key=lambda e: e.created_at, reverse=True)

    if not entries:
        print("Library is empty.")
        return

    print(f"\n  {'ID':<16}  {'SOURCE':<18}  {'SEQ':<6}  "
          f"{'EDGES':<7}  {'DEPTH':<6}  {'ELO':<8}  DESCRIPTION")
    print(f"  {'─'*16}  {'─'*18}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*30}")
    for e in entries:
        src = f"{e.source}"
        if e.template_name:  src += f"/{e.template_name}"
        if e.search_method:  src += f"/{e.search_method}"
        desc = (e.task_description or "")[:30]
        print(f"  {e.entry_id:<16}  {src:<18}  {e.seq_len:<6}  "
              f"{e.num_edges:<7}  {e.depth:<6}  {e.elo_rating:<8.1f}  {desc}")
    print()


def cmd_get(args: argparse.Namespace) -> None:
    store  = _open_store(args.db_path)
    entry  = store.get(args.entry_id)
    if entry is None:
        print(f"Entry not found: {args.entry_id}")
        sys.exit(1)
    _print_entry(entry, verbose=True)


def cmd_add(args: argparse.Namespace) -> None:
    import torch
    from dllm_reason.library.entry import DAGEntry
    from dllm_reason.graph.dag import TokenDAG

    store = _open_store(args.db_path)

    if args.adjacency_pt:
        adj = torch.load(args.adjacency_pt, map_location="cpu")
        if adj.dim() == 2:
            adj = adj.bool()
        dag = TokenDAG(adj)
        entry = DAGEntry.from_token_dag(
            dag,
            task_description=args.task_description,
            source=args.source or "search",
            search_method=args.search_method,
        )
    elif args.template_name:
        from dllm_reason.graph.templates import build_template
        dag = build_template(args.template_name, args.seq_len)
        entry = DAGEntry.from_token_dag(
            dag,
            task_description=args.task_description or f"Template: {args.template_name}",
            source="template",
            template_name=args.template_name,
        )
    else:
        print("Provide --adjacency_pt or --template_name.")
        sys.exit(1)

    if args.tags:
        entry.tags = args.tags

    store.add(entry)
    print(f"Added entry  {entry.entry_id}  "
          f"(edges={entry.num_edges}, depth={entry.depth})")


def cmd_delete(args: argparse.Namespace) -> None:
    store = _open_store(args.db_path)
    ok = store.delete(args.entry_id)
    if ok:
        print(f"Deleted {args.entry_id}")
    else:
        print(f"Entry not found: {args.entry_id}")
        sys.exit(1)


def cmd_export(args: argparse.Namespace) -> None:
    store   = _open_store(args.db_path)
    entries = store.list_all(limit=100_000)
    data    = [e.to_dict() for e in entries]
    out     = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(data)} entries → {out}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    from dllm_reason.library.retrieval import (
        RetrievalQuery, SemanticRetrieval, StructuralRetrieval,
        PerformanceRetrieval, RetrievalMode,
    )

    store = _open_store(args.db_path)

    query = RetrievalQuery(
        task_description=args.query,
        target_benchmark=args.benchmark,
        target_metric=args.metric,
    )

    if args.mode == "semantic":
        from dllm_reason.library.embedder import create_embedder
        embedder = create_embedder(args.embedder)
        channel  = SemanticRetrieval(embedder=embedder)
    elif args.mode == "structural":
        channel = StructuralRetrieval(metric=args.struct_metric)
    else:
        channel = PerformanceRetrieval()

    results = channel.retrieve(query, store, top_k=args.top_k)
    if not results:
        print("No results.")
        return

    print(f"\n  Top-{len(results)} results [{args.mode}]:")
    print(f"  {'─'*60}")
    for rank, (entry, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}]  {entry.entry_id:<16}  "
              f"{entry.source:<14}  {entry.task_description[:40]}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manage the dLLM-Reason DAG Library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db_path", default="library/dags.db",
                   help="Path to DAGStore SQLite database")

    sub = p.add_subparsers(dest="cmd", required=True)

    # stats
    sub.add_parser("stats", help="Print library statistics")

    # list
    lst = sub.add_parser("list", help="List entries")
    lst.add_argument("--sort", default="elo",
                     choices=["elo", "edges", "date"], help="Sort order")
    lst.add_argument("--top", type=int, default=20, help="Max entries to show")

    # get
    get = sub.add_parser("get", help="Show a single entry by ID")
    get.add_argument("--entry_id", required=True)

    # add
    add = sub.add_parser("add", help="Add a DAG entry to the library")
    grp = add.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adjacency_pt",  help=".pt file with adjacency tensor")
    grp.add_argument("--template_name", help="Named template (cot, skeleton, …)")
    add.add_argument("--seq_len",         type=int, default=256)
    add.add_argument("--task_description", default="")
    add.add_argument("--source",          default="manual",
                     choices=["template", "search", "manual", "merged"])
    add.add_argument("--search_method",   default=None)
    add.add_argument("--tags",            nargs="*", default=[])

    # delete
    d = sub.add_parser("delete", help="Delete an entry by ID")
    d.add_argument("--entry_id", required=True)

    # export
    exp = sub.add_parser("export", help="Export all entries to JSON")
    exp.add_argument("--out", default="library/export.json")

    # retrieve
    ret = sub.add_parser("retrieve", help="Query library with a retrieval channel")
    ret.add_argument("--mode", default="semantic",
                     choices=["semantic", "structural", "performance"])
    ret.add_argument("--query",       default="", help="Text query (semantic mode)")
    ret.add_argument("--benchmark",   default=None)
    ret.add_argument("--metric",      default="accuracy")
    ret.add_argument("--top_k",       type=int, default=5)
    ret.add_argument("--embedder",    default="tfidf",
                     choices=["tfidf", "random", "sentence_transformer"],
                     help="Embedder type for semantic retrieval")
    ret.add_argument("--struct_metric", default="edit_distance",
                     choices=["edit_distance", "spectral"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    dispatch = {
        "stats":    cmd_stats,
        "list":     cmd_list,
        "get":      cmd_get,
        "add":      cmd_add,
        "delete":   cmd_delete,
        "export":   cmd_export,
        "retrieve": cmd_retrieve,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
