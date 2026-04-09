"""Add feedback (benchmark scores, human ratings, Elo) to DAG Library entries.

Three feedback modes
---------------------
  auto    Update a DAGEntry with benchmark evaluation metrics
          (accuracy, perplexity, …).

  human   Record a human quality rating (1–5 scale) for an entry.

  elo     Run a head-to-head Elo tournament between library entries.

Usage
-----
# Record auto eval results for an entry
python scripts/add_feedback.py auto \\
    --db_path library/dags.db \\
    --entry_id abc123 \\
    --benchmark gsm8k \\
    --accuracy 0.742 \\
    --perplexity 12.3

# Add human rating
python scripts/add_feedback.py human \\
    --db_path library/dags.db \\
    --entry_id abc123 \\
    --rater_id researcher1 \\
    --score 4.0

# Elo match between two entries (entry_a wins)
python scripts/add_feedback.py elo \\
    --db_path library/dags.db \\
    --entry_a abc123 \\
    --entry_b def456 \\
    --outcome 1.0

# Elo tournament: run all pairs from a JSON matchup list
# matchups.json: [[\"id_a\", \"id_b\", outcome_a], ...]
python scripts/add_feedback.py elo \\
    --db_path library/dags.db \\
    --tournament_json matchups.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _open_store(db_path: str):
    from dllm_reason.library.store import DAGStore
    from dllm_reason.library.config import StoreConfig
    return DAGStore(config=StoreConfig(db_path=db_path))


def cmd_auto(args: argparse.Namespace) -> None:
    from dllm_reason.library.feedback import AutoFeedback

    store = _open_store(args.db_path)
    entry = store.get(args.entry_id)
    if entry is None:
        print(f"Entry not found: {args.entry_id}"); sys.exit(1)

    metrics: dict[str, float] = {}
    if args.accuracy    is not None: metrics["accuracy"]    = args.accuracy
    if args.perplexity  is not None: metrics["perplexity"]  = args.perplexity
    if args.f1          is not None: metrics["f1"]          = args.f1
    if args.exact_match is not None: metrics["exact_match"] = args.exact_match
    if args.extra_metrics:
        for kv in args.extra_metrics:
            k, v = kv.split("=")
            metrics[k] = float(v)

    handler = AutoFeedback(metric="accuracy")
    updated = handler.update(entry, store, benchmark=args.benchmark, metrics=metrics)
    print(f"Updated  {updated.entry_id}  [{args.benchmark}]  {metrics}")


def cmd_human(args: argparse.Namespace) -> None:
    from dllm_reason.library.feedback import HumanFeedback

    store = _open_store(args.db_path)
    entry = store.get(args.entry_id)
    if entry is None:
        print(f"Entry not found: {args.entry_id}"); sys.exit(1)

    handler = HumanFeedback()
    updated = handler.update(entry, store,
                             rater_id=args.rater_id, score=args.score)
    avg = updated.avg_human_rating()
    print(f"Recorded rating  {args.score}  from {args.rater_id!r}  "
          f"(avg={avg:.2f}  n={len(updated.human_ratings)})")


def cmd_elo(args: argparse.Namespace) -> None:
    from dllm_reason.library.feedback import EloFeedback

    store   = _open_store(args.db_path)
    handler = EloFeedback(k=args.k_factor)

    if args.tournament_json:
        # Batch tournament mode
        with open(args.tournament_json) as f:
            matchups_raw = json.load(f)

        # Build id → index map for run_tournament
        all_ids = list({m[0] for m in matchups_raw} | {m[1] for m in matchups_raw})
        entries = [store.get(i) for i in all_ids]
        missing = [all_ids[i] for i, e in enumerate(entries) if e is None]
        if missing:
            print(f"Missing entries: {missing}"); sys.exit(1)

        idx = {eid: i for i, eid in enumerate(all_ids)}
        outcomes = [(idx[m[0]], idx[m[1]], float(m[2])) for m in matchups_raw]
        updated_entries = handler.run_tournament(entries, outcomes, store)
        for e in updated_entries:
            print(f"  {e.entry_id}  elo={e.elo_rating:.1f}")
        print(f"\nTournament complete  ({len(outcomes)} matchups)")
    else:
        # Single match
        entry_a = store.get(args.entry_a)
        entry_b = store.get(args.entry_b)
        if entry_a is None: print(f"Not found: {args.entry_a}"); sys.exit(1)
        if entry_b is None: print(f"Not found: {args.entry_b}"); sys.exit(1)

        old_a, old_b = entry_a.elo_rating, entry_b.elo_rating
        new_a, new_b = handler.update_pair(entry_a, entry_b, args.outcome, store)
        print(f"  {args.entry_a}  {old_a:.1f} → {new_a:.1f}")
        print(f"  {args.entry_b}  {old_b:.1f} → {new_b:.1f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add feedback to DAG Library entries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db_path", default="library/dags.db")
    sub = p.add_subparsers(dest="mode", required=True)

    # auto
    a = sub.add_parser("auto",  help="Record benchmark evaluation results")
    a.add_argument("--entry_id",     required=True)
    a.add_argument("--benchmark",    default="gsm8k")
    a.add_argument("--accuracy",     type=float, default=None)
    a.add_argument("--perplexity",   type=float, default=None)
    a.add_argument("--f1",           type=float, default=None)
    a.add_argument("--exact_match",  type=float, default=None)
    a.add_argument("--extra_metrics",nargs="*",  default=[],
                   help="Additional key=value metrics")

    # human
    h = sub.add_parser("human", help="Record a human quality rating")
    h.add_argument("--entry_id",  required=True)
    h.add_argument("--rater_id",  default="anonymous")
    h.add_argument("--score",     type=float, required=True,
                   help="Rating score (1.0 – 5.0)")

    # elo
    e = sub.add_parser("elo", help="Update Elo ratings (single match or tournament)")
    e.add_argument("--entry_a",         default=None, help="Winner/entry-A ID")
    e.add_argument("--entry_b",         default=None, help="Loser/entry-B ID")
    e.add_argument("--outcome",         type=float, default=1.0,
                   help="Outcome for entry_a: 1.0=win, 0.5=draw, 0.0=loss")
    e.add_argument("--tournament_json", default=None,
                   help="JSON file with [[id_a, id_b, outcome_a], …] matchups")
    e.add_argument("--k_factor",        type=float, default=32.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    {"auto": cmd_auto, "human": cmd_human, "elo": cmd_elo}[args.mode](args)


if __name__ == "__main__":
    main()
