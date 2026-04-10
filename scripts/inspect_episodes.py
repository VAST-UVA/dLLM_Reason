"""Inspect, query, and export an EpisodeStore database.

Usage
-----
# Print summary statistics
python scripts/inspect_episodes.py stats --db_path episodes/gsm8k.db

# List episodes (most recent first)
python scripts/inspect_episodes.py list \\
    --db_path episodes/gsm8k.db \\
    --task_type math --limit 20

# Show a single episode in detail
python scripts/inspect_episodes.py get \\
    --db_path episodes/gsm8k.db \\
    --episode_id abc123def456

# Export correct episodes to JSONL for fine-tuning
python scripts/inspect_episodes.py export \\
    --db_path episodes/gsm8k.db \\
    --correct_only \\
    --out data/gsm8k_correct.jsonl

# Delete episodes below a score threshold
python scripts/inspect_episodes.py prune \\
    --db_path episodes/gsm8k.db \\
    --max_score 0.0 \\
    --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dllm_reason.library.episode import DAGEpisode, EpisodeStore


def cmd_stats(args: argparse.Namespace) -> None:
    store = EpisodeStore(args.db_path)
    store.print_stats()


def cmd_list(args: argparse.Namespace) -> None:
    store = EpisodeStore(args.db_path)
    episodes = store.query(
        task_type=args.task_type,
        strategy_name=args.strategy,
        correct=True if args.correct_only else None,
        min_score=args.min_score,
        limit=args.limit,
    )

    if not episodes:
        print("No episodes match the filter.")
        return

    print(f"\n  {'ID':<18}  {'TASK':<10}  {'STRATEGY':<16}  "
          f"{'CORRECT':<8}  {'SCORE':>6}  PROMPT")
    print(f"  {'─'*18}  {'─'*10}  {'─'*16}  {'─'*8}  {'─'*6}  {'─'*30}")
    for ep in episodes:
        correct_str = "yes" if ep.correct is True else ("no" if ep.correct is False else "?")
        score_str   = f"{ep.score:.3f}" if ep.score == ep.score else "—"
        prompt      = ep.prompt.replace("\n", " ")[:40]
        print(f"  {ep.episode_id:<18}  {ep.task_type:<10}  "
              f"{ep.strategy_name:<16}  {correct_str:<8}  {score_str:>6}  {prompt}")
    print(f"\n  {len(episodes)} episode(s) shown.\n")


def cmd_get(args: argparse.Namespace) -> None:
    store = EpisodeStore(args.db_path)
    ep    = store.get(args.episode_id)
    if ep is None:
        print(f"Episode not found: {args.episode_id}")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  episode_id   : {ep.episode_id}")
    print(f"  task_type    : {ep.task_type}")
    print(f"  strategy     : {ep.strategy_name}")
    print(f"  model_id     : {ep.model_id}")
    print(f"  correct      : {ep.correct}")
    print(f"  score        : {ep.score}")
    print(f"  reward       : {ep.reward}")
    print(f"  num_steps    : {ep.num_steps}")
    print(f"  block_length : {ep.block_length}")
    print(f"  temperature  : {ep.temperature}")
    print(f"  dag_seq_len  : {ep.dag_seq_len}")
    print(f"  has_dag      : {ep.dag_adjacency is not None}")
    print(f"\n  PROMPT:\n  {ep.prompt}")
    print(f"\n  GROUND TRUTH:\n  {ep.ground_truth}")
    print(f"\n  OUTPUT:\n  {ep.output}")
    if ep.comment:
        print(f"\n  COMMENT: {ep.comment}")
    print(f"{'─'*60}\n")


def cmd_export(args: argparse.Namespace) -> None:
    store = EpisodeStore(args.db_path)
    episodes = store.query(
        task_type=args.task_type,
        strategy_name=args.strategy,
        correct=True if args.correct_only else None,
        min_score=args.min_score,
        limit=args.limit or 1_000_000,
    )

    if not episodes:
        print("No episodes match the filter.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == "jsonl":
        with open(out, "w", encoding="utf-8") as f:
            for ep in episodes:
                record = {
                    "episode_id":   ep.episode_id,
                    "prompt":       ep.prompt,
                    "output":       ep.output,
                    "ground_truth": ep.ground_truth,
                    "correct":      ep.correct,
                    "score":        ep.score if ep.score == ep.score else None,
                    "strategy":     ep.strategy_name,
                    "task_type":    ep.task_type,
                    "model_id":     ep.model_id,
                    "dag_seq_len":  ep.dag_seq_len,
                    "has_dag":      ep.dag_adjacency is not None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    elif fmt == "json":
        data = []
        for ep in episodes:
            data.append({
                "episode_id": ep.episode_id, "prompt": ep.prompt,
                "output": ep.output, "ground_truth": ep.ground_truth,
                "correct": ep.correct, "score": ep.score if ep.score == ep.score else None,
                "strategy": ep.strategy_name, "task_type": ep.task_type,
            })
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(episodes)} episodes → {out}")


def cmd_prune(args: argparse.Namespace) -> None:
    store = EpisodeStore(args.db_path)
    episodes = store.query(limit=1_000_000)

    to_delete = [
        ep for ep in episodes
        if (args.max_score is not None and ep.score <= args.max_score)
        or (args.incorrect_only and ep.correct is False)
    ]

    if not to_delete:
        print("No episodes match the prune criteria.")
        return

    print(f"Found {len(to_delete)} episode(s) to delete.")
    if args.dry_run:
        print("  [DRY RUN] Nothing deleted. Remove --dry_run to confirm.")
        return

    for ep in to_delete:
        store.delete(ep.episode_id)
    print(f"Deleted {len(to_delete)} episode(s).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect and manage an EpisodeStore database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db_path", default="episodes/episodes.db")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Print database statistics")

    lst = sub.add_parser("list", help="List episodes")
    lst.add_argument("--task_type",   default=None)
    lst.add_argument("--strategy",    default=None)
    lst.add_argument("--correct_only",action="store_true")
    lst.add_argument("--min_score",   type=float, default=None)
    lst.add_argument("--limit",       type=int, default=50)

    get = sub.add_parser("get", help="Show one episode in detail")
    get.add_argument("--episode_id", required=True)

    exp = sub.add_parser("export", help="Export episodes to JSONL or JSON")
    exp.add_argument("--out", default="export/episodes.jsonl")
    exp.add_argument("--format", default="jsonl", choices=["jsonl", "json"])
    exp.add_argument("--task_type",   default=None)
    exp.add_argument("--strategy",    default=None)
    exp.add_argument("--correct_only",action="store_true")
    exp.add_argument("--min_score",   type=float, default=None)
    exp.add_argument("--limit",       type=int, default=None)

    prune = sub.add_parser("prune", help="Delete low-quality episodes")
    prune.add_argument("--max_score",     type=float, default=None,
                       help="Delete episodes with score ≤ this value")
    prune.add_argument("--incorrect_only",action="store_true",
                       help="Delete all incorrect episodes")
    prune.add_argument("--dry_run",       action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    dispatch = {
        "stats":  cmd_stats,
        "list":   cmd_list,
        "get":    cmd_get,
        "export": cmd_export,
        "prune":  cmd_prune,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
