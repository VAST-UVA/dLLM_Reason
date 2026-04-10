"""DAGEpisode: single interaction record + EpisodeStore (SQLite-backed).

An *episode* is one complete interaction:
    prompt  → [strategy / DAG]  → model output  → evaluation

This is distinct from DAGEntry (which focuses on DAG structure for retrieval).
Episodes are the raw experience data used for:
  - Manual review / annotation
  - SFT: train on (prompt, correct_output) pairs
  - RL / GRPO: reward = correct ? +1 : -1

Schema
------
episodes
    episode_id   TEXT PRIMARY KEY
    prompt       TEXT
    task_type    TEXT          -- "math" | "code" | "qa" | ...
    ground_truth TEXT
    strategy_name TEXT
    dag_seq_len  INTEGER
    dag_json     TEXT          -- JSON-encoded 2-D int list, or NULL
    output       TEXT
    correct      INTEGER       -- 0 | 1 | NULL (not evaluated)
    score        REAL          -- 0.0–1.0
    comment      TEXT
    model_id     TEXT
    num_steps    INTEGER
    block_length INTEGER
    temperature  REAL
    timestamp    REAL
    metadata_json TEXT         -- JSON-encoded dict
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch


# ── DAGEpisode dataclass ──────────────────────────────────────────────────────

@dataclass
class DAGEpisode:
    """One prompt → strategy → output → evaluation record.

    Attributes
    ----------
    episode_id:    Unique identifier (UUID hex[:16]).
    prompt:        Exact user prompt text.
    task_type:     Short label: "math", "code", "qa", "general", etc.
    ground_truth:  Expected / reference answer (empty string if unknown).
    strategy_name: Name of the unmasking strategy used
                   (e.g. "cot", "confidence", "adaptive_dynamic").
    dag_seq_len:   Sequence length the DAG was built for (0 if no DAG).
    dag_adjacency: 2-D list[list[int]] adjacency matrix, or None.
    output:        Full model-generated text.
    correct:       True = correct, False = wrong, None = not yet evaluated.
    score:         Numeric score 0.0–1.0 (NaN before evaluation).
    comment:       Free-form annotation or auto-extracted verdict.
    model_id:      Model checkpoint path / HuggingFace ID.
    num_steps:     Denoising steps used for generation.
    block_length:  Block length used for generation.
    temperature:   Sampling temperature.
    timestamp:     Unix time of creation.
    metadata:      Arbitrary extra key-value data.
    """

    episode_id:    str  = field(default_factory=lambda: uuid.uuid4().hex[:16])
    prompt:        str  = ""
    task_type:     str  = "general"
    ground_truth:  str  = ""
    strategy_name: str  = "confidence"
    dag_seq_len:   int  = 0
    dag_adjacency: list[list[int]] | None = None
    output:        str  = ""
    correct:       bool | None = None
    score:         float = float("nan")
    comment:       str  = ""
    model_id:      str  = ""
    num_steps:     int  = 128
    block_length:  int  = 32
    temperature:   float = 0.0
    timestamp:     float = field(default_factory=time.time)
    metadata:      dict[str, Any] = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def is_evaluated(self) -> bool:
        return self.correct is not None

    @property
    def reward(self) -> float:
        """RL reward: +1 correct, -1 wrong, 0 unevaluated."""
        if self.correct is True:
            return 1.0
        if self.correct is False:
            return -1.0
        return 0.0

    # ── DAG helpers ───────────────────────────────────────────────────────

    def to_token_dag(self, device: str = "cpu") -> Any:
        """Reconstruct a TokenDAG from stored adjacency.

        Returns None if no DAG was recorded.
        """
        if self.dag_adjacency is None or self.dag_seq_len == 0:
            return None
        from dllm_reason.graph.dag import TokenDAG
        flat = [cell for row in self.dag_adjacency for cell in row]
        adj = torch.tensor(flat, dtype=torch.bool, device=device)
        adj = adj.reshape(self.dag_seq_len, self.dag_seq_len)
        return TokenDAG(adj)

    @staticmethod
    def adjacency_from_dag(dag: Any) -> list[list[int]]:
        """Convert a TokenDAG's adjacency matrix to a 2-D int list."""
        arr = dag.adjacency.cpu().int().tolist()
        return arr

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id":    self.episode_id,
            "prompt":        self.prompt,
            "task_type":     self.task_type,
            "ground_truth":  self.ground_truth,
            "strategy_name": self.strategy_name,
            "dag_seq_len":   self.dag_seq_len,
            "dag_adjacency": self.dag_adjacency,
            "output":        self.output,
            "correct":       self.correct,
            "score":         self.score,
            "comment":       self.comment,
            "model_id":      self.model_id,
            "num_steps":     self.num_steps,
            "block_length":  self.block_length,
            "temperature":   self.temperature,
            "timestamp":     self.timestamp,
            "metadata":      self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DAGEpisode:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> DAGEpisode:
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        ev = "?" if self.correct is None else ("✓" if self.correct else "✗")
        return (
            f"DAGEpisode(id={self.episode_id!r}, "
            f"task={self.task_type!r}, "
            f"strategy={self.strategy_name!r}, "
            f"correct={ev}, "
            f"score={self.score:.3f})"
        )


# ── EpisodeStore ──────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id    TEXT PRIMARY KEY,
    prompt        TEXT NOT NULL,
    task_type     TEXT NOT NULL DEFAULT 'general',
    ground_truth  TEXT NOT NULL DEFAULT '',
    strategy_name TEXT NOT NULL DEFAULT 'confidence',
    dag_seq_len   INTEGER NOT NULL DEFAULT 0,
    dag_json      TEXT,
    output        TEXT NOT NULL DEFAULT '',
    correct       INTEGER,
    score         REAL NOT NULL DEFAULT 0.0,
    comment       TEXT NOT NULL DEFAULT '',
    model_id      TEXT NOT NULL DEFAULT '',
    num_steps     INTEGER NOT NULL DEFAULT 128,
    block_length  INTEGER NOT NULL DEFAULT 32,
    temperature   REAL NOT NULL DEFAULT 0.0,
    timestamp     REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_task_type     ON episodes (task_type);
CREATE INDEX IF NOT EXISTS idx_strategy_name ON episodes (strategy_name);
CREATE INDEX IF NOT EXISTS idx_correct       ON episodes (correct);
CREATE INDEX IF NOT EXISTS idx_timestamp     ON episodes (timestamp);
"""


class EpisodeStore:
    """SQLite-backed store for DAGEpisode records.

    Thread-safe for concurrent readers; uses WAL mode for writes.

    Usage
    -----
    store = EpisodeStore("episodes.db")

    ep = DAGEpisode(prompt="What is 2+2?", task_type="math", ...)
    store.add(ep)

    store.update_eval(ep.episode_id, correct=True, score=1.0, comment="4")

    for ep in store.iter_for_training(task_type="math", correct_only=True):
        ...
    """

    def __init__(self, db_path: str | Path = "episodes.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── DB init ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TABLE)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add(self, episode: DAGEpisode) -> None:
        """Insert a new episode (raises if episode_id already exists)."""
        dag_json  = json.dumps(episode.dag_adjacency) if episode.dag_adjacency is not None else None
        meta_json = json.dumps(episode.metadata)
        correct_int = None if episode.correct is None else int(episode.correct)
        score = 0.0 if (episode.score != episode.score) else episode.score  # NaN → 0

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO episodes
                   (episode_id, prompt, task_type, ground_truth,
                    strategy_name, dag_seq_len, dag_json,
                    output, correct, score, comment,
                    model_id, num_steps, block_length, temperature,
                    timestamp, metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    episode.episode_id, episode.prompt, episode.task_type,
                    episode.ground_truth, episode.strategy_name,
                    episode.dag_seq_len, dag_json,
                    episode.output, correct_int, score, episode.comment,
                    episode.model_id, episode.num_steps, episode.block_length,
                    episode.temperature, episode.timestamp, meta_json,
                ),
            )

    def get(self, episode_id: str) -> DAGEpisode | None:
        """Retrieve a single episode by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
        return self._row_to_episode(row) if row else None

    def update_eval(
        self,
        episode_id: str,
        correct: bool | None = None,
        score: float | None = None,
        comment: str | None = None,
    ) -> bool:
        """Update evaluation fields for an existing episode.

        Returns True if the episode was found and updated.
        """
        parts, vals = [], []
        if correct is not None:
            parts.append("correct = ?")
            vals.append(int(correct))
        if score is not None:
            parts.append("score = ?")
            vals.append(score)
        if comment is not None:
            parts.append("comment = ?")
            vals.append(comment)
        if not parts:
            return False
        vals.append(episode_id)
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE episodes SET {', '.join(parts)} WHERE episode_id = ?",
                vals,
            )
        return cur.rowcount > 0

    def delete(self, episode_id: str) -> bool:
        """Delete an episode by ID. Returns True if it existed."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM episodes WHERE episode_id = ?", (episode_id,)
            )
        return cur.rowcount > 0

    # ── Queries ───────────────────────────────────────────────────────────

    def list_all(self, limit: int = 1000, offset: int = 0) -> list[DAGEpisode]:
        """Return episodes ordered by timestamp (newest first)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def query(
        self,
        task_type: str | None = None,
        strategy_name: str | None = None,
        correct: bool | None = None,
        evaluated_only: bool = False,
        min_score: float | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[DAGEpisode]:
        """Filtered query across the store."""
        clauses, vals = [], []

        if task_type is not None:
            clauses.append("task_type = ?")
            vals.append(task_type)
        if strategy_name is not None:
            clauses.append("strategy_name = ?")
            vals.append(strategy_name)
        if correct is not None:
            clauses.append("correct = ?")
            vals.append(int(correct))
        if evaluated_only:
            clauses.append("correct IS NOT NULL")
        if min_score is not None:
            clauses.append("score >= ?")
            vals.append(min_score)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT * FROM episodes {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        )
        vals.extend([limit, offset])

        with self._conn() as conn:
            rows = conn.execute(sql, vals).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def iter_for_training(
        self,
        task_type: str | None = None,
        correct_only: bool = False,
        strategy_name: str | None = None,
        batch_size: int = 64,
    ) -> Iterator[DAGEpisode]:
        """Yield episodes one at a time, suitable for large-scale training loops.

        Uses offset-based pagination so memory usage is bounded.
        """
        offset = 0
        while True:
            batch = self.query(
                task_type=task_type,
                strategy_name=strategy_name,
                correct=True if correct_only else None,
                evaluated_only=correct_only,
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break
            yield from batch
            offset += batch_size

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return a summary dict of store contents."""
        with self._conn() as conn:
            total      = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            evaluated  = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE correct IS NOT NULL"
            ).fetchone()[0]
            correct    = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE correct = 1"
            ).fetchone()[0]
            wrong      = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE correct = 0"
            ).fetchone()[0]
            avg_score  = conn.execute(
                "SELECT AVG(score) FROM episodes WHERE correct IS NOT NULL"
            ).fetchone()[0] or 0.0

            by_task = {
                row["task_type"]: row["cnt"]
                for row in conn.execute(
                    "SELECT task_type, COUNT(*) AS cnt FROM episodes GROUP BY task_type"
                ).fetchall()
            }
            by_strategy = {
                row["strategy_name"]: row["cnt"]
                for row in conn.execute(
                    "SELECT strategy_name, COUNT(*) AS cnt FROM episodes GROUP BY strategy_name"
                ).fetchall()
            }

        return {
            "total":       total,
            "evaluated":   evaluated,
            "correct":     correct,
            "wrong":       wrong,
            "unevaluated": total - evaluated,
            "accuracy":    correct / max(evaluated, 1),
            "avg_score":   round(avg_score, 4),
            "by_task":     by_task,
            "by_strategy": by_strategy,
        }

    def print_stats(self) -> None:
        """Print a human-readable summary to stdout."""
        s = self.stats()
        print(f"\n{'='*50}")
        print(f"  EpisodeStore  →  {self.db_path}")
        print(f"{'='*50}")
        print(f"  Total episodes : {s['total']}")
        print(f"  Evaluated      : {s['evaluated']}  "
              f"({s['correct']} correct / {s['wrong']} wrong)")
        print(f"  Accuracy       : {s['accuracy']:.1%}")
        print(f"  Avg score      : {s['avg_score']:.4f}")
        print(f"\n  By task type:")
        for k, v in s["by_task"].items():
            print(f"    {k:<20} {v}")
        print(f"\n  By strategy:")
        for k, v in s["by_strategy"].items():
            print(f"    {k:<24} {v}")
        print(f"{'='*50}\n")

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> DAGEpisode:
        dag_adj = json.loads(row["dag_json"]) if row["dag_json"] else None
        metadata = json.loads(row["metadata_json"] or "{}")
        correct_raw = row["correct"]
        correct = None if correct_raw is None else bool(correct_raw)
        return DAGEpisode(
            episode_id    = row["episode_id"],
            prompt        = row["prompt"],
            task_type     = row["task_type"],
            ground_truth  = row["ground_truth"],
            strategy_name = row["strategy_name"],
            dag_seq_len   = row["dag_seq_len"],
            dag_adjacency = dag_adj,
            output        = row["output"],
            correct       = correct,
            score         = row["score"],
            comment       = row["comment"],
            model_id      = row["model_id"],
            num_steps     = row["num_steps"],
            block_length  = row["block_length"],
            temperature   = row["temperature"],
            timestamp     = row["timestamp"],
            metadata      = metadata,
        )
