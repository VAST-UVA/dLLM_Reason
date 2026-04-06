"""DAGLibrary: persistent storage backend (SQLite + optional FAISS).

The store is the single source of truth for all DAG entries.
FAISS index is rebuilt lazily when semantic retrieval is requested.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from dllm_reason.library.config import StoreConfig
from dllm_reason.library.entry import DAGEntry

logger = logging.getLogger(__name__)

# ── SQL schema ───────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS dag_entries (
    entry_id     TEXT PRIMARY KEY,
    seq_len      INTEGER NOT NULL,
    source       TEXT,
    template_name TEXT,
    search_method TEXT,
    task_description TEXT,
    elo_rating   REAL DEFAULT 1500.0,
    num_edges    INTEGER DEFAULT 0,
    depth        INTEGER DEFAULT 0,
    created_at   REAL,
    updated_at   REAL,
    data_json    TEXT NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_source ON dag_entries(source);
CREATE INDEX IF NOT EXISTS idx_seq_len ON dag_entries(seq_len);
CREATE INDEX IF NOT EXISTS idx_elo ON dag_entries(elo_rating);
"""


class DAGStore:
    """SQLite-backed persistent store for DAGEntry objects."""

    def __init__(self, config: StoreConfig | None = None):
        self.config = config or StoreConfig()
        self._db_path = self.config.db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._faiss_index = None
        self._faiss_ids: list[str] = []
        self._init_db()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── CRUD ─────────────────────────────────────────────────────────────

    def add(self, entry: DAGEntry) -> None:
        """Insert or replace a DAG entry."""
        self._conn.execute(
            """INSERT OR REPLACE INTO dag_entries
               (entry_id, seq_len, source, template_name, search_method,
                task_description, elo_rating, num_edges, depth,
                created_at, updated_at, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.entry_id, entry.seq_len, entry.source,
                entry.template_name, entry.search_method,
                entry.task_description, entry.elo_rating,
                entry.num_edges, entry.depth,
                entry.created_at, entry.updated_at,
                entry.to_json(),
            ),
        )
        self._conn.commit()
        self._faiss_index = None  # invalidate

    def get(self, entry_id: str) -> Optional[DAGEntry]:
        row = self._conn.execute(
            "SELECT data_json FROM dag_entries WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return DAGEntry.from_json(row[0])

    def delete(self, entry_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM dag_entries WHERE entry_id = ?", (entry_id,)
        )
        self._conn.commit()
        self._faiss_index = None
        return cur.rowcount > 0

    def update(self, entry: DAGEntry) -> None:
        """Update an existing entry (same as add with upsert)."""
        self.add(entry)

    def list_all(self, limit: int = 1000, offset: int = 0) -> list[DAGEntry]:
        rows = self._conn.execute(
            "SELECT data_json FROM dag_entries ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [DAGEntry.from_json(r[0]) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM dag_entries").fetchone()[0]

    # ── Filtered queries ─────────────────────────────────────────────────

    def query_by_source(self, source: str) -> list[DAGEntry]:
        rows = self._conn.execute(
            "SELECT data_json FROM dag_entries WHERE source = ?", (source,)
        ).fetchall()
        return [DAGEntry.from_json(r[0]) for r in rows]

    def query_by_seq_len(self, seq_len: int) -> list[DAGEntry]:
        rows = self._conn.execute(
            "SELECT data_json FROM dag_entries WHERE seq_len = ?", (seq_len,)
        ).fetchall()
        return [DAGEntry.from_json(r[0]) for r in rows]

    def top_by_elo(self, k: int = 10) -> list[DAGEntry]:
        rows = self._conn.execute(
            "SELECT data_json FROM dag_entries ORDER BY elo_rating DESC LIMIT ?", (k,)
        ).fetchall()
        return [DAGEntry.from_json(r[0]) for r in rows]

    # ── FAISS index for semantic retrieval ────────────────────────────────

    def build_faiss_index(self) -> None:
        """(Re)build FAISS index from stored embeddings."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss not installed — semantic retrieval will use brute-force.")
            return

        rows = self._conn.execute(
            "SELECT entry_id, data_json FROM dag_entries"
        ).fetchall()

        ids, vectors = [], []
        for entry_id, data_json in rows:
            entry = DAGEntry.from_json(data_json)
            if entry.task_embedding is not None:
                ids.append(entry_id)
                vectors.append(entry.task_embedding)

        if not vectors:
            logger.info("No embeddings found — FAISS index empty.")
            self._faiss_index = None
            self._faiss_ids = []
            return

        dim = len(vectors[0])
        matrix = np.array(vectors, dtype=np.float32)
        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        self._faiss_index = index
        self._faiss_ids = ids
        logger.info("FAISS index built with %d entries, dim=%d", len(ids), dim)

    def search_by_embedding(
        self, query_vec: list[float] | np.ndarray, top_k: int = 5
    ) -> list[tuple[DAGEntry, float]]:
        """Nearest-neighbour search. Returns (entry, score) pairs."""
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec, dtype=np.float32)
        query_vec = query_vec.reshape(1, -1)

        # Try FAISS first
        if self._faiss_index is not None:
            import faiss
            faiss.normalize_L2(query_vec)
            scores, indices = self._faiss_index.search(query_vec, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                entry = self.get(self._faiss_ids[idx])
                if entry:
                    results.append((entry, float(score)))
            return results

        # Brute-force fallback
        return self._brute_force_search(query_vec[0], top_k)

    def _brute_force_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[DAGEntry, float]]:
        entries = self.list_all(limit=10000)
        scored = []
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        for e in entries:
            if e.task_embedding is None:
                continue
            v = np.array(e.task_embedding, dtype=np.float32)
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            score = float(np.dot(q_norm, v_norm))
            scored.append((e, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
