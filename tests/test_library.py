"""Unit tests for DAG Library module."""

import os
import tempfile

import pytest
import numpy as np
import torch

from dllm_reason.library.config import (
    LibraryConfig,
    RetrievalMode,
    FusionStrategy,
    FeedbackSource,
    MergeStrategy,
    ConstraintMode,
)
from dllm_reason.library.entry import DAGEntry
from dllm_reason.library.store import DAGStore, StoreConfig
from dllm_reason.library.embedder import RandomEmbedder
from dllm_reason.library.retrieval import (
    RetrievalQuery,
    SemanticRetrieval,
    StructuralRetrieval,
    PerformanceRetrieval,
)
from dllm_reason.library.fusion import (
    WeightedFusion,
    RRFFusion,
    MaxFusion,
    VotingFusion,
)
from dllm_reason.library.feedback import AutoFeedback, HumanFeedback, EloFeedback
from dllm_reason.library.merge import UnionMerger, IntersectionMerger, WeightedMerger
from dllm_reason.library.fitness import CompositeFitness


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return StoreConfig(db_path=db_path)


@pytest.fixture
def store(tmp_db):
    s = DAGStore(tmp_db)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return RandomEmbedder(dim=32)


def _make_entry(seq_len=8, source="template", task="math problem", **kwargs) -> DAGEntry:
    adj = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    # Simple chain: 0->1->2->...
    for i in range(seq_len - 1):
        adj[i, i + 1] = True
    return DAGEntry(
        adjacency=adj.flatten().int().tolist(),
        seq_len=seq_len,
        source=source,
        task_description=task,
        num_edges=seq_len - 1,
        depth=seq_len,
        **kwargs,
    )


# ── DAGEntry tests ───────────────────────────────────────────────────────────

class TestDAGEntry:
    def test_serialization_roundtrip(self):
        entry = _make_entry()
        entry.add_benchmark_score("gsm8k", {"accuracy": 0.45})
        entry.add_human_rating("alice", 4.0)

        restored = DAGEntry.from_json(entry.to_json())
        assert restored.entry_id == entry.entry_id
        assert restored.seq_len == entry.seq_len
        assert restored.benchmark_scores == entry.benchmark_scores
        assert len(restored.human_ratings) == 1

    def test_to_token_dag(self):
        entry = _make_entry(seq_len=8)
        dag = entry.to_token_dag()
        assert dag.seq_len == 8
        assert dag.adjacency[0, 1].item() is True
        assert dag.adjacency[1, 0].item() is False

    def test_best_score(self):
        entry = _make_entry()
        entry.add_benchmark_score("gsm8k", {"accuracy": 0.3, "f1": 0.5})
        entry.add_benchmark_score("mmlu", {"accuracy": 0.6})
        assert entry.best_score("accuracy") == 0.6
        assert entry.best_score("f1") == 0.5

    def test_avg_human_rating(self):
        entry = _make_entry()
        entry.add_human_rating("a", 3.0)
        entry.add_human_rating("b", 5.0)
        assert entry.avg_human_rating() == 4.0


# ── DAGStore tests ───────────────────────────────────────────────────────────

class TestDAGStore:
    def test_add_and_get(self, store):
        entry = _make_entry()
        store.add(entry)
        retrieved = store.get(entry.entry_id)
        assert retrieved is not None
        assert retrieved.entry_id == entry.entry_id
        assert retrieved.seq_len == entry.seq_len

    def test_delete(self, store):
        entry = _make_entry()
        store.add(entry)
        assert store.delete(entry.entry_id)
        assert store.get(entry.entry_id) is None

    def test_count(self, store):
        assert store.count() == 0
        store.add(_make_entry())
        store.add(_make_entry())
        assert store.count() == 2

    def test_list_all(self, store):
        e1 = _make_entry(task="task1")
        e2 = _make_entry(task="task2")
        store.add(e1)
        store.add(e2)
        all_entries = store.list_all()
        assert len(all_entries) == 2

    def test_query_by_source(self, store):
        store.add(_make_entry(source="template"))
        store.add(_make_entry(source="search"))
        store.add(_make_entry(source="template"))
        assert len(store.query_by_source("template")) == 2
        assert len(store.query_by_source("search")) == 1

    def test_top_by_elo(self, store):
        e1 = _make_entry()
        e1.elo_rating = 1600
        e2 = _make_entry()
        e2.elo_rating = 1400
        store.add(e1)
        store.add(e2)
        top = store.top_by_elo(k=1)
        assert len(top) == 1
        assert top[0].elo_rating == 1600

    def test_brute_force_search(self, store):
        embedder = RandomEmbedder(dim=32)
        e1 = _make_entry(task="solve quadratic equation")
        e1.task_embedding = embedder.embed("solve quadratic equation").tolist()
        e2 = _make_entry(task="sort a list of numbers")
        e2.task_embedding = embedder.embed("sort a list of numbers").tolist()
        store.add(e1)
        store.add(e2)

        query_vec = embedder.embed("quadratic formula")
        results = store.search_by_embedding(query_vec, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[1], float) for r in results)


# ── Embedder tests ───────────────────────────────────────────────────────────

class TestEmbedder:
    def test_random_embedder(self):
        emb = RandomEmbedder(dim=64)
        vec = emb.embed("hello world")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32
        # Deterministic for same input
        vec2 = emb.embed("hello world")
        assert np.allclose(vec, vec2)
        # Different for different input
        vec3 = emb.embed("goodbye world")
        assert not np.allclose(vec, vec3)

    def test_batch_embed(self):
        emb = RandomEmbedder(dim=64)
        batch = emb.embed_batch(["a", "b", "c"])
        assert batch.shape == (3, 64)


# ── Retrieval tests ──────────────────────────────────────────────────────────

class TestRetrieval:
    def test_semantic_retrieval(self, store):
        embedder = RandomEmbedder(dim=32)
        for task in ["math equation", "code sorting", "logic puzzle"]:
            e = _make_entry(task=task)
            e.task_embedding = embedder.embed(task).tolist()
            store.add(e)

        channel = SemanticRetrieval(embedder)
        query = RetrievalQuery(task_description="math problem")
        results = channel.retrieve(query, store, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r[1], float) for r in results)

    def test_structural_retrieval(self, store):
        e1 = _make_entry(seq_len=8, task="a")
        e2 = _make_entry(seq_len=8, task="b")
        store.add(e1)
        store.add(e2)

        ref = _make_entry(seq_len=8, task="ref")
        channel = StructuralRetrieval(metric="edit_distance")
        query = RetrievalQuery(reference_dag=ref)
        results = channel.retrieve(query, store, top_k=5)
        assert len(results) == 2
        # Same structure → high similarity
        assert results[0][1] == 1.0  # 1/(1+0) = 1.0

    def test_performance_retrieval(self, store):
        e1 = _make_entry(task="a")
        e1.add_benchmark_score("gsm8k", {"accuracy": 0.8})
        e2 = _make_entry(task="b")
        e2.add_benchmark_score("gsm8k", {"accuracy": 0.3})
        store.add(e1)
        store.add(e2)

        channel = PerformanceRetrieval()
        query = RetrievalQuery(target_benchmark="gsm8k", target_metric="accuracy")
        results = channel.retrieve(query, store, top_k=2)
        assert results[0][1] > results[1][1]


# ── Fusion tests ─────────────────────────────────────────────────────────────

class TestFusion:
    def _make_channel_results(self):
        e1 = _make_entry(task="a")
        e2 = _make_entry(task="b")
        e3 = _make_entry(task="c")
        return {
            "semantic": [(e1, 0.9), (e2, 0.7), (e3, 0.3)],
            "structural": [(e2, 0.8), (e3, 0.6), (e1, 0.2)],
        }

    def test_weighted_fusion(self):
        fuser = WeightedFusion({"semantic": 0.6, "structural": 0.4})
        results = fuser.fuse(self._make_channel_results(), top_k=3)
        assert len(results) == 3
        # Scores should be positive
        assert all(s > 0 for _, s in results)

    def test_rrf_fusion(self):
        fuser = RRFFusion(k=60)
        results = fuser.fuse(self._make_channel_results(), top_k=2)
        assert len(results) == 2

    def test_max_fusion(self):
        fuser = MaxFusion()
        results = fuser.fuse(self._make_channel_results(), top_k=3)
        assert len(results) == 3

    def test_voting_fusion(self):
        fuser = VotingFusion()
        results = fuser.fuse(self._make_channel_results(), top_k=3)
        assert len(results) == 3


# ── Feedback tests ───────────────────────────────────────────────────────────

class TestFeedback:
    def test_auto_feedback(self, store):
        entry = _make_entry()
        store.add(entry)
        handler = AutoFeedback(metric="accuracy")
        updated = handler.update(entry, store, benchmark="gsm8k", metrics={"accuracy": 0.55})
        assert updated.benchmark_scores["gsm8k"]["accuracy"] == 0.55
        # Check persisted
        reloaded = store.get(entry.entry_id)
        assert reloaded.benchmark_scores["gsm8k"]["accuracy"] == 0.55

    def test_human_feedback(self, store):
        entry = _make_entry()
        store.add(entry)
        handler = HumanFeedback()
        handler.update(entry, store, rater_id="bob", score=4.5)
        assert len(entry.human_ratings) == 1
        assert entry.avg_human_rating() == 4.5

    def test_elo_feedback(self, store):
        e1 = _make_entry(task="a")
        e1.elo_rating = 1500
        e2 = _make_entry(task="b")
        e2.elo_rating = 1500
        store.add(e1)
        store.add(e2)

        handler = EloFeedback(k=32)
        new_a, new_b = handler.update_pair(e1, e2, outcome_a=1.0, store=store)
        assert new_a > 1500
        assert new_b < 1500
        assert abs(new_a - 1516) < 1  # Expected: 1500 + 32*(1-0.5) = 1516

    def test_elo_tournament(self, store):
        entries = [_make_entry(task=f"t{i}") for i in range(4)]
        for e in entries:
            store.add(e)
        handler = EloFeedback(k=32)
        outcomes = [(0, 1, 1.0), (2, 3, 0.0), (0, 2, 0.5)]
        handler.run_tournament(entries, outcomes, store)
        # Entry 0 won once, drew once → should be above 1500
        assert entries[0].elo_rating > 1500


# ── Merge tests ──────────────────────────────────────────────────────────────

class TestMerge:
    def _make_entries(self):
        # Two 4-node DAGs with different edges
        e1 = DAGEntry(
            adjacency=[0,1,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
            seq_len=4, num_edges=2, depth=3,
        )
        e2 = DAGEntry(
            adjacency=[0,0,0,0, 0,0,0,0, 0,0,0,1, 0,0,0,0],
            seq_len=4, num_edges=1, depth=2,
        )
        return [e1, e2]

    def test_union_merge(self):
        entries = self._make_entries()
        merger = UnionMerger()
        result = merger.merge(entries, [1.0, 1.0])
        assert result.shape == (4, 4)
        assert result[0, 1].item()  # from e1
        assert result[2, 3].item()  # from e2

    def test_intersection_merge(self):
        entries = self._make_entries()
        merger = IntersectionMerger()
        result = merger.merge(entries, [1.0, 1.0])
        # No shared edges
        assert not result.any()

    def test_weighted_merge(self):
        entries = self._make_entries()
        merger = WeightedMerger(threshold=0.3)
        result = merger.merge(entries, [0.8, 0.2])
        assert result.shape == (4, 4)


# ── Fitness tests ────────────────────────────────────────────────────────────

class TestFitness:
    def test_composite_fitness(self):
        config = LibraryConfig()
        config.feedback.sources = [FeedbackSource.AUTO]
        fitness = CompositeFitness(config)

        entry = _make_entry()
        entry.add_benchmark_score("gsm8k", {"accuracy": 0.6})
        result = fitness.evaluate(entry, benchmark="gsm8k")
        assert result.total > 0
        assert "accuracy" in result.components

    def test_fitness_without_human(self):
        config = LibraryConfig()
        config.feedback.sources = [FeedbackSource.AUTO]
        # human_weight should be excluded from normalization
        weights = config.normalized_fitness_weights()
        assert "human" not in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_fitness_all_sources(self):
        config = LibraryConfig()
        config.feedback.sources = [FeedbackSource.AUTO, FeedbackSource.HUMAN]
        weights = config.normalized_fitness_weights()
        assert "accuracy" in weights
        assert "human" in weights
        assert "structural" in weights

    def test_structural_score(self):
        entry = _make_entry(seq_len=16)
        score = CompositeFitness._structural_score(entry)
        assert 0.0 <= score <= 1.0


# ── Config tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        config = LibraryConfig()
        assert config.enabled
        assert config.retrieval.enabled
        assert config.fusion.enabled
        assert config.feedback.enabled

    def test_kill_switch(self):
        config = LibraryConfig(enabled=False)
        assert not config.enabled

    def test_active_channels_when_disabled(self):
        config = LibraryConfig()
        config.retrieval.enabled = False
        assert config.active_retrieval_channels() == []

    def test_normalized_weights_renormalize(self):
        config = LibraryConfig()
        config.feedback.sources = [FeedbackSource.AUTO]
        config.retrieval.enabled = True
        weights = config.normalized_fitness_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
