"""Unit tests for TokenDAG core data structure."""

import pytest
import torch
from dllm_reason.graph.dag import TokenDAG
from dllm_reason.graph.templates import chain_of_thought_dag, bidirectional_dag


def test_empty_dag():
    dag = TokenDAG.empty(10)
    assert dag.seq_len == 10
    assert dag.num_edges() == 0
    assert dag.is_valid()
    levels = dag.topological_levels()
    assert len(levels) == 1  # All at level 0


def test_linear_chain():
    dag = TokenDAG.linear_chain(5)
    assert dag.num_edges() == 4
    assert dag.depth() == 5
    assert dag.is_valid()
    levels = dag.topological_levels()
    assert len(levels) == 5
    for i, level in enumerate(levels):
        assert level == [i]


def test_from_edges():
    dag = TokenDAG.from_edges(5, [(0, 1), (1, 2), (0, 3)])
    assert dag.num_edges() == 3
    assert dag.is_valid()
    assert dag.adjacency[0, 1]
    assert dag.adjacency[1, 2]
    assert dag.adjacency[0, 3]
    assert not dag.adjacency[2, 3]


def test_cycle_detection():
    # Adding edges that create a cycle should raise
    with pytest.raises((ValueError, AssertionError)):
        TokenDAG.from_edges(3, [(0, 1), (1, 2), (2, 0)])


def test_ready_positions_single():
    # Linear chain: only position 0 ready initially
    dag = TokenDAG.linear_chain(4)
    is_unmasked = torch.zeros(4, dtype=torch.bool)

    ready = dag.ready_positions(is_unmasked)
    assert ready[0]
    assert not ready[1]
    assert not ready[2]
    assert not ready[3]

    # After unmasking 0, position 1 becomes ready
    is_unmasked[0] = True
    ready = dag.ready_positions(is_unmasked)
    assert ready[0]  # Still ready (already unmasked, no remaining parents)
    assert ready[1]
    assert not ready[2]


def test_ready_positions_batch():
    dag = TokenDAG.linear_chain(4)
    # Batch of 2 sequences
    is_unmasked = torch.zeros(2, 4, dtype=torch.bool)
    is_unmasked[0, 0] = True  # seq 0: position 0 unmasked
    # seq 1: nothing unmasked

    ready = dag.ready_positions(is_unmasked)
    assert ready.shape == (2, 4)
    assert ready[0, 1]   # seq 0: pos 1 ready
    assert not ready[1, 1]   # seq 1: pos 1 not ready


def test_topological_levels_cot():
    seq_len = 12
    dag = chain_of_thought_dag(seq_len=seq_len, num_steps=3)
    levels = dag.topological_levels()
    assert len(levels) == 3
    # Each level should have ~4 positions
    for level in levels:
        assert len(level) > 0


def test_mask_schedule():
    dag = chain_of_thought_dag(seq_len=12, num_steps=3)
    schedule = dag.to_mask_schedule(num_steps=8)
    assert len(schedule) == 8
    # All positions should appear in the schedule
    all_positions = set()
    for step_positions in schedule:
        all_positions.update(step_positions)
    assert all_positions == set(range(12))


def test_dag_add_remove_edges():
    dag = TokenDAG.empty(5)
    dag2 = dag.add_edges([(0, 1), (1, 2)])
    assert dag2.num_edges() == 2
    assert dag2.is_valid()

    dag3 = dag2.remove_edges([(0, 1)])
    assert dag3.num_edges() == 1


def test_bidirectional_dag():
    dag = bidirectional_dag(seq_len=10, num_segments=4)
    assert dag.is_valid()
    levels = dag.topological_levels()
    assert len(levels) >= 2


def test_from_levels():
    levels = [[0, 1], [2, 3], [4]]
    dag = TokenDAG.from_levels(levels, seq_len=5)
    assert dag.is_valid()
    # Positions in level 1 should depend on level 0
    assert dag.adjacency[0, 2] or dag.adjacency[1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
