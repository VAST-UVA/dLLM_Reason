"""Unit tests for unmasking schedulers."""

import pytest
import torch

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.scheduler.random_scheduler import RandomScheduler
from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
from dllm_reason.scheduler.linear_scheduler import LinearScheduler
from dllm_reason.scheduler.dag_scheduler import DAGScheduler


def make_inputs(batch=2, seq_len=10, vocab_size=100, num_masked=5):
    """Create dummy scheduler inputs."""
    current_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    current_mask[:, :num_masked] = True
    is_unmasked = ~current_mask
    logits = torch.randn(batch, seq_len, vocab_size)
    confidences = torch.rand(batch, seq_len)
    return current_mask, is_unmasked, logits, confidences


def test_random_scheduler():
    scheduler = RandomScheduler()
    current_mask, is_unmasked, logits, confidences = make_inputs()
    result = scheduler.select_positions(0, 10, current_mask, is_unmasked, logits, confidences)
    assert result.shape == (2, 10)
    # Selected positions must be masked
    assert not (result & ~current_mask).any()
    # At least 1 position selected
    assert result.any()


def test_confidence_scheduler():
    scheduler = ConfidenceScheduler()
    current_mask, is_unmasked, logits, confidences = make_inputs()
    result = scheduler.select_positions(0, 10, current_mask, is_unmasked, logits, confidences)
    assert result.shape == (2, 10)
    assert not (result & ~current_mask).any()


def test_linear_scheduler():
    scheduler = LinearScheduler()
    current_mask, is_unmasked, logits, confidences = make_inputs(seq_len=10, num_masked=8)
    # current_mask has positions 0..7 masked
    result = scheduler.select_positions(0, 10, current_mask, is_unmasked, logits, confidences)
    assert result.shape == (2, 10)
    # Selected positions should be the leftmost masked ones
    assert not (result & ~current_mask).any()


def test_dag_scheduler_respects_order():
    """DAG scheduler should only unmask positions whose parents are done."""
    dag = TokenDAG.linear_chain(10)
    scheduler = DAGScheduler(dag, sub_strategy="all_ready")

    # Initially, only position 0 is ready
    current_mask = torch.ones(1, 10, dtype=torch.bool)
    is_unmasked = torch.zeros(1, 10, dtype=torch.bool)
    logits = torch.randn(1, 10, 50)
    confidences = torch.rand(1, 10)

    result = scheduler.select_positions(0, 10, current_mask, is_unmasked, logits, confidences)
    # Only position 0 should be selected
    assert result[0, 0]
    assert not result[0, 1:].any()


def test_dag_scheduler_all_strategies():
    dag = TokenDAG.linear_chain(10)
    current_mask = torch.ones(1, 10, dtype=torch.bool)
    is_unmasked = torch.zeros(1, 10, dtype=torch.bool)
    is_unmasked[0, 0] = True  # position 0 already done
    current_mask[0, 0] = False

    logits = torch.randn(1, 10, 50)
    confidences = torch.rand(1, 10)

    for strategy in ["all_ready", "confidence_topk", "proportional"]:
        scheduler = DAGScheduler(dag, sub_strategy=strategy)
        result = scheduler.select_positions(0, 10, current_mask, is_unmasked, logits, confidences)
        assert result.shape == (1, 10)
        # Should NOT select position 0 (already unmasked) or positions 2+ (parents not done)
        assert not result[0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
