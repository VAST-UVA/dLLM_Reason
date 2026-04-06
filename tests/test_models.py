"""Unit tests for dLLM model implementations."""

import pytest
import torch

from dllm_reason.models.mdlm import MDLM
from dllm_reason.models.sedd import SEDD
from dllm_reason.models.d3pm import D3PM


VOCAB_SIZE = 100
SEQ_LEN = 16
BATCH_SIZE = 2


@pytest.fixture
def mdlm():
    return MDLM(vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN, dim=64, num_layers=2, num_heads=4)


@pytest.fixture
def sedd():
    return SEDD(vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN, dim=64, num_layers=2, num_heads=4)


@pytest.fixture
def d3pm():
    return D3PM(vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN, dim=64, num_layers=2, num_heads=4)


def _make_batch(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, batch=BATCH_SIZE):
    x_0 = torch.randint(0, vocab_size, (batch, seq_len))
    t = torch.rand(batch).clamp(0.01, 0.99)
    return x_0, t


# ── MDLM tests ─────────────────────────────────────────────────────────────

class TestMDLM:
    def test_forward_shape(self, mdlm):
        x_0, t = _make_batch(mdlm.vocab_size)
        x_t = mdlm.noise_input(x_0, t)
        output = mdlm.forward(x_t, t)
        assert output.logits.shape == (BATCH_SIZE, SEQ_LEN, mdlm.vocab_size)

    def test_noise_is_valid(self, mdlm):
        x_0, t = _make_batch(mdlm.vocab_size)
        x_t = mdlm.noise_input(x_0, t)
        # x_t should be either original tokens or MASK token
        valid = (x_t == x_0) | (x_t == mdlm.mask_token_id)
        assert valid.all()

    def test_loss_scalar(self, mdlm):
        x_0, _ = _make_batch(mdlm.vocab_size)
        loss = mdlm.compute_loss(x_0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_noise_increases_with_t(self, mdlm):
        x_0, _ = _make_batch(mdlm.vocab_size)
        t_low = torch.full((BATCH_SIZE,), 0.1)
        t_high = torch.full((BATCH_SIZE,), 0.9)
        x_low = mdlm.noise_input(x_0, t_low)
        x_high = mdlm.noise_input(x_0, t_high)
        # High-t should have more masks on average
        low_masked = (x_low == mdlm.mask_token_id).float().mean().item()
        high_masked = (x_high == mdlm.mask_token_id).float().mean().item()
        assert high_masked >= low_masked - 0.05  # Allow small variance


# ── SEDD tests ─────────────────────────────────────────────────────────────

class TestSEDD:
    def test_forward_shape(self, sedd):
        x_0, t = _make_batch(sedd.vocab_size)
        x_t = sedd.noise_input(x_0, t)
        output = sedd.forward(x_t, t)
        assert output.logits.shape == (BATCH_SIZE, SEQ_LEN, sedd.vocab_size)

    def test_loss_scalar(self, sedd):
        x_0, _ = _make_batch(sedd.vocab_size)
        loss = sedd.compute_loss(x_0)
        assert loss.dim() == 0
        assert loss.item() > 0


# ── D3PM tests ─────────────────────────────────────────────────────────────

class TestD3PM:
    def test_forward_shape(self, d3pm):
        x_0, t = _make_batch(d3pm.vocab_size)
        x_t = d3pm.noise_input(x_0, t)
        output = d3pm.forward(x_t, t)
        assert output.logits.shape == (BATCH_SIZE, SEQ_LEN, d3pm.vocab_size)

    def test_loss_scalar(self, d3pm):
        x_0, _ = _make_batch(d3pm.vocab_size)
        loss = d3pm.compute_loss(x_0)
        assert loss.dim() == 0
        assert loss.item() > 0


# ── Sampling tests ─────────────────────────────────────────────────────────

class TestSampling:
    def test_mdlm_random_sampling(self, mdlm):
        from dllm_reason.scheduler.random_scheduler import RandomScheduler
        from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig

        scheduler = RandomScheduler()
        sampler = DiffusionSampler(
            mdlm, scheduler,
            SamplingConfig(num_steps=4, show_progress=False),
        )
        result = sampler.sample(batch_size=2, seq_len=SEQ_LEN)
        assert result.sequences.shape == (2, SEQ_LEN)
        # No MASK tokens in output
        assert (result.sequences != mdlm.mask_token_id).all()

    def test_mdlm_dag_sampling(self, mdlm):
        from dllm_reason.graph.dag import TokenDAG
        from dllm_reason.scheduler.dag_scheduler import DAGScheduler
        from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig

        dag = TokenDAG.linear_chain(SEQ_LEN)
        scheduler = DAGScheduler(dag, sub_strategy="all_ready")
        sampler = DiffusionSampler(
            mdlm, scheduler,
            SamplingConfig(num_steps=SEQ_LEN, show_progress=False),
        )
        result = sampler.sample(batch_size=1, seq_len=SEQ_LEN)
        assert result.sequences.shape == (1, SEQ_LEN)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
