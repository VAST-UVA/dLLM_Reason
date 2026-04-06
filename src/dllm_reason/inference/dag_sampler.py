"""DAG-aware sampler with schedule-based stepping.

Extends the generic sampler with DAG-specific optimizations:
- Pre-compute the unmasking schedule from the DAG
- Use topological levels to group unmasking steps
- Support visualization of the unmasking process
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.models.base import DiffusionLM
from dllm_reason.inference.sampler import SamplingConfig, SamplingResult
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DAGSamplingConfig(SamplingConfig):
    """Extended config for DAG-based sampling."""
    refinement_steps: int = 0      # Extra refinement steps after all positions unmasked
    level_sub_strategy: str = "confidence"  # "confidence", "random", "all"


class DAGSampler:
    """DAG-guided sampler that follows topological order.

    Instead of using a scheduler per-step, this sampler pre-computes
    the full unmasking schedule from the DAG's topological levels and
    executes it directly. This is more efficient than the generic
    sampler + DAGScheduler combination.
    """

    def __init__(
        self,
        model: DiffusionLM,
        dag: TokenDAG,
        config: DAGSamplingConfig | None = None,
    ):
        self.model = model
        self.dag = dag
        self.config = config or DAGSamplingConfig()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> SamplingResult:
        seq_len = seq_len or self.model.max_seq_len
        device = device or self.model.device
        cfg = self.config

        self.model.eval()

        # Pre-compute schedule from DAG
        schedule = self.dag.to_mask_schedule(cfg.num_steps)
        total_levels = len([s for s in schedule if s])  # non-empty levels

        # Initialize
        x_t = torch.full((batch_size, seq_len), self.model.mask_token_id, dtype=torch.long, device=device)

        if prompt_ids is not None:
            x_t = prompt_ids.clone().to(device)
        if prompt_mask is None:
            prompt_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            prompt_mask = prompt_mask.to(device)

        is_unmasked = (x_t != self.model.mask_token_id) | prompt_mask

        trajectory = []
        if cfg.record_trajectory:
            trajectory.append(x_t.clone())

        steps = range(cfg.num_steps)
        if cfg.show_progress:
            steps = tqdm(steps, desc="DAG Sampling", leave=False)

        for step in steps:
            positions_this_step = schedule[step]
            if not positions_this_step:
                # Refinement step: re-predict all positions but don't change unmasking
                continue

            # Timestep
            t_val = 1.0 - step / cfg.num_steps
            t = torch.full((batch_size,), t_val, device=device)

            # Model prediction
            output = self.model.forward(x_t, t)
            logits = output.logits / cfg.temperature
            probs = torch.softmax(logits, dim=-1)

            # Create mask for positions to unmask
            pos_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            for pos in positions_this_step:
                if pos < seq_len:
                    pos_mask[:, pos] = True
            # Only unmask positions that are actually masked and not prompt
            pos_mask = pos_mask & (x_t == self.model.mask_token_id) & ~prompt_mask

            if cfg.level_sub_strategy == "confidence":
                # Among positions_this_step, unmask all (they're all at the same level)
                pass  # pos_mask already set
            elif cfg.level_sub_strategy == "random":
                # Randomly drop some positions for stochasticity
                drop = torch.rand_like(pos_mask.float()) > 0.5
                pos_mask = pos_mask & ~drop

            # Sample tokens for selected positions
            if pos_mask.any():
                sampled = torch.multinomial(
                    probs.view(-1, self.model.vocab_size), num_samples=1
                ).view(batch_size, seq_len)
                x_t = torch.where(pos_mask, sampled, x_t)
                is_unmasked = is_unmasked | pos_mask

            if cfg.record_trajectory:
                trajectory.append(x_t.clone())

        # Refinement: re-predict and resample uncertain positions
        for _ in range(cfg.refinement_steps):
            t = torch.full((batch_size,), 0.01, device=device)
            output = self.model.forward(x_t, t)
            probs = torch.softmax(output.logits / cfg.temperature, dim=-1)
            confidences = probs.max(dim=-1).values

            # Re-sample positions with low confidence (not prompt)
            low_conf = (confidences < 0.8) & ~prompt_mask
            if low_conf.any():
                sampled = torch.multinomial(
                    probs.view(-1, self.model.vocab_size), num_samples=1
                ).view(batch_size, seq_len)
                x_t = torch.where(low_conf, sampled, x_t)

        # Final cleanup: unmask any remaining
        remaining = (x_t == self.model.mask_token_id) & ~prompt_mask
        if remaining.any():
            t = torch.full((batch_size,), 0.0, device=device)
            output = self.model.forward(x_t, t)
            probs = torch.softmax(output.logits / cfg.temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.model.vocab_size), num_samples=1
            ).view(batch_size, seq_len)
            x_t = torch.where(remaining, sampled, x_t)

        return SamplingResult(sequences=x_t, trajectory=trajectory)
