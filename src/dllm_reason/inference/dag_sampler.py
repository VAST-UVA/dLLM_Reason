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
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        gen_length: int,
        device: torch.device | str | None = None,
    ) -> SamplingResult:
        device = device or prompt_ids.device
        batch_size = prompt_ids.shape[0]
        seq_len = prompt_ids.shape[1]
        cfg = self.config

        self.model.eval()

        # Pre-compute schedule from DAG
        schedule = self.dag.to_mask_schedule(cfg.num_steps)

        # Initialize
        x_t = prompt_ids.clone().to(device)
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

            # Timestep: 1.0 (fully masked) → 0.0 (clean)
            t_val = 1.0 - step / max(cfg.num_steps, 1)
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)

            # Model prediction
            output = self.model.forward(x_t, t)
            logits = output.logits / max(cfg.temperature, 1e-6)
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

            # Sample tokens — exclude mask token so positions can't be sampled back to mask
            if pos_mask.any():
                probs_sample = probs.clone()
                probs_sample[..., self.model.mask_token_id] = 0.0
                probs_sample = probs_sample / probs_sample.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                sampled = torch.multinomial(
                    probs_sample.view(-1, probs_sample.shape[-1]), num_samples=1
                ).view(batch_size, seq_len)
                x_t = torch.where(pos_mask, sampled, x_t)
                is_unmasked = is_unmasked | pos_mask

            if cfg.record_trajectory:
                trajectory.append(x_t.clone())

        # Refinement: re-predict and resample uncertain positions
        t_refine = torch.zeros(batch_size, device=device, dtype=torch.float32)
        for _ in range(cfg.refinement_steps):
            output = self.model.forward(x_t, t_refine)
            probs = torch.softmax(output.logits / max(cfg.temperature, 1e-6), dim=-1)
            confidences = probs.max(dim=-1).values

            low_conf = (confidences < 0.8) & ~prompt_mask
            if low_conf.any():
                probs_sample = probs.clone()
                probs_sample[..., self.model.mask_token_id] = 0.0
                probs_sample = probs_sample / probs_sample.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                sampled = torch.multinomial(
                    probs_sample.view(-1, probs_sample.shape[-1]), num_samples=1
                ).view(batch_size, seq_len)
                x_t = torch.where(low_conf, sampled, x_t)

        # Final cleanup: force-fill remaining mask tokens with argmax
        remaining = (x_t == self.model.mask_token_id) & ~prompt_mask
        if remaining.any():
            t_zero = torch.zeros(batch_size, device=device, dtype=torch.float32)
            output = self.model.forward(x_t, t_zero)
            final_logits = output.logits.clone()
            final_logits[..., self.model.mask_token_id] = -float("inf")
            x_t = torch.where(remaining, final_logits.argmax(dim=-1), x_t)

        return SamplingResult(sequences=x_t, trajectory=trajectory)
