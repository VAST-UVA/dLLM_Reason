"""Generic diffusion sampling loop.

Provides a clean, configurable sampling interface that works with
any DiffusionLM model and any UnmaskingScheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from dllm_reason.models.base import DiffusionLM
from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for the sampling process."""
    num_steps: int = 64
    temperature: float = 1.0
    top_k: int = 0           # 0 = no top-k filtering
    top_p: float = 1.0       # 1.0 = no nucleus filtering
    show_progress: bool = True
    record_trajectory: bool = False


@dataclass
class SamplingResult:
    """Result of a sampling run."""
    sequences: torch.Tensor        # (batch, seq_len) final token ids
    trajectory: list[torch.Tensor] = field(default_factory=list)  # list of (batch, seq_len) at each step


class DiffusionSampler:
    """Generic sampling engine for discrete diffusion models.

    Coordinates the model (token prediction) and scheduler (position selection)
    to generate sequences through iterative unmasking.
    """

    def __init__(self, model: DiffusionLM, scheduler: UnmaskingScheduler, config: SamplingConfig | None = None):
        self.model = model
        self.scheduler = scheduler
        self.config = config or SamplingConfig()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> SamplingResult:
        """Run the sampling loop.

        Args:
            batch_size: number of sequences to generate
            seq_len: sequence length (defaults to model's max_seq_len)
            prompt_ids: (batch, seq_len) prompt token ids (non-prompt positions should be mask_token_id)
            prompt_mask: (batch, seq_len) bool, True for prompt positions (not to be modified)
            device: generation device

        Returns:
            SamplingResult with final sequences and optional trajectory
        """
        seq_len = seq_len or self.model.max_seq_len
        device = device or self.model.device
        cfg = self.config

        self.model.eval()
        self.scheduler.reset()

        # Initialize fully masked
        x_t = torch.full((batch_size, seq_len), self.model.mask_token_id, dtype=torch.long, device=device)

        # Fill prompt
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
            steps = tqdm(steps, desc="Sampling", leave=False)

        for step in steps:
            # Timestep: from ~1 (noisy) to ~0 (clean)
            t_val = 1.0 - step / cfg.num_steps
            t = torch.full((batch_size,), t_val, device=device)

            # Model forward
            output = self.model.forward(x_t, t)
            logits = output.logits

            # Apply temperature
            if cfg.temperature != 1.0:
                logits = logits / cfg.temperature

            # Top-k filtering
            if cfg.top_k > 0:
                top_k_val = min(cfg.top_k, logits.shape[-1])
                topk_vals, _ = logits.topk(top_k_val, dim=-1)
                threshold = topk_vals[..., -1:]
                logits = logits.masked_fill(logits < threshold, -float("inf"))

            # Top-p (nucleus) filtering
            if cfg.top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cum_probs - sorted_logits.softmax(dim=-1) >= cfg.top_p
                sorted_logits[mask] = -float("inf")
                # Unsort
                logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

            # Compute probabilities and confidence
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.max(dim=-1).values

            # Ask scheduler which positions to unmask
            current_mask = (x_t == self.model.mask_token_id) & ~prompt_mask
            positions_to_unmask = self.scheduler.select_positions(
                step=step,
                total_steps=cfg.num_steps,
                current_mask=current_mask,
                is_unmasked=is_unmasked,
                logits=logits,
                confidences=confidences,
            )

            # Don't modify prompt positions
            positions_to_unmask = positions_to_unmask & ~prompt_mask

            # Sample tokens
            if positions_to_unmask.any():
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(batch_size, seq_len)
                x_t = torch.where(positions_to_unmask, sampled, x_t)
                is_unmasked = is_unmasked | positions_to_unmask

            if cfg.record_trajectory:
                trajectory.append(x_t.clone())

        # Final: unmask any remaining masked positions
        remaining_mask = (x_t == self.model.mask_token_id) & ~prompt_mask
        if remaining_mask.any():
            t = torch.full((batch_size,), 0.0, device=device)
            output = self.model.forward(x_t, t)
            probs = torch.softmax(output.logits / cfg.temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, seq_len)
            x_t = torch.where(remaining_mask, sampled, x_t)

        return SamplingResult(sequences=x_t, trajectory=trajectory)
