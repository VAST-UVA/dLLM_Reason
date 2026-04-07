"""Random unmasking scheduler — baseline that unmasks positions uniformly at random."""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("random")
class RandomScheduler(UnmaskingScheduler):
    """Unmask positions uniformly at random.

    At each step, randomly selects a fraction of masked positions to unmask.
    This is the simplest baseline — no structure, no confidence.
    """

    def select_positions(
        self,
        step: int,
        total_steps: int,
        current_mask: torch.Tensor,
        is_unmasked: torch.Tensor,
        logits: torch.Tensor,
        confidences: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        n_to_select: int = 1,
    ) -> torch.Tensor:
        B, L = current_mask.shape
        device = current_mask.device

        eligible = current_mask & block_mask if block_mask is not None else current_mask
        result = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            indices = eligible[b].nonzero(as_tuple=False).squeeze(-1)
            if len(indices) == 0:
                continue
            n = min(n_to_select, len(indices))
            perm = torch.randperm(len(indices), device=device)[:n]
            result[b, indices[perm]] = True

        return result
