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
    ) -> torch.Tensor:
        B, L = current_mask.shape
        device = current_mask.device

        num_masked = current_mask.sum(dim=-1)  # (B,)
        num_to_unmask = self._compute_num_to_unmask(step, total_steps, num_masked)

        # For each batch element, randomly select positions to unmask
        result = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            masked_indices = current_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if len(masked_indices) == 0:
                continue
            n = min(num_to_unmask[b].item(), len(masked_indices))
            perm = torch.randperm(len(masked_indices), device=device)[:n]
            selected = masked_indices[perm]
            result[b, selected] = True

        return result
