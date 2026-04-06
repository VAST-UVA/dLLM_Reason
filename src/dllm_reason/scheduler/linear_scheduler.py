"""Linear (left-to-right) unmasking scheduler.

Unmasks positions sequentially from left to right, mimicking
autoregressive generation order. This is a baseline that shows
how much structure the left-to-right order provides.
"""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("linear")
class LinearScheduler(UnmaskingScheduler):
    """Unmask positions left-to-right in sequential order."""

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

        num_masked = current_mask.sum(dim=-1)
        num_to_unmask = self._compute_num_to_unmask(step, total_steps, num_masked)

        result = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            # Find first N masked positions (leftmost)
            masked_indices = current_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if len(masked_indices) == 0:
                continue
            n = min(num_to_unmask[b].item(), len(masked_indices))
            # Already sorted since nonzero returns in order
            selected = masked_indices[:n]
            result[b, selected] = True

        return result
