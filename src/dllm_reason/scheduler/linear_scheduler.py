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
            # Leftmost first
            selected = indices[:min(n_to_select, len(indices))]
            result[b, selected] = True

        return result
