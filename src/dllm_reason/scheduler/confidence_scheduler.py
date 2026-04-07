"""Confidence-based unmasking scheduler (LLaDA default)."""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("confidence")
class ConfidenceScheduler(UnmaskingScheduler):
    """Commit the n_to_select highest-confidence masked positions.

    If block_mask is provided, only positions inside the current block
    are eligible — this implements LLaDA's block-wise denoising strategy.
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

        # Eligible = still masked AND (inside block if block_mask given)
        eligible = current_mask
        if block_mask is not None:
            eligible = eligible & block_mask

        # Set confidence to -inf for ineligible positions
        conf = confidences.clone()
        conf[~eligible] = -float("inf")

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            n = min(n_to_select, int(eligible[b].sum().item()))
            if n <= 0:
                continue
            _, top_idx = conf[b].topk(n)
            result[b, top_idx] = True

        return result
