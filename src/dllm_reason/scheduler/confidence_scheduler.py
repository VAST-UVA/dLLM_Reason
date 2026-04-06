"""Confidence-based unmasking scheduler.

Unmasks positions in order of model confidence (highest confidence first).
This is a strong baseline — the model decides its own unmasking order
based on how certain it is about each position.
"""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("confidence")
class ConfidenceScheduler(UnmaskingScheduler):
    """Unmask positions with highest model confidence first.

    At each step, among all still-masked positions, select those with
    the highest confidence (max softmax probability) for unmasking.
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

        num_masked = current_mask.sum(dim=-1)
        num_to_unmask = self._compute_num_to_unmask(step, total_steps, num_masked)

        # Mask out already-unmasked positions by setting their confidence to -inf
        masked_confidences = confidences.clone()
        masked_confidences[~current_mask] = -float("inf")

        result = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            n = min(num_to_unmask[b].item(), num_masked[b].item())
            if n <= 0:
                continue
            _, top_indices = masked_confidences[b].topk(n)
            result[b, top_indices] = True

        return result
