"""Entropy-based unmasking scheduler — unmask lowest-entropy (most certain) positions first."""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("entropy")
class EntropyScheduler(UnmaskingScheduler):
    """Commit the n_to_select positions with lowest Shannon entropy.

    Unlike the confidence scheduler (which uses argmax probability),
    this scheduler measures uncertainty via the full entropy of the
    softmax distribution over the vocabulary.  Lower entropy means the
    model is more certain about its prediction.

    If block_mask is provided, only positions inside the current block
    are eligible.
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

        # Compute per-position entropy from logits: H = -sum(p * log(p + eps))
        eps = 1e-8
        probs = torch.softmax(logits, dim=-1)                    # (B, L, V)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)  # (B, L)

        # Set entropy to +inf for ineligible positions so they are never selected
        entropy[~eligible] = float("inf")

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            n = min(n_to_select, int(eligible[b].sum().item()))
            if n <= 0:
                continue
            # Lowest entropy → most certain; use topk on negated entropy
            _, top_idx = (-entropy[b]).topk(n)
            result[b, top_idx] = True

        return result
