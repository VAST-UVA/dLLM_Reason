"""Semi-autoregressive unmasking scheduler — left-to-right block-wise generation."""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("semi_ar")
class SemiAutoregressiveScheduler(UnmaskingScheduler):
    """Process tokens in fixed-size blocks from left to right.

    The sequence is divided into contiguous blocks of ``block_size``
    tokens.  Within each block, positions are unmasked by confidence
    (highest first), matching the behaviour of the confidence scheduler.
    Once every position in the current block is committed, the scheduler
    advances to the next block.

    This provides an intermediate baseline between fully parallel
    (confidence) and fully autoregressive generation.

    NOTE: This scheduler ignores the sampler's ``block_mask`` parameter
    and manages its own block boundaries.  This avoids conflicts with
    the sampler's block-wise denoising loop.
    """

    def __init__(self, block_size: int = 32) -> None:
        self.block_size = block_size
        self.current_block_start: int = 0

    def reset(self) -> None:
        """Reset block pointer at the start of each generation."""
        self.current_block_start = 0

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

        # Build a positional mask for the current block
        block_end = min(self.current_block_start + self.block_size, L)
        pos_mask = torch.zeros(L, dtype=torch.bool, device=device)
        pos_mask[self.current_block_start:block_end] = True
        pos_mask = pos_mask.unsqueeze(0).expand(B, -1)  # (B, L)

        # Intersect with sampler's block_mask if provided, so we only
        # touch positions the sampler has already designated for this round
        if block_mask is not None:
            eligible = current_mask & pos_mask & block_mask
        else:
            eligible = current_mask & pos_mask

        # If no eligible positions remain in the current block, advance
        while eligible.sum() == 0 and self.current_block_start < L:
            self.current_block_start += self.block_size
            block_end = min(self.current_block_start + self.block_size, L)
            pos_mask = torch.zeros(L, dtype=torch.bool, device=device)
            pos_mask[self.current_block_start:block_end] = True
            pos_mask = pos_mask.unsqueeze(0).expand(B, -1)
            eligible = current_mask & pos_mask

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
