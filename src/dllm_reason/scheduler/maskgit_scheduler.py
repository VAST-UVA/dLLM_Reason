"""MaskGIT-style cosine scheduler — unmask a cosine-scheduled fraction per step."""

from __future__ import annotations

import math

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("maskgit_cosine")
class MaskGITCosineScheduler(UnmaskingScheduler):
    """Unmask a cosine-scheduled fraction of tokens per step.

    At step t out of T total steps, the cumulative fraction unmasked is:
        ratio(t) = cos(pi/2 * (1 - t/T))

    This means more tokens are unmasked early (when the model is less
    constrained) and fewer later (for refinement).  Within each step,
    positions are selected by highest confidence.

    Reference: Chang et al., "MaskGIT: Masked Generative Image
    Transformer", CVPR 2022.
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

        eligible = current_mask
        if block_mask is not None:
            eligible = eligible & block_mask

        # Cosine schedule: how many should be unmasked by end of this step
        # ratio goes from 0 (step=0) to 1 (step=total_steps)
        ratio_now = math.cos(math.pi / 2 * (1 - (step + 1) / total_steps))
        ratio_prev = math.cos(math.pi / 2 * (1 - step / total_steps)) if step > 0 else 0.0

        result = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            num_eligible = int(eligible[b].sum().item())
            if num_eligible == 0:
                continue

            total_in_block = int((block_mask[b] if block_mask is not None else
                                  torch.ones(L, dtype=torch.bool, device=device)).sum().item())

            # Number to unmask THIS step = delta of cumulative schedule
            n_target = max(1, int(round((ratio_now - ratio_prev) * total_in_block)))
            n = min(n_target, num_eligible)

            conf = confidences[b].clone()
            conf[~eligible[b]] = -float("inf")
            _, top_idx = conf.topk(n)
            result[b, top_idx] = True

        return result
