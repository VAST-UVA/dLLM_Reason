"""Curriculum scheduler — easy (high-frequency/certain) tokens first, hard tokens last."""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("curriculum")
class CurriculumScheduler(UnmaskingScheduler):
    """Curriculum-based unmasking: easy tokens first, hard tokens last.

    "Easiness" is measured by a combination of:
    1. Confidence (argmax probability) — high = easy
    2. Entropy (distribution spread) — low = easy

    The combined score balances both signals:
        score = confidence - alpha * entropy

    This mimics curriculum learning: resolve easy/obvious tokens first
    to provide more context for harder predictions later.

    Args:
        alpha: weight for entropy penalty (default 0.1).
               Higher alpha = more emphasis on low-entropy positions.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

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

        # Entropy from logits
        eps = 1e-8
        probs = torch.softmax(logits, dim=-1)                    # (B, L, V)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)  # (B, L)

        # Normalize entropy to [0, 1] range for stable combination
        ent_max = entropy[eligible].max().clamp(min=eps) if eligible.any() else 1.0
        entropy_norm = entropy / ent_max

        # Combined easiness score: high confidence, low entropy → easy
        score = confidences - self.alpha * entropy_norm

        # Ineligible → -inf
        score[~eligible] = -float("inf")

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            n = min(n_to_select, int(eligible[b].sum().item()))
            if n <= 0:
                continue
            _, top_idx = score[b].topk(n)
            result[b, top_idx] = True

        return result
