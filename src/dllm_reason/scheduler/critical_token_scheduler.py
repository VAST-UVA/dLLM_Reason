"""Critical-token-first scheduler — unmask tokens with highest influence first."""

from __future__ import annotations

import torch

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("critical_token_first")
class CriticalTokenFirstScheduler(UnmaskingScheduler):
    """Unmask tokens that have the most influence on other positions first.

    Influence is measured by how much each masked position affects the
    logit distributions of other masked positions.  Approximated by:

        influence(i) = sum_j |logit_gradient(j, i)|

    In practice, we use a cheaper proxy: the KL divergence between the
    model's prediction at position i and the uniform distribution.
    High KL = the model is very opinionated about this position =
    unmasking it will propagate the most information.

    Among eligible positions, select the top-n by influence score.
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
        B, L, V = logits.shape
        device = current_mask.device

        eligible = current_mask
        if block_mask is not None:
            eligible = eligible & block_mask

        # Compute KL divergence from uniform: KL(p || uniform)
        # = sum_v p(v) * log(p(v) * V) = log(V) + sum_v p(v) log p(v)
        # = log(V) - H(p)   where H is entropy
        eps = 1e-8
        probs = torch.softmax(logits, dim=-1)                    # (B, L, V)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)  # (B, L)
        log_V = torch.log(torch.tensor(float(V), device=device))
        influence = log_V - entropy  # KL from uniform; higher = more informative

        # Ineligible → -inf
        influence[~eligible] = -float("inf")

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            n = min(n_to_select, int(eligible[b].sum().item()))
            if n <= 0:
                continue
            _, top_idx = influence[b].topk(n)
            result[b, top_idx] = True

        return result
