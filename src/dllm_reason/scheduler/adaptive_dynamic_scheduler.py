"""Adaptive Dynamic DAG Scheduler — adjusts DAG constraints at runtime.

This is the core novel component: instead of using a static DAG that is
fixed before generation starts, the scheduler dynamically decides which
positions should depend on which, based on the model's evolving predictions.

Key idea: at each step, compute an "attention-like" influence score between
all masked position pairs. High influence means one position strongly
affects the other's prediction → add a soft dependency. This creates
a DAG that adapts to the actual content being generated.

This addresses the fundamental limitation of static DAGs: they assume
a fixed reasoning structure before seeing any content.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@SCHEDULER_REGISTRY.register("adaptive_dynamic")
class AdaptiveDynamicScheduler(UnmaskingScheduler):
    """Dynamically constructs and follows DAG constraints at each step.

    Algorithm:
    1. For all masked positions, compute pairwise influence scores
       from the logit distributions (KL divergence approximation).
    2. Build a soft dependency graph: position i should be unmasked
       before j if i has high influence on j AND i has higher confidence.
    3. Compute "readiness" = how many soft-dependencies are satisfied.
    4. Select top-n positions by readiness × confidence.

    Args:
        influence_threshold: minimum influence to create a soft dependency.
            Higher = fewer constraints, closer to pure confidence scheduling.
            Lower = more constraints, more structured generation order.
        momentum: EMA factor for smoothing influence scores across steps.
            0.0 = no memory (recompute each step)
            0.9 = strong smoothing (slowly adapting)
    """

    def __init__(
        self,
        influence_threshold: float = 0.3,
        momentum: float = 0.5,
    ) -> None:
        self.influence_threshold = influence_threshold
        self.momentum = momentum
        self._prev_influence: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_influence = None

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

        # ── Step 1: Compute influence between masked positions ────────────
        # Use logit distributions as proxy for how much positions depend
        # on each other: positions with similar top-k predictions are
        # likely interdependent.

        # Get top-k logit indices as a fingerprint for each position
        probs = F.softmax(logits, dim=-1)  # (B, L, V)

        # Pairwise cosine similarity of probability distributions
        # Only compute for eligible (masked) positions to save memory
        influence = self._compute_influence(probs, eligible)  # (B, L, L)

        # EMA smoothing with previous step
        if self._prev_influence is not None and self._prev_influence.shape == influence.shape:
            influence = (
                self.momentum * self._prev_influence +
                (1 - self.momentum) * influence
            )
        self._prev_influence = influence.detach()

        # ── Step 2: Build soft dependency graph ───────────────────────────
        # If position i has higher confidence than j AND strongly
        # influences j, then i should be unmasked before j.
        # readiness(j) = fraction of high-influence, higher-confidence
        # positions that are already unmasked.

        readiness = self._compute_readiness(
            influence, confidences, is_unmasked, eligible,
        )  # (B, L)

        # ── Step 3: Score = readiness × confidence ────────────────────────
        score = readiness * confidences
        score[~eligible] = -float("inf")

        # ── Step 4: Select top-n ──────────────────────────────────────────
        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            n = min(n_to_select, int(eligible[b].sum().item()))
            if n <= 0:
                continue
            _, top_idx = score[b].topk(n)
            result[b, top_idx] = True

        return result

    def _compute_influence(
        self,
        probs: torch.Tensor,      # (B, L, V)
        eligible: torch.Tensor,    # (B, L)
    ) -> torch.Tensor:
        """Compute pairwise influence between positions.

        Uses cosine similarity of probability distributions as a proxy
        for mutual information / influence.

        Returns: (B, L, L) influence matrix.
        """
        B, L, V = probs.shape
        device = probs.device

        # Normalize probability vectors for cosine similarity
        # Only compute for eligible positions (others get 0)
        probs_norm = F.normalize(probs, p=2, dim=-1)  # (B, L, V)

        # Zero out ineligible positions to save compute
        probs_norm = probs_norm * eligible.unsqueeze(-1).float()

        # Cosine similarity: (B, L, V) @ (B, V, L) → (B, L, L)
        influence = torch.bmm(probs_norm, probs_norm.transpose(1, 2))

        # Remove self-influence
        influence.diagonal(dim1=1, dim2=2).zero_()

        return influence.clamp(min=0)  # (B, L, L)

    def _compute_readiness(
        self,
        influence: torch.Tensor,     # (B, L, L)
        confidences: torch.Tensor,   # (B, L)
        is_unmasked: torch.Tensor,   # (B, L)
        eligible: torch.Tensor,      # (B, L)
    ) -> torch.Tensor:
        """Compute readiness score for each position.

        readiness(j) = fraction of positions that:
        1. Have high influence on j (influence[i,j] > threshold)
        2. Have higher confidence than j
        3. Are already unmasked

        High readiness = most "prerequisites" are already generated.
        """
        B, L = confidences.shape

        # Which positions i are "soft-parents" of position j?
        # parent(i, j) = influence(i, j) > threshold AND conf(i) > conf(j)
        high_influence = influence > self.influence_threshold  # (B, L, L)

        # conf_i > conf_j for all (i, j) pairs
        conf_higher = confidences.unsqueeze(-1) > confidences.unsqueeze(1)  # (B, L, L)

        soft_parents = high_influence & conf_higher  # (B, L, L)

        # For each j: what fraction of soft-parents are unmasked?
        parent_count = soft_parents.float().sum(dim=1)  # (B, L)
        parent_unmasked = (soft_parents & is_unmasked.unsqueeze(-1)).float().sum(dim=1)  # (B, L)

        # Readiness = fraction of satisfied parents (1.0 if no parents)
        readiness = torch.where(
            parent_count > 0,
            parent_unmasked / parent_count.clamp(min=1),
            torch.ones_like(parent_count),
        )

        return readiness
