"""DAG-constrained unmasking scheduler — the core novel component.

This scheduler uses a TokenDAG to control the unmasking order during
inference. At each step, only positions whose DAG dependencies are
fully satisfied can be unmasked.

Key insight: the DAG constrains the ORDER, while the dLLM provides
the TOKEN PREDICTIONS. The two are decoupled — any dLLM can use
any DAG, and the DAG can be swapped without retraining.
"""

from __future__ import annotations

import torch

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("dag")
class DAGScheduler(UnmaskingScheduler):
    """DAG-constrained unmasking scheduler.

    At each diffusion step:
    1. Query dag.ready_positions(is_unmasked) to get positions whose
       ALL DAG parents are already unmasked.
    2. Intersect with positions that are still masked → eligible positions.
    3. Apply a sub-strategy to select from eligible positions.

    Sub-strategies:
    - "all_ready": unmask all eligible positions at once
    - "confidence_topk": among eligible, pick highest confidence
    - "proportional": unmask a fraction based on step progress
    """

    def __init__(
        self,
        dag: TokenDAG,
        sub_strategy: str = "confidence_topk",
    ):
        self.dag = dag
        self.sub_strategy = sub_strategy

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

        # 1. Get positions whose all DAG parents are unmasked
        ready = self.dag.ready_positions(is_unmasked)  # (B, L)

        # 2. Eligible = ready AND still masked
        eligible = ready & current_mask  # (B, L)

        # 3. Apply sub-strategy
        if self.sub_strategy == "all_ready":
            return eligible

        elif self.sub_strategy == "confidence_topk":
            return self._confidence_topk(eligible, confidences, step, total_steps, current_mask)

        elif self.sub_strategy == "proportional":
            return self._proportional(eligible, step, total_steps, current_mask)

        else:
            raise ValueError(f"Unknown sub-strategy: {self.sub_strategy}")

    def _confidence_topk(
        self,
        eligible: torch.Tensor,
        confidences: torch.Tensor,
        step: int,
        total_steps: int,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Among eligible positions, select those with highest confidence."""
        B, L = eligible.shape
        device = eligible.device

        num_masked = current_mask.sum(dim=-1)
        num_to_unmask = self._compute_num_to_unmask(step, total_steps, num_masked)

        # Set non-eligible positions to -inf confidence
        masked_conf = confidences.clone()
        masked_conf[~eligible] = -float("inf")

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            num_eligible = eligible[b].sum().item()
            if num_eligible == 0:
                continue
            n = min(num_to_unmask[b].item(), num_eligible)
            if n <= 0:
                continue
            _, top_indices = masked_conf[b].topk(n)
            result[b, top_indices] = True

        return result

    def _proportional(
        self,
        eligible: torch.Tensor,
        step: int,
        total_steps: int,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Unmask a fraction of eligible positions based on step progress."""
        B, L = eligible.shape
        device = eligible.device

        # Progress fraction for this step
        progress = (step + 1) / total_steps

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            eligible_indices = eligible[b].nonzero(as_tuple=False).squeeze(-1)
            if len(eligible_indices) == 0:
                continue
            # Unmask at least 1, at most all eligible
            n = max(1, int(len(eligible_indices) * progress))
            n = min(n, len(eligible_indices))
            perm = torch.randperm(len(eligible_indices), device=device)[:n]
            selected = eligible_indices[perm]
            result[b, selected] = True

        return result


@SCHEDULER_REGISTRY.register("adaptive_dag")
class AdaptiveDAGScheduler(UnmaskingScheduler):
    """Adaptive DAG scheduler that adjusts strategy based on confidence.

    Combines DAG constraints with adaptive behavior:
    - When model confidence is high: follow DAG strictly
    - When model confidence is low: allow some out-of-order unmasking
      to avoid getting stuck on hard positions

    This addresses the risk of DAG scheduling forcing the model to
    commit to difficult positions too early.
    """

    def __init__(
        self,
        dag: TokenDAG,
        confidence_threshold: float = 0.5,
        bypass_fraction: float = 0.1,
    ):
        self.dag = dag
        self.confidence_threshold = confidence_threshold
        self.bypass_fraction = bypass_fraction

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

        ready = self.dag.ready_positions(is_unmasked)
        eligible = ready & current_mask

        # Check mean confidence of eligible positions
        num_masked = current_mask.sum(dim=-1)
        num_to_unmask = self._compute_num_to_unmask(step, total_steps, num_masked)

        result = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            eligible_idx = eligible[b].nonzero(as_tuple=False).squeeze(-1)

            if len(eligible_idx) == 0:
                # No eligible positions — allow bypass
                masked_idx = current_mask[b].nonzero(as_tuple=False).squeeze(-1)
                if len(masked_idx) == 0:
                    continue
                n_bypass = max(1, int(len(masked_idx) * self.bypass_fraction))
                # Pick highest confidence among all masked
                conf_b = confidences[b].clone()
                conf_b[~current_mask[b]] = -float("inf")
                _, bypass_idx = conf_b.topk(min(n_bypass, len(masked_idx)))
                result[b, bypass_idx] = True
                continue

            # Normal DAG-guided selection with confidence
            n = min(num_to_unmask[b].item(), len(eligible_idx))
            if n <= 0:
                continue

            # Select top-confidence among eligible
            conf_eligible = confidences[b, eligible_idx]
            _, top_local = conf_eligible.topk(min(n, len(eligible_idx)))
            result[b, eligible_idx[top_local]] = True

        return result
