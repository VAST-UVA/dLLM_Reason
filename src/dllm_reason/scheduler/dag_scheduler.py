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
        self._prompt_len: int | None = None  # auto-detected on first call

    def reset(self):
        """Reset internal state."""
        self._prompt_len = None

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
        dag_len = self.dag.seq_len

        # ── Dimension alignment ──────────────────────────────────────────
        # The DAG covers only the generation area (dag_len positions),
        # but current_mask / is_unmasked cover the full sequence
        # (prompt_len + gen_len).  We slice out the generation part,
        # query the DAG, then pad back.
        if L > dag_len:
            # Auto-detect prompt length on first call
            if self._prompt_len is None:
                self._prompt_len = L - dag_len

            pl = self._prompt_len
            gen_unmasked = is_unmasked[:, pl:pl + dag_len]  # (B, dag_len)
            ready_gen = self.dag.ready_positions(gen_unmasked)  # (B, dag_len)

            # Pad to full sequence: prompt positions are always "ready"
            ready = torch.ones(B, L, dtype=torch.bool, device=device)
            ready[:, pl:pl + dag_len] = ready_gen
        else:
            # Sequence matches DAG size exactly
            ready = self.dag.ready_positions(is_unmasked)  # (B, L)

        # 2. Eligible = ready AND still masked
        eligible = ready & current_mask  # (B, L)

        # 3. Apply sub-strategy
        if self.sub_strategy == "all_ready":
            return eligible

        elif self.sub_strategy == "confidence_topk":
            return self._confidence_topk(eligible, confidences, n_to_select)

        elif self.sub_strategy == "proportional":
            return self._proportional(eligible, n_to_select)

        else:
            raise ValueError(f"Unknown sub-strategy: {self.sub_strategy}")

    def _confidence_topk(
        self,
        eligible: torch.Tensor,
        confidences: torch.Tensor,
        n_to_select: int,
    ) -> torch.Tensor:
        """Among eligible positions, select top-n by confidence."""
        B, L = eligible.shape
        device = eligible.device

        conf = confidences.clone()
        conf[~eligible] = -float("inf")

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            n = min(n_to_select, int(eligible[b].sum().item()))
            if n <= 0:
                continue
            _, top_indices = conf[b].topk(n)
            result[b, top_indices] = True

        return result

    def _proportional(
        self,
        eligible: torch.Tensor,
        n_to_select: int,
    ) -> torch.Tensor:
        """Randomly select n_to_select from eligible positions."""
        B, L = eligible.shape
        device = eligible.device

        result = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            indices = eligible[b].nonzero(as_tuple=False).squeeze(-1)
            if len(indices) == 0:
                continue
            n = min(n_to_select, len(indices))
            perm = torch.randperm(len(indices), device=device)[:n]
            result[b, indices[perm]] = True

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
        self._prompt_len: int | None = None

    def reset(self):
        self._prompt_len = None

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
        dag_len = self.dag.seq_len

        # Dimension alignment (same logic as DAGScheduler)
        if L > dag_len:
            if self._prompt_len is None:
                self._prompt_len = L - dag_len
            pl = self._prompt_len
            gen_unmasked = is_unmasked[:, pl:pl + dag_len]
            ready_gen = self.dag.ready_positions(gen_unmasked)
            ready = torch.ones(B, L, dtype=torch.bool, device=device)
            ready[:, pl:pl + dag_len] = ready_gen
        else:
            ready = self.dag.ready_positions(is_unmasked)

        eligible = ready & current_mask

        result = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            eligible_idx = eligible[b].nonzero(as_tuple=False).squeeze(-1)

            if len(eligible_idx) == 0:
                # No DAG-ready positions — bypass with highest-confidence masked token
                masked_idx = current_mask[b].nonzero(as_tuple=False).squeeze(-1)
                if len(masked_idx) == 0:
                    continue
                n_bypass = max(1, int(len(masked_idx) * self.bypass_fraction))
                conf_b = confidences[b].clone()
                conf_b[~current_mask[b]] = -float("inf")
                _, bypass_idx = conf_b.topk(min(n_bypass, len(masked_idx)))
                result[b, bypass_idx] = True
                continue

            # Select top-confidence among DAG-ready eligible positions
            n = min(n_to_select, len(eligible_idx))
            conf_eligible = confidences[b, eligible_idx]
            _, top_local = conf_eligible.topk(n)
            result[b, eligible_idx[top_local]] = True

        return result
