"""Abstract base class for unmasking schedulers."""

from __future__ import annotations

import abc

import torch


class UnmaskingScheduler(abc.ABC):
    """Decides WHICH positions to commit at each denoising sub-step.

    The sampler calls select_positions once per sub-step and uses the
    returned boolean mask to commit x0 predictions into x.

    Subclasses implement different selection strategies:
      - ConfidenceScheduler : top-k by model confidence (LLaDA default)
      - RandomScheduler      : random selection
      - DAGScheduler         : DAG-readiness constrained selection
    """

    @abc.abstractmethod
    def select_positions(
        self,
        step: int,
        total_steps: int,
        current_mask: torch.Tensor,       # (B, L) True where still masked
        is_unmasked: torch.Tensor,         # (B, L) True where already committed
        logits: torch.Tensor,              # (B, L, V) raw model logits
        confidences: torch.Tensor,         # (B, L) per-position confidence
        block_mask: torch.Tensor | None = None,  # (B, L) True for current block
        n_to_select: int = 1,              # how many to commit this step
    ) -> torch.Tensor:                     # (B, L) bool — positions to commit
        ...

    def reset(self):
        """Reset internal state at the start of each generation."""
        pass
