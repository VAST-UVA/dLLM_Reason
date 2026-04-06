"""Abstract base class for unmasking schedulers.

The scheduler determines WHICH positions to unmask at each diffusion step.
This is the key abstraction that decouples the unmasking ORDER from the
model's token PREDICTIONS.

All schedulers (random, confidence, left-to-right, DAG-guided) implement
the same interface, allowing clean comparison.
"""

from __future__ import annotations

import abc

import torch


class UnmaskingScheduler(abc.ABC):
    """Abstract interface for unmasking position selection.

    At each diffusion step, the scheduler receives the current state
    and returns a boolean mask of positions to unmask.
    """

    @abc.abstractmethod
    def select_positions(
        self,
        step: int,
        total_steps: int,
        current_mask: torch.Tensor,
        is_unmasked: torch.Tensor,
        logits: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """Select which positions to unmask at this step.

        Args:
            step: current step index (0 to total_steps-1)
            total_steps: total number of diffusion steps
            current_mask: (batch, seq_len) bool, True where still masked
            is_unmasked: (batch, seq_len) bool, True where already unmasked
            logits: (batch, seq_len, vocab_size) model predictions
            confidences: (batch, seq_len) per-position confidence scores

        Returns:
            (batch, seq_len) bool tensor, True for positions to unmask now
        """
        ...

    def reset(self):
        """Reset any internal state (called at the start of each generation)."""
        pass

    def _compute_num_to_unmask(
        self,
        step: int,
        total_steps: int,
        num_masked: torch.Tensor,
    ) -> torch.Tensor:
        """Compute how many positions to unmask at this step.

        Default schedule: unmask roughly equal numbers per step,
        ensuring all are unmasked by the last step.

        Args:
            step: current step
            total_steps: total steps
            num_masked: (batch,) number of currently masked positions

        Returns:
            (batch,) number of positions to unmask
        """
        # Linear schedule: fraction of remaining to unmask
        remaining_steps = total_steps - step
        if remaining_steps <= 0:
            return num_masked
        num_to_unmask = (num_masked.float() / remaining_steps).ceil().long()
        return num_to_unmask.clamp(min=1)
