"""Abstract base class for all discrete diffusion language models."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn as nn

from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DiffusionOutput:
    """Output of a diffusion model forward pass."""
    logits: torch.Tensor          # (batch, seq_len, vocab_size)
    loss: torch.Tensor | None     # scalar, if computed
    confidences: torch.Tensor | None  # (batch, seq_len), per-token confidence


class DiffusionLM(nn.Module, abc.ABC):
    """Abstract interface for discrete diffusion language models.

    All dLLM variants (MDLM, SEDD, D3PM, LLaDA) implement this interface.
    The key design: model predicts clean tokens from noisy input, and the
    unmasking order is determined externally by an UnmaskingScheduler.
    """

    def __init__(self, vocab_size: int, max_seq_len: int, mask_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id

    @abc.abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        """Predict clean tokens from noisy input at timestep t.

        Args:
            x_t: (batch, seq_len) noisy token ids
            t: (batch,) or scalar, continuous timestep in [0, 1]
            attention_mask: (batch, seq_len) optional

        Returns:
            DiffusionOutput with logits (batch, seq_len, vocab_size)
        """
        ...

    @abc.abstractmethod
    def compute_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute training loss from clean data.

        Internally samples t and applies noise, then computes the diffusion loss.

        Args:
            x_0: (batch, seq_len) clean token ids
            attention_mask: (batch, seq_len) optional

        Returns:
            Scalar loss tensor.
        """
        ...

    @abc.abstractmethod
    def noise_input(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Apply forward diffusion noise at timestep t.

        For absorbing-state models: replace tokens with MASK.
        For multinomial models: sample from transition matrix.

        Args:
            x_0: (batch, seq_len) clean token ids
            t: (batch,) continuous timestep in [0, 1]

        Returns:
            x_t: (batch, seq_len) noisy token ids
        """
        ...

    def get_token_confidences(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-position confidence scores for scheduler use.

        Default: max softmax probability at each position.

        Args:
            x_t: (batch, seq_len) noisy token ids
            t: (batch,) timestep
            attention_mask: optional

        Returns:
            (batch, seq_len) confidence scores in [0, 1]
        """
        with torch.no_grad():
            output = self.forward(x_t, t, attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            confidences = probs.max(dim=-1).values
        return confidences

    def sample(
        self,
        scheduler,
        batch_size: int = 1,
        seq_len: int | None = None,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        num_steps: int = 64,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Generate sequences using the given unmasking scheduler.

        This is the generic sampling loop. The scheduler determines which
        positions to unmask at each step — this is the injection point for
        DAG constraints.

        Args:
            scheduler: UnmaskingScheduler instance
            batch_size: number of sequences to generate
            seq_len: sequence length (defaults to max_seq_len)
            prompt_ids: (batch, prompt_len) optional prompt token ids
            prompt_mask: (batch, seq_len) bool, True for prompt positions
            num_steps: number of diffusion steps
            temperature: sampling temperature
            device: device to generate on

        Returns:
            (batch, seq_len) generated token ids
        """
        seq_len = seq_len or self.max_seq_len

        # Initialize fully masked
        x_t = torch.full(
            (batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        # Fill in prompt if provided
        if prompt_ids is not None and prompt_mask is not None:
            x_t[prompt_mask] = prompt_ids[prompt_mask]

        # Track which positions are already unmasked
        is_unmasked = (x_t != self.mask_token_id)

        self.eval()
        with torch.no_grad():
            for step in range(num_steps):
                # Current timestep: goes from ~1 (fully noisy) to ~0 (clean)
                t_val = 1.0 - step / num_steps
                t = torch.full((batch_size,), t_val, device=device)

                # Get model predictions
                output = self.forward(x_t, t)
                logits = output.logits.clone()  # (batch, seq_len, vocab_size)

                # Suppress mask token so it can never be sampled as output
                if self.mask_token_id < logits.shape[-1]:
                    logits[..., self.mask_token_id] = -float("inf")

                # Compute confidences from clean distribution
                probs = torch.softmax(logits / temperature, dim=-1)
                confidences = probs.max(dim=-1).values  # (batch, seq_len)

                # Current mask: True where still masked
                current_mask = (x_t == self.mask_token_id)

                # Ask scheduler which positions to unmask
                positions_to_unmask = scheduler.select_positions(
                    step=step,
                    total_steps=num_steps,
                    current_mask=current_mask,
                    is_unmasked=is_unmasked,
                    logits=logits,
                    confidences=confidences,
                )

                # Sample tokens for selected positions
                if positions_to_unmask.any():
                    sampled = torch.multinomial(
                        probs.view(-1, probs.shape[-1]), num_samples=1
                    ).view(batch_size, seq_len)
                    x_t = torch.where(positions_to_unmask, sampled, x_t)
                    is_unmasked = is_unmasked | positions_to_unmask

        return x_t

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
