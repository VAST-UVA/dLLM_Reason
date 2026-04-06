"""MDLM: Masked Discrete Language Model.

Implements absorbing-state continuous-time discrete diffusion as described in:
"Simple and Effective Masked Diffusion Language Models" (Sahoo et al., 2024)

Key idea: tokens are noised by replacing them with [MASK] with probability
that increases with timestep t. The model learns to predict the original
tokens from the masked input. Uses SUBS (substitution) parameterization.

Reference: https://github.com/kuleshov-group/mdlm
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_reason.models.base import DiffusionLM, DiffusionOutput
from dllm_reason.models.backbone.transformer import BidirectionalTransformer
from dllm_reason.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("mdlm")
class MDLM(DiffusionLM):
    """Masked Discrete Language Model with absorbing-state diffusion.

    Forward process: q(x_t | x_0) — each token independently stays as x_0
    with probability (1 - sigma(t)), or becomes MASK with probability sigma(t).

    sigma(t) is the noise schedule, typically sigma(t) = 1 - (1-sigma_min)^t
    for geometric schedule, or sigma(t) = t for linear schedule.

    Reverse process: model predicts p(x_0 | x_t), then we can compute
    the posterior p(x_{t-dt} | x_t, x_0).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 1024,
        mask_token_id: int | None = None,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        noise_schedule: str = "geometric",
        sigma_min: float = 1e-4,
    ):
        # Reserve last token id as MASK if not specified
        if mask_token_id is None:
            mask_token_id = vocab_size  # add one extra token for MASK
            vocab_size = vocab_size + 1

        super().__init__(vocab_size, max_seq_len, mask_token_id)

        self.noise_schedule = noise_schedule
        self.sigma_min = sigma_min

        self.backbone = BidirectionalTransformer(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise schedule: probability of being masked at timestep t.

        Args:
            t: (batch,) in [0, 1], where t=0 is clean and t=1 is fully masked
        Returns:
            (batch,) sigma(t) in [0, 1]
        """
        if self.noise_schedule == "geometric":
            return 1.0 - (1.0 - self.sigma_min) ** t
        elif self.noise_schedule == "linear":
            return t
        elif self.noise_schedule == "cosine":
            return 1.0 - torch.cos(t * torch.pi / 2)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def dsigma(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of noise schedule d(sigma)/dt.

        Needed for continuous-time ELBO computation.
        """
        if self.noise_schedule == "geometric":
            return -(1.0 - self.sigma_min) ** t * torch.log(
                torch.tensor(1.0 - self.sigma_min, device=t.device)
            )
        elif self.noise_schedule == "linear":
            return torch.ones_like(t)
        elif self.noise_schedule == "cosine":
            return torch.sin(t * torch.pi / 2) * torch.pi / 2
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def noise_input(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply absorbing-state noise: replace tokens with MASK.

        Each token independently becomes MASK with probability sigma(t).
        """
        sigma_t = self.sigma(t)  # (batch,)
        # (batch, 1) for broadcasting over seq_len
        mask_prob = sigma_t[:, None].expand_as(x_0)
        # Sample mask: True where token should be masked
        mask = torch.rand_like(mask_prob.float()) < mask_prob
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        """Predict clean tokens from masked input."""
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])

        logits = self.backbone(x_t, t, attention_mask)

        return DiffusionOutput(
            logits=logits,
            loss=None,
            confidences=None,
        )

    def compute_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the continuous-time ELBO loss for MDLM.

        L = E_t[ dsigma(t)/sigma(t) * sum_i 1[x_t^i = MASK] *
              (-log p(x_0^i | x_t)) ]

        This is a Monte Carlo estimate with a single t sample per batch element.
        """
        B, L = x_0.shape
        device = x_0.device

        # Sample timestep uniformly from (0, 1)
        t = torch.rand(B, device=device).clamp(min=1e-5, max=1.0 - 1e-5)

        # Apply noise
        x_t = self.noise_input(x_0, t)

        # Forward pass
        output = self.forward(x_t, t, attention_mask)
        logits = output.logits  # (B, L, V)

        # Cross-entropy loss only at masked positions
        is_masked = (x_t == self.mask_token_id)  # (B, L)

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)

        # Gather log prob of true token at each position
        # x_0: (B, L) -> (B, L, 1) for gather
        nll = -log_probs.gather(dim=-1, index=x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Mask: only count loss at masked positions
        if attention_mask is not None:
            is_masked = is_masked & attention_mask.bool()

        # Weight by dsigma/sigma (importance weight for continuous-time ELBO)
        sigma_t = self.sigma(t)  # (B,)
        dsigma_t = self.dsigma(t)  # (B,)
        weight = dsigma_t / sigma_t.clamp(min=1e-8)  # (B,)

        # Compute loss: weighted NLL at masked positions
        masked_nll = (nll * is_masked.float()).sum(dim=-1)  # (B,)
        num_masked = is_masked.float().sum(dim=-1).clamp(min=1.0)  # (B,)
        per_sample_loss = weight * masked_nll / num_masked  # (B,)

        return per_sample_loss.mean()
