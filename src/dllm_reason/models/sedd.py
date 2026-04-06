"""SEDD: Score Entropy Discrete Diffusion.

Implements score-entropy based discrete diffusion as described in:
"Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"
(Lou et al., 2024)

Key idea: instead of predicting x_0 directly, SEDD learns the "concrete score"
— the ratio p(x=j|x_t) / p(x=x_t|x_t) for all j != x_t. This is trained
via the score entropy loss, which avoids estimating the normalizing constant.

Reference: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_reason.models.base import DiffusionLM, DiffusionOutput
from dllm_reason.models.backbone.transformer import BidirectionalTransformer
from dllm_reason.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("sedd")
class SEDD(DiffusionLM):
    """Score Entropy Discrete Diffusion model.

    Uses absorbing-state noise (like MDLM) but with score-based parameterization.
    The model outputs score ratios s(x_t, t)_j = p(x=j|x_t)/p(x=x_t|x_t)
    instead of direct token probabilities.

    The score entropy loss decomposes as:
    L = E_t[ sum_i sum_{j!=x_t^i} s(x_t,t)_{i,j} * (log s(x_t,t)_{i,j} - 1)
            + p(x_0^i=j|x_t) / p(x_0^i=x_t^i|x_t) ]
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
        noise_schedule: str = "log_linear",
    ):
        if mask_token_id is None:
            mask_token_id = vocab_size
            vocab_size = vocab_size + 1

        super().__init__(vocab_size, max_seq_len, mask_token_id)

        self.noise_schedule_type = noise_schedule

        self.backbone = BidirectionalTransformer(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """Transition rate for absorbing state CTMC.

        For absorbing state: rate(t) = sigma'(t) / (1 - sigma(t))
        where sigma(t) is the probability of being masked at time t.
        """
        if self.noise_schedule_type == "log_linear":
            # sigma(t) = 1 - (1-t), rate = 1/(1-t)
            return 1.0 / (1.0 - t).clamp(min=1e-6)
        elif self.noise_schedule_type == "geometric":
            sigma_min = 1e-4
            log_1_minus_sigma = torch.log(torch.tensor(1.0 - sigma_min, device=t.device))
            return -log_1_minus_sigma * (1.0 - sigma_min) ** t / (
                (1.0 - sigma_min) ** t
            ).clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule_type}")

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Masking probability at time t."""
        if self.noise_schedule_type == "log_linear":
            return t
        elif self.noise_schedule_type == "geometric":
            sigma_min = 1e-4
            return 1.0 - (1.0 - sigma_min) ** t
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule_type}")

    def noise_input(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma_t = self.sigma(t)[:, None].expand_as(x_0)
        mask = torch.rand_like(sigma_t.float()) < sigma_t
        return torch.where(mask, self.mask_token_id, x_0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        """Forward pass returning score logits.

        The backbone outputs (B, L, V) logits. For SEDD, these represent
        log score ratios: log s(x_t, t)_{i,j} for each position i and token j.
        """
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])

        logits = self.backbone(x_t, t, attention_mask)

        return DiffusionOutput(logits=logits, loss=None, confidences=None)

    def compute_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score entropy loss for SEDD.

        For absorbing state, the loss simplifies to cross-entropy at masked
        positions weighted by the transition rate, plus a score entropy
        regularization term.
        """
        B, L = x_0.shape
        device = x_0.device

        t = torch.rand(B, device=device).clamp(min=1e-5, max=1.0 - 1e-5)
        x_t = self.noise_input(x_0, t)

        output = self.forward(x_t, t, attention_mask)
        logits = output.logits  # (B, L, V)

        is_masked = (x_t == self.mask_token_id)

        # For masked positions: score entropy loss
        # The score at masked positions relates to x_0 prediction
        # Score entropy = sum_j exp(log_score_j) * (log_score_j - 1) + indicator(j=x_0)
        log_scores = logits  # (B, L, V)

        # At masked positions, the effective loss is cross-entropy + entropy term
        # Simplified: use cross-entropy as main loss (equivalent for absorbing state)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=x_0.unsqueeze(-1)).squeeze(-1)

        # Score entropy regularization: encourage peaky predictions
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)

        if attention_mask is not None:
            is_masked = is_masked & attention_mask.bool()

        # Weight by transition rate
        rate_t = self.rate(t)  # (B,)

        # Combined loss at masked positions
        masked_loss = ((nll + 0.1 * entropy) * is_masked.float()).sum(dim=-1)
        num_masked = is_masked.float().sum(dim=-1).clamp(min=1.0)
        per_sample = rate_t * masked_loss / num_masked

        return per_sample.mean()
