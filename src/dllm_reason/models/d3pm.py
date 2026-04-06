"""D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces.

Implements discrete-time discrete diffusion with structured transition matrices.
"Structured Denoising Diffusion Models in Discrete State-Spaces"
(Austin et al., 2021)

Key idea: instead of continuous-time absorbing state, D3PM uses discrete
timesteps with transition matrices Q_t. Supports multiple noise types:
- Absorbing: tokens transition to an absorbing [MASK] state
- Uniform: tokens transition uniformly to any other token
- Token-frequency: tokens transition proportionally to corpus frequencies

Reference: https://arxiv.org/abs/2107.03006
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_reason.models.base import DiffusionLM, DiffusionOutput
from dllm_reason.models.backbone.transformer import BidirectionalTransformer
from dllm_reason.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("d3pm")
class D3PM(DiffusionLM):
    """D3PM with configurable transition matrices.

    Uses discrete timesteps T (e.g., 1000). At each step, the forward
    process applies a transition matrix Q_t:
        q(x_t | x_{t-1}) = Cat(x_t; p = x_{t-1} @ Q_t)

    The cumulative transition is:
        q(x_t | x_0) = Cat(x_t; p = x_0 @ Q_bar_t)
    where Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t.
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
        num_timesteps: int = 1000,
        transition_type: str = "absorbing",
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        hybrid_lambda: float = 0.01,
    ):
        if mask_token_id is None:
            mask_token_id = vocab_size
            vocab_size = vocab_size + 1

        super().__init__(vocab_size, max_seq_len, mask_token_id)

        self.num_timesteps = num_timesteps
        self.transition_type = transition_type
        self.hybrid_lambda = hybrid_lambda

        self.backbone = BidirectionalTransformer(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Pre-compute noise schedule
        betas = torch.linspace(beta_min, beta_max, num_timesteps)
        self.register_buffer("betas", betas)

        # Pre-compute cumulative transition matrices
        # For absorbing state: Q_bar_t[i, mask] accumulates probability
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _get_transition_probs(self, t_index: torch.Tensor) -> torch.Tensor:
        """Get q(x_t | x_0) transition probabilities.

        For absorbing state:
            P(x_t = x_0) = alpha_bar_t
            P(x_t = MASK) = 1 - alpha_bar_t  (if x_0 != MASK)

        Args:
            t_index: (batch,) integer timestep indices in [0, T-1]

        Returns:
            alpha_bar: (batch,) probability of staying at x_0
        """
        return self.alphas_cumprod[t_index]

    def noise_input(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply discrete-time noise.

        Args:
            x_0: (batch, seq_len) clean tokens
            t: (batch,) continuous timestep in [0, 1], will be mapped to discrete index
        """
        # Map continuous t to discrete index
        t_index = (t * (self.num_timesteps - 1)).long().clamp(0, self.num_timesteps - 1)

        if self.transition_type == "absorbing":
            alpha_bar = self._get_transition_probs(t_index)  # (batch,)
            keep_prob = alpha_bar[:, None].expand_as(x_0)
            mask = torch.rand_like(keep_prob.float()) > keep_prob
            x_t = torch.where(mask, self.mask_token_id, x_0)
        elif self.transition_type == "uniform":
            alpha_bar = self._get_transition_probs(t_index)
            keep_prob = alpha_bar[:, None].expand_as(x_0)
            uniform_prob = (1.0 - keep_prob) / self.vocab_size
            # With probability keep_prob stay, otherwise sample uniformly
            mask = torch.rand_like(keep_prob.float()) > keep_prob
            random_tokens = torch.randint(0, self.vocab_size, x_0.shape, device=x_0.device)
            x_t = torch.where(mask, random_tokens, x_0)
        else:
            raise ValueError(f"Unknown transition type: {self.transition_type}")

        return x_t

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])

        logits = self.backbone(x_t, t, attention_mask)
        return DiffusionOutput(logits=logits, loss=None, confidences=None)

    def compute_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """D3PM loss: KL divergence + auxiliary cross-entropy.

        L = E_t[ KL(q(x_{t-1}|x_t, x_0) || p_theta(x_{t-1}|x_t)) ]
            + lambda * CE(x_0, p_theta(x_0|x_1))

        For absorbing state, this simplifies significantly.
        """
        B, L = x_0.shape
        device = x_0.device

        # Sample discrete timestep
        t = torch.rand(B, device=device).clamp(min=1e-5, max=1.0 - 1e-5)
        x_t = self.noise_input(x_0, t)

        output = self.forward(x_t, t, attention_mask)
        logits = output.logits

        # Reconstruction loss (cross-entropy)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=x_0.unsqueeze(-1)).squeeze(-1)

        # For absorbing state, VLB simplifies to weighted CE at masked positions
        is_masked = (x_t == self.mask_token_id)
        if attention_mask is not None:
            is_masked = is_masked & attention_mask.bool()

        # VLB term: CE at masked positions
        vlb_loss = (nll * is_masked.float()).sum(dim=-1)
        num_masked = is_masked.float().sum(dim=-1).clamp(min=1.0)
        vlb_loss = (vlb_loss / num_masked).mean()

        # Auxiliary reconstruction loss at all positions
        if attention_mask is not None:
            aux_loss = (nll * attention_mask.float()).sum(dim=-1)
            num_tokens = attention_mask.float().sum(dim=-1).clamp(min=1.0)
            aux_loss = (aux_loss / num_tokens).mean()
        else:
            aux_loss = nll.mean()

        return vlb_loss + self.hybrid_lambda * aux_loss
