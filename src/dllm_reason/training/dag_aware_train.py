"""DAG-aware training: bias masking during training to match inference.

Standard dLLM training masks tokens uniformly at random. This creates a
train-inference mismatch when using DAG-guided unmasking at inference time.

DAG-aware training biases the masking to respect the DAG structure:
tokens in later topological levels are more likely to be masked.
This aligns the training distribution with inference-time behavior.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.models.base import DiffusionLM
from dllm_reason.training.pretrain import Trainer, TrainConfig
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


class DAGAwareTrainer(Trainer):
    """Training loop that uses DAG-biased masking.

    Instead of uniform random masking, positions in later DAG levels
    have higher masking probability. This means the model learns to
    predict later-level tokens conditioned on earlier-level tokens
    being present, matching the DAG-guided inference pattern.
    """

    def __init__(
        self,
        model: DiffusionLM,
        train_loader: DataLoader,
        dag: TokenDAG,
        dag_bias_strength: float = 0.5,
        val_loader: DataLoader | None = None,
        config: TrainConfig | None = None,
    ):
        super().__init__(model, train_loader, val_loader, config)
        self.dag = dag
        self.dag_bias_strength = dag_bias_strength

        # Pre-compute level-based masking bias
        levels = dag.topological_levels()
        self._level_bias = torch.zeros(dag.seq_len)
        for level_idx, positions in enumerate(levels):
            # Higher levels get higher masking probability
            bias = level_idx / max(len(levels) - 1, 1)
            for pos in positions:
                self._level_bias[pos] = bias

    def _dag_biased_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply noise with DAG-level bias.

        Positions at higher topological levels are masked with higher
        probability, while positions at lower levels are masked less.

        The bias is controlled by dag_bias_strength:
        - 0.0: uniform masking (standard training)
        - 1.0: fully DAG-ordered masking
        """
        B, L = x_0.shape
        device = x_0.device

        level_bias = self._level_bias[:L].to(device)

        # Base masking probability from the noise schedule
        base_prob = t[:, None].expand(B, L)

        # Modulate by level bias
        # Higher level -> higher mask prob at same t
        alpha = self.dag_bias_strength
        biased_prob = (1 - alpha) * base_prob + alpha * (base_prob * (0.5 + level_bias))
        biased_prob = biased_prob.clamp(0, 1)

        # Sample mask
        mask = torch.rand(B, L, device=device) < biased_prob
        x_t = torch.where(mask, self.model.mask_token_id, x_0)

        return x_t

    def train(self):
        """Override training to use DAG-biased masking.

        Monkey-patches the model's noise_input to use DAG-biased noise,
        then runs the standard training loop.
        """
        original_noise = self.model.noise_input

        def dag_noise(x_0, t):
            return self._dag_biased_noise(x_0, t)

        self.model.noise_input = dag_noise
        try:
            super().train()
        finally:
            self.model.noise_input = original_noise
