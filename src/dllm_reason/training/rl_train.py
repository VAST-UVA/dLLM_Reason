"""RL-based training for dLLMs (diffu-GRPO style).

Applies reinforcement learning to fine-tune dLLMs for reasoning tasks.
Uses Group Relative Policy Optimization adapted for diffusion models.

Reference: d1 - Scaling Reasoning in Diffusion LLMs
(https://github.com/dllm-reasoning/d1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dllm_reason.models.base import DiffusionLM
from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RLTrainConfig:
    lr: float = 1e-5
    num_iterations: int = 1000
    group_size: int = 8           # Number of samples per prompt for GRPO
    kl_coeff: float = 0.01       # KL penalty coefficient
    clip_ratio: float = 0.2      # PPO-style clipping
    max_grad_norm: float = 1.0
    num_steps: int = 32           # Diffusion sampling steps
    temperature: float = 0.8
    log_every: int = 10


class DiffuGRPO:
    """Diffusion Group Relative Policy Optimization.

    Adapts GRPO for discrete diffusion models:
    1. For each prompt, generate a group of K samples
    2. Score each sample with a reward function
    3. Compute advantages relative to the group mean
    4. Update model to increase probability of high-reward samples
    """

    def __init__(
        self,
        model: DiffusionLM,
        ref_model: DiffusionLM,
        scheduler: UnmaskingScheduler,
        reward_fn: Callable[[torch.Tensor, dict], float],
        train_loader: DataLoader,
        config: RLTrainConfig | None = None,
    ):
        self.model = model
        self.ref_model = ref_model  # Frozen reference for KL
        self.scheduler = scheduler
        self.reward_fn = reward_fn
        self.train_loader = train_loader
        self.config = config or RLTrainConfig()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def train(self):
        cfg = self.config
        device = self.model.device
        data_iter = iter(self.train_loader)

        for iteration in range(cfg.num_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            prompt_ids = batch["input_ids"].to(device)
            prompt_mask = batch.get("prompt_mask", torch.zeros_like(prompt_ids, dtype=torch.bool)).to(device)
            B = prompt_ids.shape[0]

            all_rewards = []
            all_log_probs = []
            all_ref_log_probs = []

            # Generate group of samples for each prompt
            for _ in range(cfg.group_size):
                # Sample from current model
                sampler = DiffusionSampler(
                    self.model, self.scheduler,
                    SamplingConfig(num_steps=cfg.num_steps, temperature=cfg.temperature, show_progress=False),
                )
                result = sampler.sample(
                    batch_size=B,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                )

                # Compute log probabilities under current and reference model
                sequences = result.sequences
                log_prob = self._compute_sequence_log_prob(self.model, sequences, prompt_mask)
                ref_log_prob = self._compute_sequence_log_prob(self.ref_model, sequences, prompt_mask)

                # Compute reward
                rewards = []
                for i in range(B):
                    r = self.reward_fn(sequences[i], batch)
                    rewards.append(r)
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)

                all_rewards.append(rewards)
                all_log_probs.append(log_prob)
                all_ref_log_probs.append(ref_log_prob)

            # Stack: (group_size, B)
            all_rewards = torch.stack(all_rewards)
            all_log_probs = torch.stack(all_log_probs)
            all_ref_log_probs = torch.stack(all_ref_log_probs)

            # Group-relative advantages
            mean_reward = all_rewards.mean(dim=0, keepdim=True)
            std_reward = all_rewards.std(dim=0, keepdim=True).clamp(min=1e-8)
            advantages = (all_rewards - mean_reward) / std_reward

            # Policy gradient with KL penalty
            kl = all_log_probs - all_ref_log_probs  # (group_size, B)
            policy_loss = -(advantages * all_log_probs).mean()
            kl_loss = cfg.kl_coeff * kl.mean()
            loss = policy_loss + kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            if (iteration + 1) % cfg.log_every == 0:
                mean_r = all_rewards.mean().item()
                logger.info(
                    f"Iter {iteration+1}: reward={mean_r:.4f}, "
                    f"policy_loss={policy_loss.item():.4f}, kl={kl.mean().item():.4f}"
                )

    def _compute_sequence_log_prob(
        self,
        model: DiffusionLM,
        sequences: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute approximate log probability of sequences under the model.

        For dLLMs, exact log probability is intractable. We approximate
        by computing the model's prediction loss on the sequences at t≈0.
        """
        B, L = sequences.shape
        t = torch.full((B,), 0.01, device=sequences.device)

        # Noise a small fraction and compute prediction likelihood
        x_t = model.noise_input(sequences, t)
        output = model.forward(x_t, t)
        log_probs = F.log_softmax(output.logits, dim=-1)

        # Gather log prob of actual tokens
        token_log_probs = log_probs.gather(-1, sequences.unsqueeze(-1)).squeeze(-1)

        # Only count non-prompt positions
        gen_mask = ~prompt_mask
        seq_log_prob = (token_log_probs * gen_mask.float()).sum(dim=-1)

        return seq_log_prob
