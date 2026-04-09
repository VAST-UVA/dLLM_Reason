"""RL-based training for dLLMs.

Two RL algorithms are implemented here:

DiffuGRPO
---------
Group Relative Policy Optimization adapted for discrete diffusion LMs.
For each prompt, K completions are sampled; advantages are computed
relative to the group mean reward.

Reference: d1 - Scaling Reasoning in Diffusion LLMs
(https://github.com/dllm-reasoning/d1)

DiFFPO  (Diffusion Fast and Furious Policy Optimization)
---------------------------------------------------------
PPO-style RL for dLLMs with two key innovations over GRPO:

1. Surrogate-policy off-policy correction
   A learned surrogate policy (log-likelihood approximation) allows
   importance-ratio clipping (PPO clip) for more stable updates.
   The importance ratio ρ = π_θ(y|x) / π_ref(y|x) is clipped to
   [1-ε, 1+ε] to bound the policy update.

2. Joint sampler–model training (adaptive NFE per prompt)
   A lightweight step-budget controller θ_s is trained jointly with the
   language model θ.  Given a prompt, θ_s predicts a stopping threshold
   T* ∈ {T_min, …, T_max}; the diffusion chain is truncated at T*.
   This lets the model allocate more denoising steps to harder prompts
   and fewer to easier ones, improving the inference-time compute Pareto
   frontier without changing the model architecture.

   The controller is trained with a composite loss:
     L_total = L_PPO(θ) + λ · L_step(θ_s)
   where L_step penalises unnecessary denoising steps while rewarding
   correct outputs.

Reference: DiFFPO — Training Diffusion LLMs to Reason Fast and Furious
via Reinforcement Learning.  Zhao et al., 2024.
https://arxiv.org/abs/2510.02212
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
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    gen_length=prompt_ids.shape[1] - int(prompt_mask[0].sum().item()),
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


# ── DiFFPO ────────────────────────────────────────────────────────────────────

@dataclass
class DiFFPOConfig:
    """Configuration for DiFFPO training.

    Reference: https://arxiv.org/abs/2510.02212
    """
    # Policy optimisation
    lr: float = 1e-5
    num_iterations: int = 1000
    group_size: int = 8            # K completions per prompt
    ppo_clip_eps: float = 0.2      # ε for PPO importance-ratio clipping
    kl_coeff: float = 0.01         # KL penalty weight (ref model)
    max_grad_norm: float = 1.0
    log_every: int = 10

    # Sampling
    temperature: float = 0.8

    # Joint sampler / adaptive NFE
    train_sampler: bool = True     # jointly train the step-budget controller
    step_lr: float = 1e-4          # separate LR for controller parameters
    step_budget_lambda: float = 0.1  # weight of L_step in total loss
    min_steps: int = 8             # minimum denoising steps
    max_steps: int = 128           # maximum denoising steps
    step_candidates: int = 8       # number of step budgets to try per prompt


class StepBudgetController(torch.nn.Module):
    """Lightweight MLP that predicts a step-budget scalar from prompt embeddings.

    Given a prompt embedding (pooled last-hidden-state), outputs a
    scalar logit over ``step_candidates`` discrete budgets.

    Architecture: Linear(hidden_dim, 128) → ReLU → Linear(128, 1)
    """

    def __init__(self, hidden_dim: int, num_budgets: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_budgets),
        )

    def forward(self, prompt_emb: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities over step-budget indices.

        Args:
            prompt_emb: (B, hidden_dim) pooled prompt representations.

        Returns:
            (B, num_budgets) log-softmax scores.
        """
        return F.log_softmax(self.net(prompt_emb), dim=-1)


class DiFFPO:
    """Diffusion Fast and Furious Policy Optimization for dLLMs.

    Implements two innovations from Zhao et al. 2024
    (https://arxiv.org/abs/2510.02212):

    1. **Surrogate-policy PPO** — importance-ratio clipping replaces the
       plain policy-gradient of DiffuGRPO, giving more stable updates.

    2. **Joint sampler training** — a ``StepBudgetController`` predicts the
       optimal number of denoising steps for each prompt.  Its loss is:
           L_step = -reward · log π_s(T* | x) + λ_step · T* / T_max
       encouraging the controller to use fewer steps on easy prompts.

    Training loss per iteration:
        L_total = L_PPO + kl_coeff · KL(π_θ || π_ref) + step_lambda · L_step
    """

    def __init__(
        self,
        model: DiffusionLM,
        ref_model: DiffusionLM,
        scheduler: UnmaskingScheduler,
        reward_fn: Callable[[torch.Tensor, dict], float],
        train_loader: DataLoader,
        config: DiFFPOConfig | None = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.reward_fn = reward_fn
        self.train_loader = train_loader
        self.config = config or DiFFPOConfig()

        # Step-budget candidates as a 1-D tensor of step counts
        cfg = self.config
        step_range = torch.linspace(cfg.min_steps, cfg.max_steps,
                                    cfg.step_candidates).long()
        self.register_buffer_or_store("_step_candidates", step_range)

        # Main model optimiser
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Controller (built lazily when we know hidden_dim)
        self._controller: StepBudgetController | None = None
        self._ctrl_optimizer = None

    # ── Public helpers ────────────────────────────────────────────────────

    def register_buffer_or_store(self, name: str, tensor: torch.Tensor) -> None:
        """Store a non-parameter tensor (used for step_candidates)."""
        setattr(self, name, tensor)

    def _get_or_build_controller(self, hidden_dim: int) -> StepBudgetController:
        if self._controller is None:
            device = self.model.device
            self._controller = StepBudgetController(
                hidden_dim, self.config.step_candidates
            ).to(device)
            self._ctrl_optimizer = torch.optim.Adam(
                self._controller.parameters(), lr=self.config.step_lr
            )
            logger.info(
                f"Built StepBudgetController "
                f"(hidden={hidden_dim}, budgets={self.config.step_candidates})"
            )
        return self._controller

    def _embed_prompt(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Pool last-hidden-state over prompt positions → (B, hidden_dim)."""
        # Run model at t≈0 in eval mode to get hidden states
        B, L = prompt_ids.shape
        t = torch.full((B,), 0.0, device=prompt_ids.device)
        with torch.no_grad():
            out = self.model.forward(prompt_ids, t)
        # out.logits is (B, L, V); use mean over L as a proxy embedding
        # (proper implementation would use the encoder hidden states)
        emb = out.logits.mean(dim=1)   # (B, V) — used as embedding proxy
        return emb

    # ── Core training loop ────────────────────────────────────────────────

    def train(self) -> None:
        cfg = self.config
        device = self.model.device
        self.model.train()
        data_iter = iter(self.train_loader)

        for iteration in range(cfg.num_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            prompt_ids  = batch["input_ids"].to(device)
            prompt_mask = batch.get(
                "prompt_mask",
                torch.zeros_like(prompt_ids, dtype=torch.bool),
            ).to(device)
            B = prompt_ids.shape[0]
            prompt_len = int(prompt_mask[0].sum().item())
            gen_length = prompt_ids.shape[1] - prompt_len

            # ── Adaptive step budget (joint sampler) ──────────────────
            if cfg.train_sampler:
                prompt_emb = self._embed_prompt(prompt_ids)
                controller = self._get_or_build_controller(prompt_emb.shape[-1])
                ctrl_log_probs = controller(prompt_emb.detach())  # (B, K)
                # Pick the step budget with highest probability for inference
                budget_indices = ctrl_log_probs.argmax(dim=-1)    # (B,)
                step_candidates = self._step_candidates.to(device)
                num_steps_batch = step_candidates[budget_indices]  # (B,)
                # Use per-sample step budget (take sample 0 for simplicity
                # when B>1 and all prompts use the same budget this step)
                num_steps = int(num_steps_batch[0].item())
            else:
                ctrl_log_probs = None
                num_steps = cfg.max_steps

            # ── Sample K completions ──────────────────────────────────
            all_rewards:       list[torch.Tensor] = []
            all_policy_lp:     list[torch.Tensor] = []
            all_ref_lp:        list[torch.Tensor] = []

            for _ in range(cfg.group_size):
                sampler_cfg = SamplingConfig(
                    num_steps=num_steps,
                    temperature=cfg.temperature,
                    show_progress=False,
                )
                sampler = DiffusionSampler(self.model, self.scheduler, sampler_cfg)
                result = sampler.sample(
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    gen_length=gen_length,
                )
                seqs = result.sequences  # (B, L)

                lp   = self._seq_log_prob(self.model,     seqs, prompt_mask)
                rlp  = self._seq_log_prob(self.ref_model, seqs, prompt_mask)

                rewards = torch.tensor(
                    [self.reward_fn(seqs[i], batch) for i in range(B)],
                    device=device, dtype=torch.float32,
                )
                all_rewards.append(rewards)
                all_policy_lp.append(lp)
                all_ref_lp.append(rlp)

            rewards_t   = torch.stack(all_rewards)    # (K, B)
            policy_lp_t = torch.stack(all_policy_lp)  # (K, B)
            ref_lp_t    = torch.stack(all_ref_lp)     # (K, B)

            # ── Group-relative advantage ──────────────────────────────
            mean_r = rewards_t.mean(0, keepdim=True)
            std_r  = rewards_t.std(0,  keepdim=True).clamp(min=1e-8)
            adv    = (rewards_t - mean_r) / std_r      # (K, B)

            # ── PPO importance-ratio clipping ─────────────────────────
            log_ratio = policy_lp_t - ref_lp_t         # (K, B)
            ratio     = log_ratio.exp()
            surr1     = ratio * adv
            surr2     = ratio.clamp(1 - cfg.ppo_clip_eps,
                                    1 + cfg.ppo_clip_eps) * adv
            ppo_loss  = -torch.min(surr1, surr2).mean()

            # ── KL penalty ────────────────────────────────────────────
            kl_loss = cfg.kl_coeff * log_ratio.mean()

            total_loss = ppo_loss + kl_loss

            # ── Step controller loss (joint sampler training) ─────────
            ctrl_loss = torch.tensor(0.0, device=device)
            if cfg.train_sampler and ctrl_log_probs is not None:
                # L_step = -mean_reward · log π_s(T*|x) + λ · T*/T_max
                mean_reward_this = rewards_t.mean(0)  # (B,)
                selected_lp = ctrl_log_probs[
                    torch.arange(B, device=device), budget_indices
                ]  # (B,)
                step_frac = num_steps_batch.float() / cfg.max_steps
                ctrl_loss = (
                    -(mean_reward_this * selected_lp).mean()
                    + cfg.step_budget_lambda * step_frac.mean()
                )
                total_loss = total_loss + cfg.step_budget_lambda * ctrl_loss

            # ── Optimiser step ────────────────────────────────────────
            self.optimizer.zero_grad()
            if self._ctrl_optimizer is not None:
                self._ctrl_optimizer.zero_grad()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm
            )
            self.optimizer.step()
            if self._ctrl_optimizer is not None:
                self._ctrl_optimizer.step()

            if (iteration + 1) % cfg.log_every == 0:
                mean_r_val = rewards_t.mean().item()
                logger.info(
                    f"Iter {iteration+1}/{cfg.num_iterations}  "
                    f"reward={mean_r_val:.4f}  "
                    f"ppo_loss={ppo_loss.item():.4f}  "
                    f"kl={kl_loss.item():.4f}  "
                    f"ctrl_loss={ctrl_loss.item():.4f}  "
                    f"num_steps={num_steps}"
                )

    # ── Shared helper (same as DiffuGRPO) ────────────────────────────────

    def _seq_log_prob(
        self,
        model: DiffusionLM,
        sequences: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate sequence log-probability at t≈0."""
        B, L = sequences.shape
        t = torch.full((B,), 0.01, device=sequences.device)
        x_t = model.noise_input(sequences, t)
        output = model.forward(x_t, t)
        log_probs = F.log_softmax(output.logits, dim=-1)
        token_lp = log_probs.gather(-1, sequences.unsqueeze(-1)).squeeze(-1)
        gen_mask = ~prompt_mask
        return (token_lp * gen_mask.float()).sum(dim=-1)
