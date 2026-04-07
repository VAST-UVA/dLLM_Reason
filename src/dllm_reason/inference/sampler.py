"""Diffusion sampling loop — LLaDA-style block-wise denoising.

Core algorithm (from GSAI-ML/LLaDA):
  1. Divide generation area into blocks processed left-to-right.
  2. Within each block, run `steps_per_block` denoising sub-steps.
  3. Each sub-step: forward → Gumbel noise + argmax → commit top-k
     confidence positions inside the block.
  4. Optional CFG for instruct models.

The UnmaskingScheduler abstraction is preserved so DAG-guided or other
custom schedulers can override which positions are selected at each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dllm_reason.models.base import DiffusionLM
from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for temperature-controlled discrete sampling.

    temperature=0  → pure argmax (greedy)
    temperature>0  → stochastic; higher = more diverse
    """
    if temperature == 0:
        return logits
    noise = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)
    return logits + temperature * noise


def _spread_tokens(num_masked: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute `num_masked` tokens as evenly as possible across `steps`.

    Returns: (B, steps) int tensor — how many to commit at each step.
    """
    B = num_masked.shape[0]
    base = num_masked // steps          # floor per step
    extra = num_masked % steps          # leftover tokens
    schedule = base.unsqueeze(1).expand(B, steps).clone()
    # Distribute remainder to the first `extra` steps
    for b in range(B):
        schedule[b, :extra[b]] += 1
    return schedule                     # (B, steps)


# ── Config & result ───────────────────────────────────────────────────────────

@dataclass
class SamplingConfig:
    """Configuration for the sampling process."""
    num_steps:    int   = 128     # total denoising steps (spread across blocks)
    block_length: int   = 32      # tokens per block; gen_length must be divisible
    temperature:  float = 0.0     # Gumbel noise scale; 0 = greedy argmax
    cfg_scale:    float = 0.0     # classifier-free guidance scale; 0 = disabled
    remasking:    str   = "low_confidence"  # "low_confidence" | "random"
    show_progress: bool = True
    record_trajectory: bool = False
    debug: bool = False


@dataclass
class SamplingResult:
    """Result of a sampling run."""
    sequences:   torch.Tensor                    # (batch, seq_len)
    trajectory:  list[torch.Tensor] = field(default_factory=list)


# ── Sampler ───────────────────────────────────────────────────────────────────

class DiffusionSampler:
    """Block-wise diffusion sampler for LLaDA-style discrete diffusion models.

    Coordinates:
      - model  : token prediction (no timestep input)
      - scheduler : position selection within each block
    """

    def __init__(
        self,
        model: DiffusionLM,
        scheduler: UnmaskingScheduler,
        config: SamplingConfig | None = None,
    ):
        self.model = model
        self.scheduler = scheduler
        self.config = config or SamplingConfig()

    @torch.no_grad()
    def sample(
        self,
        prompt_ids:   torch.Tensor,
        prompt_mask:  torch.Tensor,
        gen_length:   int,
        device: torch.device | str | None = None,
    ) -> SamplingResult:
        """Run block-wise denoising.

        Args:
            prompt_ids:  (1, prompt_len + gen_length) — prompt tokens followed
                         by mask tokens in the generation area.
            prompt_mask: (1, prompt_len + gen_length) bool — True for prompt.
            gen_length:  number of tokens to generate.
            device:      target device.
        """
        cfg = self.config
        device = device or next(self.model.parameters()).device

        assert gen_length % cfg.block_length == 0, (
            f"gen_length ({gen_length}) must be divisible by "
            f"block_length ({cfg.block_length})"
        )
        num_blocks = gen_length // cfg.block_length
        assert cfg.num_steps % num_blocks == 0, (
            f"num_steps ({cfg.num_steps}) must be divisible by "
            f"num_blocks ({num_blocks})"
        )
        steps_per_block = cfg.num_steps // num_blocks

        x = prompt_ids.clone().to(device)
        prompt_mask = prompt_mask.to(device)
        mask_id = self.model.mask_token_id
        prompt_len = int(prompt_mask[0].sum().item())

        # True for prompt positions — used for CFG unconditional branch
        prompt_index = prompt_mask.clone()

        self.model.eval()
        self.scheduler.reset()

        trajectory = []
        if cfg.record_trajectory:
            trajectory.append(x.clone())

        blocks = range(num_blocks)
        if cfg.show_progress:
            blocks = tqdm(blocks, desc="Blocks", leave=False)

        for block_idx in blocks:
            b_start = prompt_len + block_idx * cfg.block_length
            b_end   = prompt_len + (block_idx + 1) * cfg.block_length

            # Boolean mask: positions belonging to this block
            block_mask = torch.zeros_like(x, dtype=torch.bool)
            block_mask[:, b_start:b_end] = True

            # How many tokens to commit at each sub-step (evenly spread)
            block_masked_now = (x[:, b_start:b_end] == mask_id)
            num_schedule = _spread_tokens(
                block_masked_now.sum(dim=-1), steps_per_block
            )  # (B, steps_per_block)

            for step in range(steps_per_block):
                # ── Forward pass ──────────────────────────────────────────
                if cfg.cfg_scale > 0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    logits_all = self.model.forward(
                        torch.cat([x, un_x], dim=0)
                    ).logits
                    logits, un_logits = logits_all.chunk(2, dim=0)
                    logits = un_logits + (cfg.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model.forward(x).logits  # (1, L, V)

                # ── Sample candidate x0 for every position ────────────────
                logits_noisy = _add_gumbel_noise(logits, cfg.temperature)
                x0 = logits_noisy.argmax(dim=-1)           # (1, L)

                # ── Confidence (used by scheduler for position selection) ──
                if cfg.remasking == "low_confidence":
                    p = F.softmax(logits.double(), dim=-1)
                    confidences = torch.gather(
                        p, dim=-1, index=x0.unsqueeze(-1)
                    ).squeeze(-1).float()                  # (1, L)
                else:  # random
                    confidences = torch.rand(x.shape, device=device)

                # ── Which positions are still masked overall ───────────────
                current_mask = (x == mask_id) & ~prompt_mask

                # ── Scheduler picks positions to commit ───────────────────
                # Passes block_mask so schedulers can restrict to this block;
                # DAG or other schedulers may use different logic.
                n_this_step = int(num_schedule[0, step].item())
                positions_to_unmask = self.scheduler.select_positions(
                    step=step,
                    total_steps=steps_per_block,
                    current_mask=current_mask,
                    is_unmasked=~current_mask,
                    logits=logits,
                    confidences=confidences,
                    block_mask=block_mask,
                    n_to_select=n_this_step,
                )
                positions_to_unmask = positions_to_unmask & ~prompt_mask

                # Keep prompt / already-committed tokens as-is; commit new ones
                x0_safe = torch.where(current_mask, x0, x)
                x = torch.where(positions_to_unmask, x0_safe, x)

                if cfg.record_trajectory:
                    trajectory.append(x.clone())

            if cfg.debug:
                remaining = int((x[:, b_start:b_end] == mask_id).sum().item())
                logger.debug(
                    f"block {block_idx+1}/{num_blocks} "
                    f"filled {cfg.block_length - remaining}/{cfg.block_length}"
                )

        # Force-fill any remaining mask tokens (shouldn't happen with correct
        # step counts, but guards against off-by-one in custom schedulers).
        remaining_mask = (x == mask_id) & ~prompt_mask
        if remaining_mask.any():
            logits = self.model.forward(x).logits
            logits[..., mask_id] = -float("inf")
            x = torch.where(remaining_mask, logits.argmax(dim=-1), x)

        return SamplingResult(sequences=x, trajectory=trajectory)
