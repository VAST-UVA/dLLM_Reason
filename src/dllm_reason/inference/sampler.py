"""Generic diffusion sampling loop.

Provides a clean, configurable sampling interface that works with
any DiffusionLM model and any UnmaskingScheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from dllm_reason.models.base import DiffusionLM
from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for the sampling process."""
    num_steps: int = 64
    temperature: float = 1.0
    top_k: int = 0           # 0 = no top-k filtering
    top_p: float = 1.0       # 1.0 = no nucleus filtering
    show_progress: bool = True
    record_trajectory: bool = False
    debug: bool = False      # print per-step unmasking stats to DEBUG log


@dataclass
class SamplingResult:
    """Result of a sampling run."""
    sequences: torch.Tensor        # (batch, seq_len) final token ids
    trajectory: list[torch.Tensor] = field(default_factory=list)  # list of (batch, seq_len) at each step


class DiffusionSampler:
    """Generic sampling engine for discrete diffusion models.

    Coordinates the model (token prediction) and scheduler (position selection)
    to generate sequences through iterative unmasking.
    """

    def __init__(self, model: DiffusionLM, scheduler: UnmaskingScheduler, config: SamplingConfig | None = None):
        self.model = model
        self.scheduler = scheduler
        self.config = config or SamplingConfig()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> SamplingResult:
        """Run the sampling loop.

        Args:
            batch_size: number of sequences to generate
            seq_len: sequence length (defaults to model's max_seq_len)
            prompt_ids: (batch, seq_len) prompt token ids (non-prompt positions should be mask_token_id)
            prompt_mask: (batch, seq_len) bool, True for prompt positions (not to be modified)
            device: generation device

        Returns:
            SamplingResult with final sequences and optional trajectory
        """
        seq_len = seq_len or self.model.max_seq_len
        device = device or self.model.device
        cfg = self.config

        self.model.eval()
        self.scheduler.reset()

        # Initialize fully masked
        x_t = torch.full((batch_size, seq_len), self.model.mask_token_id, dtype=torch.long, device=device)

        # Fill prompt
        if prompt_ids is not None:
            x_t = prompt_ids.clone().to(device)
        if prompt_mask is None:
            prompt_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            prompt_mask = prompt_mask.to(device)

        is_unmasked = (x_t != self.model.mask_token_id) | prompt_mask

        trajectory = []
        if cfg.record_trajectory:
            trajectory.append(x_t.clone())

        steps = range(cfg.num_steps)
        if cfg.show_progress:
            steps = tqdm(steps, desc="Sampling", leave=False)

        mask_id = self.model.mask_token_id
        _debug = cfg.debug if hasattr(cfg, "debug") else False
        prompt_len = int(prompt_mask[0].sum().item()) if prompt_mask is not None else 0

        for step in steps:
            # Timestep: from ~1 (noisy) to ~0 (clean)
            t_val = 1.0 - step / cfg.num_steps
            t = torch.full((batch_size,), t_val, device=device)

            # Model forward
            output = self.model.forward(x_t, t)
            logits = output.logits  # (B, L, V)

            # ── 1. Suppress mask token in raw logits ─────────────────────────
            # Must happen before any probability computation so that the mask
            # token never dominates confidence scores or gets sampled.
            logits = logits.clone()
            if mask_id < logits.shape[-1]:
                logits[..., mask_id] = -float("inf")

            # ── 2. Confidence from RAW logits (no temperature distortion) ────
            # The scheduler selects positions by confidence rank; applying
            # temperature here would distort those ranks without benefit.
            raw_probs = torch.softmax(logits, dim=-1)
            confidences = raw_probs.max(dim=-1).values  # (B, L)

            # ── Step-0 diagnostics ────────────────────────────────────────────
            if step == 0:
                gen_slice = x_t[0, prompt_len:]
                current_mask_check = (x_t == mask_id) & ~prompt_mask
                logger.info(
                    f"[DIAG step=0] mask_token_id={mask_id} | "
                    f"logits.shape={list(logits.shape)} | "
                    f"x_t gen unique={gen_slice.unique().tolist()} | "
                    f"current_mask sum={current_mask_check.sum().item()}/{seq_len - prompt_len} | "
                    f"logits[0,prompt_len,mask_id]="
                    f"{output.logits[0, prompt_len, mask_id].item() if mask_id < output.logits.shape[-1] else 'OOB'} "
                    f"(before suppression) | "
                    f"logits[0,prompt_len].argmax={output.logits[0, prompt_len].argmax().item()} | "
                    f"logits[0,prompt_len] top5 tokens={output.logits[0, prompt_len].topk(5).indices.tolist()}"
                )
            # ─────────────────────────────────────────────────────────────────

            # ── 3. Scheduler: which positions to unmask ───────────────────────
            current_mask = (x_t == mask_id) & ~prompt_mask
            positions_to_unmask = self.scheduler.select_positions(
                step=step,
                total_steps=cfg.num_steps,
                current_mask=current_mask,
                is_unmasked=is_unmasked,
                logits=logits,       # raw (mask-suppressed, no temperature)
                confidences=confidences,
            )
            positions_to_unmask = positions_to_unmask & ~prompt_mask
            n_to_unmask = int(positions_to_unmask.sum().item())

            # ── Step-0 post-scheduler diagnostic ─────────────────────────────
            if step == 0:
                logger.info(
                    f"[DIAG step=0 post-sched] "
                    f"current_mask.sum={current_mask.sum().item()} | "
                    f"positions_to_unmask.sum={n_to_unmask} | "
                    f"confidences[0,prompt_len:prompt_len+4]={confidences[0, prompt_len:prompt_len+4].tolist()} | "
                    f"raw_probs[0,prompt_len].max={raw_probs[0, prompt_len].max().item():.4f} | "
                    f"raw_probs[0,prompt_len].argmax={raw_probs[0, prompt_len].argmax().item()}"
                )
            # ─────────────────────────────────────────────────────────────────

            # ── 4. Sample ONLY for positions being unmasked ───────────────────
            # Temperature / top-k / top-p are applied here — after the
            # scheduler has made its selection — so they affect the token
            # identity, not the position-selection decision.
            if n_to_unmask > 0:
                if cfg.temperature == 0.0:
                    # Greedy: argmax directly, no softmax needed.
                    sampled = logits.argmax(dim=-1)  # (B, L)
                else:
                    sample_logits = logits.clone()

                    # Temperature scaling
                    if cfg.temperature != 1.0:
                        sample_logits = sample_logits / cfg.temperature

                    # Top-k filtering
                    if cfg.top_k > 0:
                        top_k_val = min(cfg.top_k, sample_logits.shape[-1])
                        topk_vals, _ = sample_logits.topk(top_k_val, dim=-1)
                        threshold = topk_vals[..., -1:]
                        sample_logits = sample_logits.masked_fill(
                            sample_logits < threshold, -float("inf")
                        )

                    # Top-p (nucleus) filtering
                    if cfg.top_p < 1.0:
                        sorted_logits, sorted_indices = sample_logits.sort(
                            dim=-1, descending=True
                        )
                        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                        nucleus_mask = (
                            cum_probs - sorted_logits.softmax(dim=-1) >= cfg.top_p
                        )
                        sorted_logits[nucleus_mask] = -float("inf")
                        sample_logits = sorted_logits.scatter(
                            -1, sorted_indices, sorted_logits
                        )

                    # Convert to probabilities; hard-zero mask token as a
                    # final guard against any numerical residue.
                    sample_probs = torch.softmax(sample_logits, dim=-1)
                    if mask_id < sample_probs.shape[-1]:
                        sample_probs = sample_probs.clone()
                        sample_probs[..., mask_id] = 0.0

                    # Re-normalise; fall back to uniform if all zeros.
                    prob_sum = sample_probs.sum(dim=-1, keepdim=True)
                    zero_rows = (prob_sum == 0).squeeze(-1)
                    if zero_rows.any():
                        sample_probs[zero_rows] = 1.0
                        prob_sum = sample_probs.sum(dim=-1, keepdim=True)
                    sample_probs = sample_probs / prob_sum

                    sampled = torch.multinomial(
                        sample_probs.view(-1, sample_probs.shape[-1]), num_samples=1
                    ).view(batch_size, seq_len)

                x_t = torch.where(positions_to_unmask, sampled, x_t)
                is_unmasked = is_unmasked | positions_to_unmask

            if _debug and step % max(1, cfg.num_steps // 8) == 0:
                n_still_masked = int((x_t[0, prompt_len:] == mask_id).sum().item())
                gen_len = seq_len - prompt_len
                logger.info(
                    f"[DBG] step {step:3d}/{cfg.num_steps} | "
                    f"unmasked_this_step={n_to_unmask:4d} | "
                    f"still_masked={n_still_masked}/{gen_len} | "
                    f"gen_tokens={x_t[0, prompt_len:prompt_len+8].tolist()}"  # first 8 gen tokens
                )

            if cfg.record_trajectory:
                trajectory.append(x_t.clone())

        # Final: force-unmask any remaining masked positions with greedy argmax.
        # Using argmax (not multinomial) and excluding mask_token_id guarantees
        # no mask tokens survive into the decoded output.
        remaining_mask = (x_t == self.model.mask_token_id) & ~prompt_mask
        if remaining_mask.any():
            t = torch.full((batch_size,), 0.0, device=device)
            output = self.model.forward(x_t, t)
            final_logits = output.logits.clone()
            final_logits[..., self.model.mask_token_id] = -float("inf")
            sampled = final_logits.argmax(dim=-1)
            x_t = torch.where(remaining_mask, sampled, x_t)

        return SamplingResult(sequences=x_t, trajectory=trajectory)
