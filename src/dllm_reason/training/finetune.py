"""Fine-tuning loop for dLLMs on reasoning tasks.

Fine-tunes a pretrained dLLM (e.g., LLaDA) on reasoning datasets
like GSM8K using the standard masked diffusion objective, but restricted
to the generation (answer) portion of the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dllm_reason.models.base import DiffusionLM
from dllm_reason.training.pretrain import Trainer, TrainConfig
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FinetuneConfig(TrainConfig):
    """Config for fine-tuning. Inherits from TrainConfig with adjusted defaults."""
    lr: float = 2e-5            # Lower LR for fine-tuning
    max_steps: int = 10000      # Fewer steps
    warmup_steps: int = 200
    loss_on_answer_only: bool = True  # Only compute loss on generation positions
    prompt_loss_weight: float = 0.0   # Weight for prompt loss (0 = ignore prompt)


class Finetuner(Trainer):
    """Fine-tune a dLLM on a reasoning dataset.

    Key difference from pretraining: when loss_on_answer_only=True,
    the loss is computed only at positions where the model generates the answer
    (i.e., positions NOT in the prompt). This focuses the training signal
    on the answer portion.
    """

    def __init__(
        self,
        model: DiffusionLM,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: FinetuneConfig | None = None,
    ):
        cfg = config or FinetuneConfig()
        super().__init__(model, train_loader, val_loader, cfg)
        self.ft_config = cfg

        # Use cosine with warm restarts for fine-tuning
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )

    def _compute_finetune_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: torch.Tensor | None,
        prompt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Loss restricted to answer positions.

        If prompt_mask is provided and loss_on_answer_only is True,
        only compute loss at non-prompt positions.
        """
        import torch.nn.functional as F

        B, L = x_0.shape
        device = x_0.device

        t = torch.rand(B, device=device).clamp(1e-5, 1.0 - 1e-5)
        x_t = self.model.noise_input(x_0, t)

        from dllm_reason.models.base import DiffusionOutput
        output = self.model.forward(x_t, t, attention_mask)
        logits = output.logits

        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)

        # Mask: positions that are masked AND not prompt (if answer-only mode)
        is_masked = (x_t == self.model.mask_token_id)

        if self.ft_config.loss_on_answer_only and prompt_mask is not None:
            answer_positions = ~prompt_mask.bool()
            is_masked = is_masked & answer_positions

        if attention_mask is not None:
            is_masked = is_masked & attention_mask.bool()

        masked_nll = (nll * is_masked.float()).sum(-1)
        num_masked = is_masked.float().sum(-1).clamp(min=1.0)
        return (masked_nll / num_masked).mean()

    def train(self):
        """Fine-tuning loop using answer-only loss."""
        cfg = self.ft_config
        self.model.train()
        device = self.model.device
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        accum_loss = 0.0
        data_iter = iter(self.train_loader)

        while self.global_step < cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            x_0 = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            prompt_mask = batch.get("prompt_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if prompt_mask is not None:
                prompt_mask = prompt_mask.to(device)

            loss = self._compute_finetune_loss(x_0, attention_mask, prompt_mask)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (self.global_step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            if self.global_step % cfg.log_every == 0:
                avg_loss = accum_loss / cfg.log_every
                logger.info(f"Finetune step {self.global_step}: loss={avg_loss:.4f}")
                accum_loss = 0.0

            if self.val_loader and self.global_step % cfg.eval_every == 0:
                val_loss = self.evaluate()
                logger.info(f"Finetune step {self.global_step}: val_loss={val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(save_dir / "best.pt")

            if self.global_step % cfg.save_every == 0:
                self.save_checkpoint(save_dir / f"step_{self.global_step}.pt")
