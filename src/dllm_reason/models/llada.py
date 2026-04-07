"""LLaDA: Large Language Diffusion with mAsking.

Wrapper around the pretrained LLaDA-8B-Instruct model from HuggingFace.
LLaDA is based on LLaMA-3 architecture with bidirectional attention and
absorbing-state masked diffusion.

This wrapper adapts LLaDA for use with our scheduler/DAG infrastructure.
The key adaptation: LLaDA's generation loop is replaced by our generic
sampler, which calls back into the LLaDA forward pass for predictions.

Reference: https://github.com/ML-GSAI/LLaDA
Model: GSAI-ML/LLaDA-8B-Instruct (or LLaDA-8B-Base)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

from dllm_reason.models.base import DiffusionLM, DiffusionOutput
from dllm_reason.utils.registry import MODEL_REGISTRY
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)

LLADA_HF_ID = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_BASE_HF_ID = "GSAI-ML/LLaDA-8B-Base"


@MODEL_REGISTRY.register("llada")
class LLaDAWrapper(DiffusionLM):
    """Wrapper around pretrained LLaDA-8B for DAG-guided inference.

    This class wraps the HuggingFace LLaDA model and adapts its interface
    to the DiffusionLM abstract class. The forward pass delegates to LLaDA's
    underlying model; our schedulers handle the unmasking order.
    """

    def __init__(
        self,
        model_id: str = LLADA_HF_ID,
        max_seq_len: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        # Load tokenizer first to get vocab info
        logger.info(f"Loading tokenizer from {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        # LLaDA uses [MASK] as the absorbing-state token.
        # tokenizer.mask_token_id is unreliable — it may point to an unrelated
        # special token (e.g. <|startoftext|>).  Look up by token string first.
        mask_token_id = None
        for _candidate in ("[MASK]", "<mask>", "[mask]"):
            _id = tokenizer.convert_tokens_to_ids(_candidate)
            # convert_tokens_to_ids returns unk_token_id when not found
            if _id is not None and _id != tokenizer.unk_token_id:
                mask_token_id = _id
                logger.info(f"Found mask token '{_candidate}' at id {_id}")
                break
        if mask_token_id is None:
            # Fall back to tokenizer.mask_token_id, then vocab boundary
            if tokenizer.mask_token_id is not None:
                mask_token_id = tokenizer.mask_token_id
                logger.warning(
                    f"[MASK] token not found by name; using tokenizer.mask_token_id="
                    f"{mask_token_id} ({tokenizer.decode([mask_token_id])})"
                )
            else:
                mask_token_id = len(tokenizer) - 1
                logger.warning(
                    f"[MASK] token not found; falling back to last vocab index {mask_token_id}"
                )

        vocab_size = len(tokenizer)

        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            mask_token_id=mask_token_id,
        )

        self.tokenizer = tokenizer
        self.model_id = model_id

        # Load the pretrained model
        logger.info(f"Loading LLaDA model from {model_id} ...")
        self._llada = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        logger.info("LLaDA model loaded.")

        # Sync vocab_size with model's actual output dimension
        # (may differ from len(tokenizer) due to padding for GPU alignment)
        if hasattr(self._llada, "config") and hasattr(self._llada.config, "vocab_size"):
            self.vocab_size = self._llada.config.vocab_size

        # Freeze by default — we only use it for inference
        for p in self._llada.parameters():
            p.requires_grad = False

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | None = None,    # unused; kept for interface compat
        attention_mask: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        """Forward pass through LLaDA.

        LLaDA takes only input_ids — no timestep embedding.
        The noise level is implicitly encoded by how many positions are masked.
        """
        # LLaDA typically takes input_ids + attention_mask
        model_inputs = {"input_ids": x_t}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        outputs = self._llada(**model_inputs)

        # LLaDA outputs logits directly
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        return DiffusionOutput(logits=logits, loss=None, confidences=None)

    def noise_input(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply absorbing-state noise: replace tokens with [MASK]."""
        sigma = t[:, None].expand_as(x_0)
        mask = torch.rand_like(sigma.float()) < sigma
        return torch.where(mask, self.mask_token_id, x_0)

    def compute_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MDLM-style loss (for fine-tuning if needed)."""
        B, L = x_0.shape
        device = x_0.device

        t = torch.rand(B, device=device).clamp(1e-5, 1.0 - 1e-5)
        x_t = self.noise_input(x_0, t)

        output = self.forward(x_t, t, attention_mask)
        logits = output.logits

        is_masked = (x_t == self.mask_token_id)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            is_masked = is_masked & attention_mask.bool()

        masked_nll = (nll * is_masked.float()).sum(-1)
        num_masked = is_masked.float().sum(-1).clamp(min=1.0)
        return (masked_nll / num_masked).mean()

    def encode_prompt(
        self,
        prompt: str,
        generation_len: int = 512,
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a prompt + blank generation space.

        Returns:
            input_ids: (1, total_len) with MASK tokens in generation positions
            prompt_mask: (1, total_len) True for prompt positions
        """
        # Always apply the chat template for instruct-tuned models.
        # Passing raw text without the template causes the model to produce
        # degenerate predictions (every position gets the same token).
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for tokenizers without chat template support
            text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        prompt_ids = self.tokenizer.encode(text, return_tensors="pt")
        prompt_len = prompt_ids.shape[1]

        # Append MASK tokens for generation
        mask_ids = torch.full((1, generation_len), self.mask_token_id, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, mask_ids], dim=1)

        # Prompt mask: True for prompt tokens
        prompt_mask = torch.zeros(1, input_ids.shape[1], dtype=torch.bool)
        prompt_mask[0, :prompt_len] = True

        return input_ids, prompt_mask

    def generate(
        self,
        prompt: str,
        generation_len: int = 128,
        block_length: int = 32,
        scheduler=None,
        num_steps: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        system_prompt: str | None = None,
    ) -> str:
        """High-level generation interface.

        Args:
            prompt:         user message (chat template applied internally)
            generation_len: tokens to generate (divisible by block_length)
            block_length:   tokens per denoising block
            scheduler:      UnmaskingScheduler (defaults to ConfidenceScheduler)
            num_steps:      total denoising steps (divisible by num_blocks)
            temperature:    Gumbel noise; 0 = greedy argmax
            cfg_scale:      classifier-free guidance scale; 0 = disabled
            remasking:      "low_confidence" | "random"
            system_prompt:  optional system message
        """
        from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig

        if scheduler is None:
            from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
            scheduler = ConfidenceScheduler()

        device = next(self._llada.parameters()).device
        input_ids, prompt_mask = self.encode_prompt(prompt, generation_len, system_prompt)
        input_ids = input_ids.to(device)
        prompt_mask = prompt_mask.to(device)

        sampler = DiffusionSampler(
            self,
            scheduler,
            SamplingConfig(
                num_steps=num_steps,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                show_progress=False,
            ),
        )

        result = sampler.sample(
            prompt_ids=input_ids,
            prompt_mask=prompt_mask,
            gen_length=generation_len,
        )

        # Decode only the generated part
        prompt_len = int(prompt_mask[0].sum().item())
        gen_ids = result.sequences[0, prompt_len:]

        logger.debug(
            f"generate(): prompt_len={prompt_len}, gen_len={gen_ids.shape[0]}, "
            f"gen_ids[:20]={gen_ids[:20].tolist()}"
        )
        # ───────────────────────────────────────────────────────────────────────

        # Decode — skip only padding/EOS, not the mask token (already removed).
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        if not generated_text.strip():
            logger.warning(
                f"generate(): decoded output is still empty after safety-net fix. "
                f"mask_token_id={self.mask_token_id}, "
                f"unique gen token ids={gen_ids.unique().tolist()}"
            )

        return generated_text

    @property
    def device(self) -> torch.device:
        return next(self._llada.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._llada.parameters()).dtype
