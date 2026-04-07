"""Standalone LLaDA inference script.

Based on the official GSAI-ML/LLaDA sampling algorithm:
  - Gumbel noise + argmax  (NOT softmax + multinomial)
  - Block-wise denoising
  - Low-confidence remasking
  - Optional CFG for instruct models

Usage:
    python scripts/infer_llada.py --prompt "Write a Python fibonacci function"
    python scripts/infer_llada.py --prompt "..." --gen_length 256 --steps 256 --block_length 32
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ── Mask token lookup ────────────────────────────────────────────────────────

def _get_mask_token_id(model, tokenizer) -> int:
    """Resolve the mask token id from multiple sources in priority order."""
    # 1. model.config.mask_token_id — most reliable for LLaDA
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "mask_token_id", None) is not None:
        return cfg.mask_token_id

    # 2. Look up <|mdm_mask|> or similar by token string
    unk = getattr(tokenizer, "unk_token_id", None)
    for candidate in ("<|mdm_mask|>", "[MASK]", "<mask>", "[mask]"):
        tid = tokenizer.convert_tokens_to_ids(candidate)
        if tid is not None and tid != unk:
            return tid

    # 3. tokenizer.mask_token_id last (may point to wrong token for LLaDA)
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return tokenizer.mask_token_id

    raise ValueError(
        "Cannot find mask_token_id. "
        "Pass it explicitly via --mask_token_id or check your tokenizer."
    )


# ── Core sampling helpers ─────────────────────────────────────────────────────

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for temperature-controlled discrete sampling.

    temperature=0  →  pure argmax (greedy)
    temperature>0  →  stochastic, higher = more diverse
    """
    if temperature == 0:
        return logits
    noise = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)
    return logits + temperature * noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute how many tokens to unmask per step (spread evenly)."""
    # mask_index: (B, L) bool
    mask_counts = mask_index.sum(dim=1)          # (B,)
    base = mask_counts // steps                  # floor
    remainder = mask_counts % steps              # leftover
    # First `remainder` steps get one extra token
    num_tokens = torch.zeros(mask_counts.shape[0], steps,
                              dtype=torch.long, device=mask_index.device)
    num_tokens += base.unsqueeze(1)
    for b in range(mask_counts.shape[0]):
        num_tokens[b, :remainder[b]] += 1
    return num_tokens                            # (B, steps)


# ── Main generation function ──────────────────────────────────────────────────

@torch.no_grad()
def llada_generate(
    model,
    tokenizer,
    prompt: str,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",   # "low_confidence" | "random"
    mask_token_id: int | None = None,
) -> str:
    """Generate text with LLaDA block-wise diffusion sampling.

    Args:
        model:        loaded AutoModel (LLaDA)
        tokenizer:    matching tokenizer
        prompt:       user message (chat template applied internally)
        gen_length:   number of tokens to generate (must be divisible by block_length)
        steps:        total denoising steps  (must be divisible by num_blocks)
        block_length: tokens per denoising block
        temperature:  Gumbel noise scale; 0 = greedy
        cfg_scale:    classifier-free guidance scale (0 = disabled)
        remasking:    confidence-based or random remasking strategy
    """
    assert gen_length % block_length == 0, \
        f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, \
        f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
    steps_per_block = steps // num_blocks

    device = next(model.parameters()).device
    mask_id = mask_token_id if mask_token_id is not None else _get_mask_token_id(model, tokenizer)
    print(f"[INFO] mask_token_id = {mask_id}  ({repr(tokenizer.decode([mask_id]))})")

    # ── Encode prompt with chat template ──────────────────────────────────────
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    print(f"[INFO] prompt_len = {prompt_len},  gen_length = {gen_length}")

    # ── Initialise: prompt tokens + all-mask generation area ─────────────────
    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device
    )
    x[:, :prompt_len] = input_ids
    prompt_index = (x != mask_id)   # True for prompt positions (used by CFG)

    # ── Block-wise denoising ──────────────────────────────────────────────────
    for block_idx in range(num_blocks):
        b_start = prompt_len + block_idx * block_length
        b_end   = prompt_len + (block_idx + 1) * block_length

        # Which positions belong to this block and are still masked
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, b_start:b_end] = True

        # Pre-compute how many tokens to unmask at each step in this block
        block_masked = x[:, b_start:b_end] == mask_id   # (1, block_length)
        num_transfer = get_num_transfer_tokens(block_masked, steps_per_block)
        # num_transfer: (1, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)

            # ── Forward pass (+ optional CFG) ─────────────────────────────
            if cfg_scale > 0:
                # Unconditional: replace prompt tokens with mask
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)      # (2, L)
                logits_cat = model(x_cat).logits          # (2, L, V)
                logits, un_logits = logits_cat.chunk(2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits                  # (1, L, V)

            # ── Sample candidate tokens ───────────────────────────────────
            logits_noisy = add_gumbel_noise(logits, temperature)
            x0 = logits_noisy.argmax(dim=-1)              # (1, L)

            # ── Confidence for remasking / token selection ────────────────
            if remasking == "low_confidence":
                p = F.softmax(logits.double(), dim=-1)    # stable fp64
                x0_p = torch.gather(
                    p, dim=-1, index=x0.unsqueeze(-1)
                ).squeeze(-1).float()                     # (1, L)
            else:  # random
                x0_p = torch.rand(x.shape, device=device)

            # Only consider current block; ignore non-block positions
            x0_p = x0_p.masked_fill(~block_mask, -float("inf"))

            # Keep prompt tokens unchanged
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p,
                torch.full_like(x0_p, -float("inf"))
            )

            # ── Select top-k tokens to commit this step ───────────────────
            n = int(num_transfer[0, step].item())
            if n > 0:
                _, top_idx = torch.topk(confidence[0], k=n)
                transfer = torch.zeros_like(x, dtype=torch.bool)
                transfer[0, top_idx] = True
                x[transfer] = x0[transfer]

        # Debug: show how much of this block was filled
        remaining = int((x[:, b_start:b_end] == mask_id).sum().item())
        filled = block_length - remaining
        print(f"[INFO] block {block_idx+1}/{num_blocks}: filled {filled}/{block_length} tokens")

    # ── Decode generated tokens ───────────────────────────────────────────────
    generated_ids = x[0, prompt_len:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLaDA inference")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt", default="Write a Python function that returns the nth Fibonacci number.")
    parser.add_argument("--gen_length",   type=int,   default=128)
    parser.add_argument("--steps",        type=int,   default=128)
    parser.add_argument("--block_length", type=int,   default=32)
    parser.add_argument("--temperature",  type=float, default=0.0)
    parser.add_argument("--cfg_scale",    type=float, default=0.0)
    parser.add_argument("--remasking",    default="low_confidence",
                        choices=["low_confidence", "random"])
    parser.add_argument("--mask_token_id", type=int, default=None,
                        help="Override mask token id if tokenizer does not expose it")
    parser.add_argument("--dtype",        default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }

    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model from {args.model} ...")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
    ).cuda().eval()

    print(f"\nPrompt:\n{args.prompt}\n")
    output = llada_generate(
        model, tokenizer,
        prompt=args.prompt,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        mask_token_id=args.mask_token_id,
    )
    print(f"\n{'='*60}\nGenerated output:\n{'='*60}\n{output}\n{'='*60}")


if __name__ == "__main__":
    main()
