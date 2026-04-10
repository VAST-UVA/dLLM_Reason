"""Tokenizer utilities for dLLM models."""

from __future__ import annotations

from transformers import AutoTokenizer

from dllm_reason.utils.local_resolve import resolve_model_path


def get_tokenizer(
    name_or_path: str = "gpt2",
    add_mask_token: bool = True,
) -> AutoTokenizer:
    """Load and configure a tokenizer for dLLM use.

    Args:
        name_or_path: HuggingFace tokenizer name or path
        add_mask_token: whether to add a [MASK] token if not present

    Returns:
        Configured tokenizer with mask token
    """
    name_or_path = resolve_model_path(name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add mask token if needed
    if add_mask_token and tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    return tokenizer
