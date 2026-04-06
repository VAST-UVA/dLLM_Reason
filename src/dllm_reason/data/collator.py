"""Data collators for dLLM training."""

from __future__ import annotations

from typing import Any

import torch


class DiffusionCollator:
    """Collator that prepares batches for diffusion model training.

    Handles padding and creates the mask_token_id-filled tensors
    that dLLMs expect.
    """

    def __init__(self, mask_token_id: int, pad_token_id: int = 0):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "prompt_mask" in batch[0]:
            result["prompt_mask"] = torch.stack([item["prompt_mask"] for item in batch])

        if "answer" in batch[0]:
            result["answer"] = [item["answer"] for item in batch]

        if "question" in batch[0]:
            result["question"] = [item["question"] for item in batch]

        return result
