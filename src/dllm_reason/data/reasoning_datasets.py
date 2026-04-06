"""Dataset loading for reasoning benchmarks.

Supports: GSM8K, MATH, ARC-Challenge, ProntoQA.
All datasets are loaded via HuggingFace datasets and formatted
into a common structure for training and evaluation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


class ReasoningDataset(Dataset):
    """Common wrapper for reasoning datasets.

    Each item contains:
    - input_ids: tokenized input (prompt + generation space)
    - attention_mask: 1 for real tokens, 0 for padding
    - prompt_mask: True for prompt positions (not to be masked during inference)
    - answer: ground truth answer string
    - question: original question string
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer,
        max_seq_len: int = 512,
        prompt_template: str = "Q: {question}\nA: {answer}",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]

        # For training: full sequence (question + answer)
        full_text = self.prompt_template.format(question=question, answer=answer)
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Compute prompt mask (question part only)
        prompt_text = f"Q: {question}\nA: "
        prompt_encoding = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_len = min(prompt_encoding["input_ids"].shape[1], self.max_seq_len)

        prompt_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        prompt_mask[:prompt_len] = True

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt_mask": prompt_mask,
            "answer": answer,
            "question": question,
        }


def load_gsm8k(split: str = "train", **kwargs) -> list[dict]:
    """Load GSM8K dataset."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    data = []
    for item in ds:
        # Extract final numerical answer from the solution
        answer_text = item["answer"]
        # GSM8K answers end with "#### <number>"
        final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
        data.append({
            "question": item["question"],
            "answer": final_answer,
            "solution": answer_text,
        })
    return data


def load_math(split: str = "train", **kwargs) -> list[dict]:
    """Load MATH dataset."""
    from datasets import load_dataset

    ds = load_dataset("hendrycks/competition_math", split=split)
    data = []
    for item in ds:
        data.append({
            "question": item["problem"],
            "answer": item["solution"],
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        })
    return data


def load_arc(split: str = "train", challenge: bool = True, **kwargs) -> list[dict]:
    """Load ARC dataset (Challenge or Easy)."""
    from datasets import load_dataset

    subset = "ARC-Challenge" if challenge else "ARC-Easy"
    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    data = []
    for item in ds:
        choices = item["choices"]
        answer_key = item["answerKey"]
        # Find answer text
        answer_idx = choices["label"].index(answer_key) if answer_key in choices["label"] else 0
        answer_text = choices["text"][answer_idx]

        question = item["question"]
        for label, text in zip(choices["label"], choices["text"]):
            question += f"\n{label}. {text}"

        data.append({
            "question": question,
            "answer": answer_text,
            "answer_key": answer_key,
        })
    return data


def load_prontoqa(split: str = "train", **kwargs) -> list[dict]:
    """Load ProntoQA logical reasoning dataset."""
    from datasets import load_dataset

    ds = load_dataset("renma/ProntoQA", split=split)
    data = []
    for item in ds:
        data.append({
            "question": item.get("question", item.get("context", "")),
            "answer": item.get("answer", item.get("label", "")),
        })
    return data


DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "arc": load_arc,
    "prontoqa": load_prontoqa,
}


def load_reasoning_dataset(name: str, split: str = "train", **kwargs) -> list[dict]:
    """Load a reasoning dataset by name."""
    if name not in DATASET_LOADERS:
        available = ", ".join(sorted(DATASET_LOADERS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_LOADERS[name](split=split, **kwargs)
