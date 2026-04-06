"""Evaluation metrics for reasoning benchmarks."""

from __future__ import annotations

import re
import string
from collections import Counter


def exact_match(prediction: str, ground_truth: str) -> float:
    """Normalized exact match."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score (for open-domain QA)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def extract_number(text: str) -> str | None:
    """Extract the final numerical answer from model output."""
    # Look for patterns like "#### 42", "= 42", "answer is 42", etc.
    patterns = [
        r"####\s*([\d,.-]+)",
        r"answer[:\s]+(-?[\d,.-]+)",
        r"=\s*(-?[\d,.-]+)\s*$",
        r"(-?[\d,.-]+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(",", "")
            return num_str
    return None


def extract_multiple_choice(text: str) -> str | None:
    """Extract a multiple-choice answer (A/B/C/D) from model output."""
    text = text.strip()
    # First: look for standalone letter at the end
    patterns = [
        r"\b([ABCD])\b(?:\s*[.\):]?\s*$)",
        r"answer[:\s]+([ABCD])\b",
        r"option[:\s]+([ABCD])\b",
        r"^([ABCD])[.\):]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k using the unbiased estimator from HumanEval paper.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    Returns:
        pass@k score
    """
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)
