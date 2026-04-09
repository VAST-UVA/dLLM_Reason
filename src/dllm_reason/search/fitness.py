"""Fitness functions for DAG search evaluation.

A fitness function takes a model and a DAG, runs inference using the DAG
as the unmasking schedule, and returns a scalar score indicating how well
the DAG works for reasoning.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from dllm_reason.models.base import DiffusionLM
from dllm_reason.graph.dag import TokenDAG
from dllm_reason.scheduler.dag_scheduler import DAGScheduler
from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig


def accuracy_fitness(
    model: DiffusionLM,
    dag: TokenDAG,
    dataloader: DataLoader,
    answer_extractor: Callable[[torch.Tensor], str],
    max_samples: int = 50,
    num_steps: int = 64,
    temperature: float = 0.7,
) -> float:
    """Evaluate a DAG by measuring accuracy on a dataset.

    Generates sequences using the DAG scheduler, extracts answers,
    and computes exact-match accuracy.

    Args:
        model: the dLLM
        dag: the DAG to evaluate
        dataloader: dataset with (input, target_answer) pairs
        answer_extractor: function to extract answer string from generated tokens
        max_samples: max number of samples to evaluate
        num_steps: sampling steps
        temperature: sampling temperature

    Returns:
        Accuracy score in [0, 1]
    """
    scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
    sampler = DiffusionSampler(
        model, scheduler,
        SamplingConfig(num_steps=num_steps, temperature=temperature, show_progress=False),
    )

    correct = 0
    total = 0

    for batch in dataloader:
        if total >= max_samples:
            break

        prompt_ids = batch["input_ids"].to(model.device)
        prompt_mask = batch.get("prompt_mask", None)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(model.device)
        target_answers = batch["answer"]

        if prompt_mask is None:
            prompt_mask = torch.zeros(prompt_ids.shape, dtype=torch.bool, device=prompt_ids.device)
        result = sampler.sample(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            gen_length=prompt_ids.shape[1] - int(prompt_mask[0].sum().item()),
        )

        for i in range(result.sequences.shape[0]):
            pred_answer = answer_extractor(result.sequences[i])
            if pred_answer.strip() == target_answers[i].strip():
                correct += 1
            total += 1

    return correct / max(total, 1)


def perplexity_fitness(
    model: DiffusionLM,
    dag: TokenDAG,
    dataloader: DataLoader,
    max_samples: int = 50,
) -> float:
    """Evaluate a DAG by measuring perplexity (proxy for quality).

    Lower perplexity is better, so we return negative perplexity
    as the fitness (higher = better).
    """
    total_nll = 0.0
    total_tokens = 0

    for batch in dataloader:
        if total_tokens > max_samples * 512:
            break

        x_0 = batch["input_ids"].to(model.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        loss = model.compute_loss(x_0, attention_mask)
        B, L = x_0.shape
        total_nll += loss.item() * B * L
        total_tokens += B * L

    avg_nll = total_nll / max(total_tokens, 1)
    return -avg_nll  # Higher (less negative) = better


def combined_fitness(
    model: DiffusionLM,
    dag: TokenDAG,
    dataloader: DataLoader,
    answer_extractor: Callable,
    accuracy_weight: float = 0.8,
    perplexity_weight: float = 0.2,
    **kwargs,
) -> float:
    """Combined fitness: weighted sum of accuracy and perplexity."""
    acc = accuracy_fitness(model, dag, dataloader, answer_extractor, **kwargs)
    ppl = perplexity_fitness(model, dag, dataloader)

    # Normalize perplexity to [0, 1] range (rough)
    ppl_normalized = max(0, 1.0 + ppl / 10.0)  # ppl is negative

    return accuracy_weight * acc + perplexity_weight * ppl_normalized
