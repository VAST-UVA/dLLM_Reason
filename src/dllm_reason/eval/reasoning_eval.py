"""Reasoning-task evaluation for dLLMs with different unmasking strategies.

Evaluates model + scheduler combinations on reasoning datasets (GSM8K, MATH, ARC)
and computes standard reasoning metrics. Designed to produce the core results table
for the paper comparing baseline schedulers vs DAG-guided schedulers.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dllm_reason.eval.metrics import (
    exact_match, f1_score, extract_number, extract_multiple_choice,
)
from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig
from dllm_reason.models.base import DiffusionLM
from dllm_reason.scheduler.base import UnmaskingScheduler
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating one (model, scheduler, dataset) combination."""
    model_name: str
    scheduler_name: str
    dataset_name: str
    accuracy: float
    f1: float
    num_samples: int
    avg_steps: int
    elapsed_seconds: float
    per_sample: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "scheduler": self.scheduler_name,
            "dataset": self.dataset_name,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "num_samples": self.num_samples,
            "avg_steps": self.avg_steps,
            "elapsed_seconds": self.elapsed_seconds,
        }


class ReasoningEvaluator:
    """Evaluate dLLM reasoning performance across models, schedulers, and datasets.

    Supports GSM8K, MATH, ARC-Challenge as reasoning benchmarks.
    Produces per-sample results and aggregate metrics.
    """

    def __init__(
        self,
        model: DiffusionLM,
        tokenizer,
        max_seq_len: int = 512,
        generation_len: int = 256,
        num_steps: int = 64,
        temperature: float = 0.7,
        num_samples: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.generation_len = generation_len
        self.num_steps = num_steps
        self.temperature = temperature
        self.num_samples = num_samples
        self.device = device or (model.device if hasattr(model, "device") else "cpu")

    def evaluate(
        self,
        scheduler: UnmaskingScheduler,
        dataset: list[dict],
        dataset_name: str,
        scheduler_name: str,
        model_name: str = "model",
    ) -> EvalResult:
        """Run evaluation of one (model, scheduler, dataset) combination.

        Args:
            scheduler: the unmasking scheduler to evaluate
            dataset: list of {"question": str, "answer": str, ...}
            dataset_name: name for logging/results
            scheduler_name: name for logging/results
            model_name: name for logging/results

        Returns:
            EvalResult with accuracy, F1, and per-sample details
        """
        if self.num_samples:
            dataset = dataset[:self.num_samples]

        sampler = DiffusionSampler(
            self.model,
            scheduler,
            SamplingConfig(
                num_steps=self.num_steps,
                temperature=self.temperature,
                show_progress=False,
            ),
        )

        em_scores = []
        f1_scores = []
        per_sample = []
        start_time = time.time()

        for item in tqdm(dataset, desc=f"{model_name}+{scheduler_name} on {dataset_name}"):
            question = item["question"]
            gold_answer = str(item["answer"]).strip()

            # Prepare input
            prompt = self._format_prompt(question, dataset_name)
            prompt_ids, prompt_mask = self._encode_prompt(prompt)

            prompt_ids = prompt_ids.to(self.device)
            prompt_mask = prompt_mask.to(self.device)

            # Generate
            result = sampler.sample(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                gen_length=prompt_ids.shape[1] - int(prompt_mask[0].sum().item()),
            )

            # Decode generated portion
            prompt_len = prompt_mask[0].sum().item()
            gen_ids = result.sequences[0, prompt_len:]
            generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Extract answer
            pred_answer = self._extract_answer(generated, dataset_name)

            em = exact_match(pred_answer, gold_answer)
            f1 = f1_score(pred_answer, gold_answer)
            em_scores.append(em)
            f1_scores.append(f1)

            per_sample.append({
                "question": question[:100],
                "gold": gold_answer,
                "generated": generated[:200],
                "predicted": pred_answer,
                "em": em,
                "f1": f1,
            })

        elapsed = time.time() - start_time
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        logger.info(
            f"{model_name}+{scheduler_name} on {dataset_name}: "
            f"EM={avg_em:.4f}, F1={avg_f1:.4f}, "
            f"n={len(em_scores)}, t={elapsed:.1f}s"
        )

        return EvalResult(
            model_name=model_name,
            scheduler_name=scheduler_name,
            dataset_name=dataset_name,
            accuracy=avg_em,
            f1=avg_f1,
            num_samples=len(em_scores),
            avg_steps=self.num_steps,
            elapsed_seconds=elapsed,
            per_sample=per_sample,
        )

    def _format_prompt(self, question: str, dataset_name: str) -> str:
        if dataset_name == "gsm8k":
            return (
                f"Solve the following math problem step by step.\n"
                f"Problem: {question}\n"
                f"Solution:"
            )
        elif dataset_name == "math":
            return f"Problem: {question}\nSolution:"
        elif dataset_name in ("arc", "arc_challenge"):
            return f"Question: {question}\nAnswer:"
        else:
            return f"Q: {question}\nA:"

    def _encode_prompt(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompt and append MASK tokens for generation."""
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = prompt_ids.shape[1]

        # Cap prompt length
        if prompt_len > self.max_seq_len - self.generation_len:
            prompt_ids = prompt_ids[:, -(self.max_seq_len - self.generation_len):]
            prompt_len = prompt_ids.shape[1]

        mask_ids = torch.full(
            (1, self.generation_len),
            self.model.mask_token_id,
            dtype=torch.long,
        )
        input_ids = torch.cat([prompt_ids, mask_ids], dim=1)

        prompt_mask = torch.zeros(1, input_ids.shape[1], dtype=torch.bool)
        prompt_mask[0, :prompt_len] = True

        return input_ids, prompt_mask

    def _extract_answer(self, generated: str, dataset_name: str) -> str:
        if dataset_name == "gsm8k":
            num = extract_number(generated)
            return num if num is not None else generated.strip()
        elif dataset_name in ("arc", "arc_challenge"):
            letter = extract_multiple_choice(generated)
            return letter if letter is not None else generated.strip()
        else:
            return generated.strip().split("\n")[0]


class MultiSchedulerComparison:
    """Run a grid comparison: N schedulers × M datasets → results table.

    Produces the core ablation table for the paper:
    rows = schedulers, columns = datasets, cells = accuracy.
    """

    def __init__(
        self,
        evaluator: ReasoningEvaluator,
        output_dir: str | Path = "results",
    ):
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        schedulers: dict[str, UnmaskingScheduler],
        datasets: dict[str, list[dict]],
        model_name: str = "model",
        resume: bool = True,
    ) -> dict[str, dict[str, EvalResult]]:
        """Run all (scheduler, dataset) combinations.

        Args:
            schedulers: {name: scheduler_instance}
            datasets: {name: list_of_examples}
            model_name: model identifier for output files
            resume: skip already-computed results

        Returns:
            {scheduler_name: {dataset_name: EvalResult}}
        """
        all_results: dict[str, dict[str, EvalResult]] = {}

        for sched_name, scheduler in schedulers.items():
            all_results[sched_name] = {}
            for ds_name, dataset in datasets.items():
                run_key = f"{model_name}_{sched_name}_{ds_name}"
                result_path = self.output_dir / f"{run_key}.json"

                if resume and result_path.exists():
                    logger.info(f"[SKIP] {run_key}")
                    with open(result_path) as f:
                        data = json.load(f)
                    # Reconstruct minimal result
                    result = EvalResult(
                        model_name=data["model"],
                        scheduler_name=data["scheduler"],
                        dataset_name=data["dataset"],
                        accuracy=data["accuracy"],
                        f1=data["f1"],
                        num_samples=data["num_samples"],
                        avg_steps=data["avg_steps"],
                        elapsed_seconds=data["elapsed_seconds"],
                    )
                else:
                    result = self.evaluator.evaluate(
                        scheduler=scheduler,
                        dataset=dataset,
                        dataset_name=ds_name,
                        scheduler_name=sched_name,
                        model_name=model_name,
                    )
                    with open(result_path, "w") as f:
                        json.dump(
                            {**result.to_dict(), "per_sample": result.per_sample},
                            f, indent=2,
                        )

                all_results[sched_name][ds_name] = result

        # Print summary table
        self._print_table(all_results, datasets.keys())

        # Save summary
        summary = {
            sched: {ds: r.to_dict() for ds, r in ds_results.items()}
            for sched, ds_results in all_results.items()
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return all_results

    def _print_table(
        self,
        results: dict[str, dict[str, EvalResult]],
        dataset_names,
    ):
        ds_names = list(dataset_names)
        header = f"{'Scheduler':<22}" + "".join(f"  {d:>14}" for d in ds_names)
        print(f"\n{'='*len(header)}")
        print("RESULTS (Exact Match Accuracy)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for sched_name, ds_results in results.items():
            row = f"{sched_name:<22}"
            for ds in ds_names:
                if ds in ds_results:
                    row += f"  {ds_results[ds].accuracy:>14.4f}"
                else:
                    row += f"  {'N/A':>14}"
            print(row)
        print("=" * len(header))
