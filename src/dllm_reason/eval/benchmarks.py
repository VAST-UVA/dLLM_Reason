"""Benchmark evaluators for MBPP, HumanEval, HotpotQA, MMLU.

Each evaluator:
1. Loads the dataset
2. Formats prompts for LLaDA
3. Generates responses using a given scheduler
4. Computes the benchmark metric
5. Returns a detailed results dict
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dllm_reason.eval.metrics import (
    exact_match, f1_score, extract_number,
    extract_multiple_choice, pass_at_k,
)
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkEvaluator:
    """Abstract benchmark evaluator."""

    def __init__(
        self,
        model,           # LLaDAWrapper or any DiffusionLM
        scheduler,       # UnmaskingScheduler
        num_steps: int = 128,
        temperature: float = 0.0,
        batch_size: int = 1,
        max_new_tokens: int = 512,
        num_samples: int | None = None,  # None = all
    ):
        self.model = model
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples

    def _generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.model.generate(
            prompt,
            generation_len=self.max_new_tokens,
            scheduler=self.scheduler,
            num_steps=self.num_steps,
            temperature=self.temperature,
            system_prompt=system_prompt,
        )

    def evaluate(self) -> dict[str, Any]:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# MBPP
# ──────────────────────────────────────────────────────────────────────────────

class MBPPEvaluator(BenchmarkEvaluator):
    """MBPP: Mostly Basic Programming Problems.

    Metrics: pass@1, pass@10
    Format: Given a docstring, generate a Python function.
    """

    SYSTEM_PROMPT = (
        "You are an expert Python programmer. "
        "Write a complete Python function based on the given description. "
        "Output only the function code, no explanation."
    )

    def evaluate(self) -> dict[str, Any]:
        from datasets import load_dataset

        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        items = list(dataset)
        if self.num_samples:
            items = items[:self.num_samples]

        results = []
        for item in tqdm(items, desc="MBPP"):
            task_id = item.get("task_id", item.get("source_file", str(len(results))))
            prompt = item["prompt"]
            test_list = item["test_list"]
            canonical = item["code"]

            # Generate N samples for pass@k
            n_samples = 1
            passed = 0

            for _ in range(n_samples):
                generated = self._generate(prompt, self.SYSTEM_PROMPT)
                code = self._extract_code(generated)
                if self._run_tests(code, test_list):
                    passed += 1

            results.append({
                "task_id": task_id,
                "n": n_samples,
                "passed": passed,
                "pass@1": pass_at_k(n_samples, passed, 1),
            })

        pass_1 = sum(r["pass@1"] for r in results) / len(results)
        logger.info(f"MBPP pass@1: {pass_1:.4f} ({len(results)} problems)")

        return {
            "benchmark": "mbpp",
            "pass@1": pass_1,
            "num_problems": len(results),
            "per_problem": results,
        }

    def _extract_code(self, text: str) -> str:
        """Extract Python code from model output."""
        # Try to find code block
        code_block = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if code_block:
            return code_block.group(1)
        code_block = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if code_block:
            return code_block.group(1)
        # Assume the whole output is code
        return text

    def _run_tests(self, code: str, tests: list[str], timeout: int = 10) -> bool:
        """Execute code + tests in a subprocess, return True if all pass."""
        test_code = "\n".join(tests)
        full_code = f"{code}\n\n{test_code}\n"

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(full_code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HumanEval
# ──────────────────────────────────────────────────────────────────────────────

class HumanEvalEvaluator(BenchmarkEvaluator):
    """HumanEval: Python coding benchmark.

    Metrics: pass@1, pass@10
    Format: Complete a function given its signature + docstring.
    """

    SYSTEM_PROMPT = (
        "Complete the following Python function. "
        "Output only the complete function implementation, no explanation."
    )

    def evaluate(self) -> dict[str, Any]:
        from datasets import load_dataset

        dataset = load_dataset("openai/openai_humaneval", split="test")
        items = list(dataset)
        if self.num_samples:
            items = items[:self.num_samples]

        results = []
        for item in tqdm(items, desc="HumanEval"):
            task_id = item["task_id"]
            prompt = item["prompt"]
            canonical_solution = item["canonical_solution"]
            test = item["test"]
            entry_point = item["entry_point"]

            n_samples = 1
            passed = 0

            for _ in range(n_samples):
                generated = self._generate(prompt, self.SYSTEM_PROMPT)
                # Prepend the original prompt (function signature)
                full_code = prompt + self._extract_completion(generated, prompt)
                if self._run_humaneval_test(full_code, test, entry_point):
                    passed += 1

            results.append({
                "task_id": task_id,
                "n": n_samples,
                "passed": passed,
                "pass@1": pass_at_k(n_samples, passed, 1),
            })

        pass_1 = sum(r["pass@1"] for r in results) / len(results)
        logger.info(f"HumanEval pass@1: {pass_1:.4f} ({len(results)} problems)")

        return {
            "benchmark": "humaneval",
            "pass@1": pass_1,
            "num_problems": len(results),
            "per_problem": results,
        }

    def _extract_completion(self, generated: str, prompt: str) -> str:
        """Extract the function body from generated text."""
        # Remove the prompt if echoed
        if generated.startswith(prompt):
            return generated[len(prompt):]
        # Extract code block
        code_block = re.search(r"```python\n(.*?)```", generated, re.DOTALL)
        if code_block:
            code = code_block.group(1)
            # Remove the function definition if present
            if "def " in code:
                lines = code.split("\n")
                body_lines = []
                in_body = False
                for line in lines:
                    if line.strip().startswith("def "):
                        in_body = True
                        continue
                    if in_body:
                        body_lines.append(line)
                return "\n".join(body_lines)
            return code
        return generated

    def _run_humaneval_test(self, solution: str, test: str, entry_point: str, timeout: int = 10) -> bool:
        full_code = f"{solution}\n\n{test}\n\ncheck({entry_point})\n"
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HotpotQA
# ──────────────────────────────────────────────────────────────────────────────

class HotpotQAEvaluator(BenchmarkEvaluator):
    """HotpotQA: Multi-hop reasoning QA.

    Metrics: Exact Match (EM), F1
    Format: Given a question + context passages, answer the question.
    """

    SYSTEM_PROMPT = (
        "Answer the following question based on the given context. "
        "Think step by step, then provide a concise answer."
    )

    def evaluate(self) -> dict[str, Any]:
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        items = list(dataset)
        if self.num_samples:
            items = items[:self.num_samples]

        em_scores = []
        f1_scores_list = []

        for item in tqdm(items, desc="HotpotQA"):
            question = item["question"]
            answer = item["answer"]
            context = item["context"]

            # Format context: title + sentences
            context_text = ""
            for title, sentences in zip(context["title"], context["sentences"]):
                context_text += f"[{title}]\n" + " ".join(sentences) + "\n\n"

            prompt = f"Context:\n{context_text}\nQuestion: {question}\nAnswer:"

            generated = self._generate(prompt, self.SYSTEM_PROMPT)
            generated = generated.strip().split("\n")[0]  # Take first line

            em = exact_match(generated, answer)
            f1 = f1_score(generated, answer)
            em_scores.append(em)
            f1_scores_list.append(f1)

        avg_em = sum(em_scores) / len(em_scores)
        avg_f1 = sum(f1_scores_list) / len(f1_scores_list)
        logger.info(f"HotpotQA EM: {avg_em:.4f}, F1: {avg_f1:.4f}")

        return {
            "benchmark": "hotpotqa",
            "exact_match": avg_em,
            "f1": avg_f1,
            "num_examples": len(em_scores),
        }


# ──────────────────────────────────────────────────────────────────────────────
# MMLU
# ──────────────────────────────────────────────────────────────────────────────

class MMLUEvaluator(BenchmarkEvaluator):
    """MMLU: Massive Multitask Language Understanding.

    Metrics: Accuracy (5-shot)
    Format: Multiple choice questions (A/B/C/D).
    """

    SYSTEM_PROMPT = (
        "Answer the following multiple choice question. "
        "Think briefly, then respond with just the letter (A, B, C, or D)."
    )

    # Default subjects to test (full MMLU has 57 subjects)
    DEFAULT_SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy",
        "college_computer_science", "college_mathematics",
        "high_school_mathematics", "high_school_physics",
        "logical_fallacies", "machine_learning", "moral_scenarios",
    ]

    def __init__(self, *args, subjects: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.subjects = subjects or self.DEFAULT_SUBJECTS

    def evaluate(self) -> dict[str, Any]:
        from datasets import load_dataset

        all_correct = 0
        all_total = 0
        per_subject = {}

        for subject in tqdm(self.subjects, desc="MMLU subjects"):
            try:
                dataset = load_dataset(
                    "cais/mmlu", subject, split="test"
                )
                val_dataset = load_dataset(
                    "cais/mmlu", subject, split="validation"
                )
            except Exception as e:
                logger.warning(f"Could not load MMLU subject '{subject}': {e}")
                continue

            items = list(dataset)
            if self.num_samples:
                items = items[:self.num_samples // len(self.subjects)]

            # Build few-shot examples from validation
            few_shot = self._build_few_shot(list(val_dataset)[:5])

            correct = 0
            total = 0

            for item in items:
                question = item["question"]
                choices = item["choices"]
                answer_idx = item["answer"]  # 0,1,2,3 -> A,B,C,D
                answer_letter = "ABCD"[answer_idx]

                prompt = self._format_mcq(question, choices, few_shot)
                generated = self._generate(prompt, self.SYSTEM_PROMPT)

                pred = extract_multiple_choice(generated)
                if pred == answer_letter:
                    correct += 1
                total += 1

            acc = correct / max(total, 1)
            per_subject[subject] = {"accuracy": acc, "n": total}
            all_correct += correct
            all_total += total

        overall_acc = all_correct / max(all_total, 1)
        logger.info(f"MMLU accuracy: {overall_acc:.4f} ({all_total} questions)")

        return {
            "benchmark": "mmlu",
            "accuracy": overall_acc,
            "num_questions": all_total,
            "per_subject": per_subject,
        }

    def _format_mcq(self, question: str, choices: list[str], few_shot: str = "") -> str:
        prompt = few_shot
        prompt += f"Question: {question}\n"
        for letter, choice in zip("ABCD", choices):
            prompt += f"{letter}. {choice}\n"
        prompt += "Answer:"
        return prompt

    def _build_few_shot(self, examples: list[dict]) -> str:
        text = ""
        for ex in examples:
            question = ex["question"]
            choices = ex["choices"]
            answer_letter = "ABCD"[ex["answer"]]
            text += f"Question: {question}\n"
            for letter, choice in zip("ABCD", choices):
                text += f"{letter}. {choice}\n"
            text += f"Answer: {answer_letter}\n\n"
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_REGISTRY = {
    "mbpp": MBPPEvaluator,
    "humaneval": HumanEvalEvaluator,
    "hotpotqa": HotpotQAEvaluator,
    "mmlu": MMLUEvaluator,
}
