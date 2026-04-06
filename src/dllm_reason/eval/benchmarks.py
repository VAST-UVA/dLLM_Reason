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
            prompt = item.get("prompt", item.get("text", ""))
            test_list = item["test_list"]
            canonical = item["code"]

            # Generate N samples for pass@k
            n_samples = 1
            passed = 0
            sample_details = []

            for _ in range(n_samples):
                generated = self._generate(prompt, self.SYSTEM_PROMPT)
                code = self._extract_code(generated)

                # Diagnostic: log first few samples so raw output is always visible
                if len(results) < 3:
                    logger.info(
                        f"MBPP #{task_id} raw output ({len(generated)} chars):\n"
                        f"{generated[:500]}\n---extracted---\n{code[:500]}"
                    )

                run_info = self._run_tests(code, test_list)
                if run_info["passed"]:
                    passed += 1
                sample_details.append({
                    "raw_output": generated,
                    "extracted_code": code,
                    "passed": run_info["passed"],
                    "timed_out": run_info["timed_out"],
                    "stderr": run_info["stderr"],
                    "stdout": run_info["stdout"],
                    "error": run_info["error"],
                })

            # Log failures immediately so problems are visible in real time
            if passed == 0:
                detail = sample_details[0]
                # Always show what code was actually extracted, so mismatched
                # function names / empty extractions are immediately visible.
                logger.warning(
                    f"[MBPP {task_id}] FAILED\n"
                    f"  extracted_code: {detail['extracted_code'][:300]!r}\n"
                    f"  raw_output:     {detail['raw_output'][:200]!r}"
                )
                if detail["timed_out"]:
                    logger.warning(f"[MBPP {task_id}] TIMEOUT")
                elif detail["stderr"]:
                    logger.warning(f"[MBPP {task_id}] STDERR: {detail['stderr'][:300]}")
                elif detail["error"]:
                    logger.warning(f"[MBPP {task_id}] ERROR: {detail['error']}")

            results.append({
                "task_id": task_id,
                "n": n_samples,
                "passed": passed,
                "pass@1": pass_at_k(n_samples, passed, 1),
                "samples": sample_details,
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

    def _run_tests(self, code: str, tests: list[str], timeout: int = 10) -> dict:
        """Execute code + tests in a subprocess.

        Returns a dict with keys:
            passed    (bool) – all tests passed
            timed_out (bool) – subprocess hit the timeout
            stderr    (str)  – captured stderr (syntax / runtime errors)
            stdout    (str)  – captured stdout
            error     (str)  – unexpected exception message, if any
        """
        test_code = "\n".join(tests)
        full_code = f"{code}\n\n{test_code}\n"
        tmp_path = None

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
            return {
                "passed": result.returncode == 0,
                "timed_out": False,
                "stderr": result.stderr.strip(),
                "stdout": result.stdout.strip(),
                "error": "",
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "timed_out": True,
                    "stderr": "", "stdout": "", "error": f"timeout>{timeout}s"}
        except Exception as exc:
            return {"passed": False, "timed_out": False,
                    "stderr": "", "stdout": "", "error": str(exc)}
        finally:
            if tmp_path:
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
            sample_details = []

            for _ in range(n_samples):
                generated = self._generate(prompt, self.SYSTEM_PROMPT)
                completion = self._extract_completion(generated, prompt)
                full_code = prompt + completion

                # Diagnostic: log first few samples so raw output is always visible
                if len(results) < 3:
                    logger.info(
                        f"HumanEval {task_id} raw output ({len(generated)} chars):\n"
                        f"{generated[:500]}\n---completion---\n{completion[:500]}"
                    )

                run_info = self._run_humaneval_test(full_code, test, entry_point)
                if run_info["passed"]:
                    passed += 1
                sample_details.append({
                    "raw_output": generated,
                    "extracted_completion": completion,
                    "full_code": full_code,
                    "passed": run_info["passed"],
                    "timed_out": run_info["timed_out"],
                    "stderr": run_info["stderr"],
                    "stdout": run_info["stdout"],
                    "error": run_info["error"],
                })

            # Log failures immediately
            if passed == 0:
                detail = sample_details[0]
                logger.warning(
                    f"[HumanEval {task_id}] FAILED\n"
                    f"  extracted_completion: {detail['extracted_completion'][:300]!r}\n"
                    f"  raw_output:           {detail['raw_output'][:200]!r}"
                )
                if detail["timed_out"]:
                    logger.warning(f"[HumanEval {task_id}] TIMEOUT")
                elif detail["stderr"]:
                    logger.warning(f"[HumanEval {task_id}] STDERR: {detail['stderr'][:300]}")
                elif detail["error"]:
                    logger.warning(f"[HumanEval {task_id}] ERROR: {detail['error']}")

            results.append({
                "task_id": task_id,
                "n": n_samples,
                "passed": passed,
                "pass@1": pass_at_k(n_samples, passed, 1),
                "samples": sample_details,
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
        """Extract the function body completion from generated text.

        LLaDA is a masked-diffusion model: it does NOT echo the prompt back.
        The generated text is the completion only (function body).
        We try four strategies in order:

        1. Strip prompt prefix if the model did echo it (rare).
        2. Extract a fenced code block (```python / ```) and strip any leading
           function-def line so only the body (with correct indentation) remains.
        3. If the raw output contains a def block, extract the indented body.
        4. Fall back: indent each non-empty line by 4 spaces so it can be
           appended to the function signature in `prompt`.
        """
        # Strategy 1: model echoed the prompt
        if generated.startswith(prompt):
            return generated[len(prompt):]

        # Strategy 2: fenced code block
        code_block = re.search(r"```(?:python)?\n(.*?)```", generated, re.DOTALL)
        if code_block:
            code = code_block.group(1)
            lines = code.split("\n")
            body_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    body_start = i + 1
                    break
            if body_start is not None:
                return "\n".join(lines[body_start:])
            return code

        # Strategy 3: dLLM raw output contains a def block — extract body only
        if "def " in generated:
            lines = generated.split("\n")
            body_lines: list[str] = []
            in_body = False
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith("def "):
                    in_body = True
                    continue
                if in_body:
                    # Stop at next top-level def or class
                    if stripped and not line[0].isspace() and not stripped.startswith("#"):
                        if stripped.startswith("def ") or stripped.startswith("class "):
                            break
                    body_lines.append(line)
            if body_lines:
                return "\n".join(body_lines)

        # Strategy 4: assume the whole output is the function body — indent it
        logger.debug(
            "HumanEval _extract_completion: no code fence / def found, "
            f"indenting raw output (first 120 chars): {generated[:120]!r}"
        )
        lines = generated.strip().split("\n")
        indented = []
        for line in lines:
            if line.strip():
                if not line.startswith(" ") and not line.startswith("\t"):
                    line = "    " + line
                indented.append(line)
            else:
                indented.append(line)
        return "\n".join(indented)

    def _run_humaneval_test(self, solution: str, test: str, entry_point: str, timeout: int = 10) -> dict:
        """Run HumanEval test harness in a subprocess.

        Returns a dict with keys:
            passed    (bool) – check() passed
            timed_out (bool) – subprocess hit the timeout
            stderr    (str)  – captured stderr
            stdout    (str)  – captured stdout
            error     (str)  – unexpected exception message, if any
        """
        full_code = f"{solution}\n\n{test}\n\ncheck({entry_point})\n"
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
            return {
                "passed": result.returncode == 0,
                "timed_out": False,
                "stderr": result.stderr.strip(),
                "stdout": result.stdout.strip(),
                "error": "",
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "timed_out": True,
                    "stderr": "", "stdout": "", "error": f"timeout>{timeout}s"}
        except Exception as exc:
            return {"passed": False, "timed_out": False,
                    "stderr": "", "stdout": "", "error": str(exc)}
        finally:
            if tmp_path:
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
