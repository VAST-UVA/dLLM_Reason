"""Collect DAG episodes: prompt → strategy → generate → evaluate → store.

For each (prompt, strategy) pair the script:
  1. Builds the requested scheduler / DAG
  2. Runs the diffusion model to generate an answer
  3. Auto-evaluates correctness (math answer matching or model-as-judge)
  4. Stores the episode in an EpisodeStore (SQLite)

Usage
-----
# Quick single prompt
python scripts/collect_episodes.py \\
    --model_id checkpoints/llada-instruct \\
    --prompt "What is 12 * 15?" \\
    --strategy cot \\
    --ground_truth "180"

# Dataset mode (reads JSONL where each line has {"prompt":..., "answer":...})
python scripts/collect_episodes.py \\
    --model_id checkpoints/llada-instruct \\
    --dataset_path data/gsm8k_test.jsonl \\
    --strategy confidence entropy cot \\
    --n_samples 200 \\
    --db_path episodes/gsm8k.db

# With manual annotation (pauses for user input instead of auto-eval)
python scripts/collect_episodes.py \\
    --dataset_path data/custom.jsonl \\
    --strategy cot \\
    --eval_mode manual
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

# Make sure src/ is on the path when running as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dllm_reason.library.episode import DAGEpisode, EpisodeStore
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


# ── Answer evaluation helpers ─────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d+(?:[\.,]\d+)*")


def _extract_last_number(text: str) -> str | None:
    """Return the last number found in text (comma-stripped), or None."""
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def auto_eval_math(output: str, ground_truth: str) -> tuple[bool, float, str]:
    """Simple numeric answer matching for math tasks.

    Returns (correct, score, comment).
    """
    pred = _extract_last_number(output)
    gold = _extract_last_number(ground_truth)
    if pred is None or gold is None:
        return False, 0.0, f"could not parse numbers: pred={pred!r} gold={gold!r}"
    try:
        match = abs(float(pred.replace(",", "")) - float(gold.replace(",", ""))) < 1e-3
    except ValueError:
        match = pred.strip() == gold.strip()
    score = 1.0 if match else 0.0
    comment = f"pred={pred}  gold={gold}  {'✓' if match else '✗'}"
    return match, score, comment


def auto_eval_exact(output: str, ground_truth: str) -> tuple[bool, float, str]:
    """Exact-match or substring evaluation."""
    norm = lambda s: s.strip().lower()
    match = norm(ground_truth) in norm(output)
    score = 1.0 if match else 0.0
    return match, score, f"exact_match={match}"


def manual_eval(prompt: str, output: str, ground_truth: str) -> tuple[bool, float, str]:
    """Pause for user to annotate correctness."""
    print("\n" + "=" * 60)
    print(f"PROMPT:\n{prompt}")
    print(f"\nMODEL OUTPUT:\n{output}")
    if ground_truth:
        print(f"\nGROUND TRUTH:\n{ground_truth}")
    print("=" * 60)
    while True:
        raw = input("Correct? [y/n/s(skip)]: ").strip().lower()
        if raw in ("y", "yes"):
            comment = input("Comment (Enter to skip): ").strip()
            return True, 1.0, comment
        if raw in ("n", "no"):
            comment = input("Comment (Enter to skip): ").strip()
            return False, 0.0, comment
        if raw in ("s", "skip"):
            return None, float("nan"), "skipped"
        print("  Please enter y / n / s")


# ── Scheduler / DAG builder (reused from scripts/serve.py) ───────────────────

def build_scheduler(strategy: str, gen_len: int, block_length: int, device):
    """Build an UnmaskingScheduler from a strategy name."""
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
    from dllm_reason.scheduler.random_scheduler import RandomScheduler
    from dllm_reason.scheduler.linear_scheduler import LinearScheduler
    from dllm_reason.scheduler.entropy_scheduler import EntropyScheduler
    from dllm_reason.scheduler.semi_ar_scheduler import SemiAutoregressiveScheduler
    from dllm_reason.scheduler.maskgit_scheduler import MaskGITCosineScheduler
    from dllm_reason.scheduler.critical_token_scheduler import CriticalTokenFirstScheduler
    from dllm_reason.scheduler.curriculum_scheduler import CurriculumScheduler
    from dllm_reason.scheduler.adaptive_dynamic_scheduler import AdaptiveDynamicScheduler

    flat_schedulers = {
        "confidence":           ConfidenceScheduler,
        "random":               RandomScheduler,
        "linear":               LinearScheduler,
        "entropy":              EntropyScheduler,
        "maskgit_cosine":       MaskGITCosineScheduler,
        "critical_token_first": CriticalTokenFirstScheduler,
        "curriculum":           CurriculumScheduler,
        "adaptive_dynamic":     AdaptiveDynamicScheduler,
    }
    if strategy in flat_schedulers:
        if strategy == "semi_ar":
            return SemiAutoregressiveScheduler(block_size=block_length)
        return flat_schedulers[strategy]()

    # DAG-based strategies
    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag, answer_first_dag,
    )

    dag = None
    if strategy == "cot":
        dag = chain_of_thought_dag(gen_len, num_steps=4, device=device)
    elif strategy == "skeleton":
        dag = skeleton_then_detail_dag(
            gen_len, list(range(0, gen_len, 3)), list(range(1, gen_len, 3)),
            device=device,
        )
    elif strategy == "bidirectional":
        dag = bidirectional_dag(gen_len, num_segments=4, device=device)
    elif strategy == "answer_first":
        dag = answer_first_dag(
            gen_len, list(range(int(gen_len * 0.8), gen_len)), device=device,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return DAGScheduler(dag, sub_strategy="confidence_topk"), dag


# ── Single-episode collection ─────────────────────────────────────────────────

def collect_one(
    *,
    model,
    tokenizer,
    prompt: str,
    strategy: str,
    ground_truth: str = "",
    task_type: str = "general",
    gen_length: int = 128,
    num_steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    eval_mode: str = "auto",   # "auto" | "manual" | "none"
    model_id: str = "",
    metadata: dict | None = None,
) -> DAGEpisode:
    """Generate one output with the given strategy and return a DAGEpisode."""
    device = next(model.parameters()).device

    # Build scheduler (and possibly dag)
    result = build_scheduler(strategy, gen_length, block_length, device)
    if isinstance(result, tuple):
        scheduler, dag = result
    else:
        scheduler, dag = result, None

    # Encode prompt
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt",
    ).to(device)
    prompt_len = input_ids.shape[1]

    # Mask-fill generation area
    gen_ids = torch.full(
        (1, gen_length), model.mask_token_id,
        dtype=torch.long, device=device,
    )
    x = torch.cat([input_ids, gen_ids], dim=1)
    prompt_mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=device)
    prompt_mask[:, :prompt_len] = True

    # Run sampler
    from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig
    cfg = SamplingConfig(
        num_steps=num_steps,
        block_length=block_length,
        temperature=temperature,
        show_progress=False,
    )
    sampler = DiffusionSampler(model, scheduler, config=cfg)
    t0 = time.time()
    result_ids = sampler.sample(x, prompt_mask, gen_length, device=device).sequences
    elapsed = time.time() - t0

    # Decode only the generated part
    gen_tokens = result_ids[0, prompt_len:]
    output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Determine task_type from caller or ground_truth heuristics
    if task_type == "general" and ground_truth:
        if _extract_last_number(ground_truth):
            task_type = "math"

    # Evaluate
    correct: bool | None = None
    score:   float = float("nan")
    comment: str = ""

    if eval_mode == "auto" and ground_truth:
        if task_type == "math":
            correct, score, comment = auto_eval_math(output_text, ground_truth)
        else:
            correct, score, comment = auto_eval_exact(output_text, ground_truth)
    elif eval_mode == "manual":
        ev = manual_eval(prompt, output_text, ground_truth)
        correct, score, comment = ev
    # eval_mode == "none": leave correct=None

    # Build adjacency if a DAG was used
    dag_adj = None
    dag_seq_len = 0
    if dag is not None:
        dag_adj = DAGEpisode.adjacency_from_dag(dag)
        dag_seq_len = dag.seq_len

    ep = DAGEpisode(
        prompt        = prompt,
        task_type     = task_type,
        ground_truth  = ground_truth,
        strategy_name = strategy,
        dag_seq_len   = dag_seq_len,
        dag_adjacency = dag_adj,
        output        = output_text,
        correct       = correct,
        score         = score,
        comment       = comment,
        model_id      = model_id,
        num_steps     = num_steps,
        block_length  = block_length,
        temperature   = temperature,
        metadata      = {
            **(metadata or {}),
            "elapsed_seconds": round(elapsed, 3),
        },
    )
    return ep


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DAG episodes by running the model on prompts."
    )

    # Model
    parser.add_argument("--model_id", default="checkpoints/llada-instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--model_type", default="llada",
                        choices=["llada", "mdlm", "sedd", "d3pm"],
                        help="Wrapper type")
    parser.add_argument("--torch_dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", help="Single prompt (inline)")
    input_group.add_argument("--dataset_path",
                             help="Path to a JSONL file with prompt/answer fields")
    parser.add_argument("--prompt_field", default="prompt",
                        help="Field name for prompt in JSONL")
    parser.add_argument("--answer_field", default="answer",
                        help="Field name for answer in JSONL")
    parser.add_argument("--task_type_field", default="",
                        help="Optional field for task_type in JSONL")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Max number of samples to collect from dataset")

    # Single prompt extras
    parser.add_argument("--ground_truth", default="",
                        help="Expected answer for single-prompt mode")
    parser.add_argument("--task_type", default="general")

    # Strategy
    parser.add_argument(
        "--strategy", nargs="+",
        default=["confidence"],
        help="One or more strategy names. Episodes are collected for ALL of them.",
    )

    # Generation
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Evaluation
    parser.add_argument("--eval_mode", default="auto",
                        choices=["auto", "manual", "none"],
                        help="auto: rule-based, manual: human review, none: skip eval")

    # Storage
    parser.add_argument("--db_path", default="episodes/episodes.db",
                        help="SQLite database path for EpisodeStore")

    # Misc
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def load_model(args):
    """Load the diffusion model and tokenizer."""
    import torch
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.torch_dtype]

    print(f"Loading {args.model_type} from {args.model_id!r} ...")

    if args.model_type == "llada":
        from dllm_reason.models.llada import LLaDAWrapper
        model = LLaDAWrapper(
            model_id=args.model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        tokenizer = model.tokenizer
    elif args.model_type == "mdlm":
        from dllm_reason.models.mdlm import MDLMWrapper
        model = MDLMWrapper(model_id=args.model_id, torch_dtype=dtype, device_map="auto")
        tokenizer = model.tokenizer
    else:
        raise ValueError(f"Unsupported model_type={args.model_type!r}")

    print(f"Model ready on {model.device}")
    return model, tokenizer


def load_dataset(args) -> list[dict]:
    """Read JSONL and return list of dicts."""
    path = Path(args.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.n_samples:
        records = records[: args.n_samples]
    print(f"Loaded {len(records)} records from {path}")
    return records


def main():
    args = parse_args()

    store = EpisodeStore(args.db_path)
    print(f"EpisodeStore at: {args.db_path}")

    model, tokenizer = load_model(args)

    # Collect input list
    if args.prompt:
        inputs = [{"prompt": args.prompt, "answer": args.ground_truth,
                   "task_type": args.task_type}]
    else:
        raw = load_dataset(args)
        inputs = []
        for r in raw:
            inputs.append({
                "prompt":    r.get(args.prompt_field, ""),
                "answer":    r.get(args.answer_field, ""),
                "task_type": r.get(args.task_type_field, args.task_type)
                             if args.task_type_field else args.task_type,
            })

    total = len(inputs) * len(args.strategy)
    print(f"\nCollecting {total} episodes "
          f"({len(inputs)} prompts × {len(args.strategy)} strategies)\n")

    done = 0
    correct_count = 0
    eval_count = 0

    for item in inputs:
        for strategy in args.strategy:
            try:
                ep = collect_one(
                    model         = model,
                    tokenizer     = tokenizer,
                    prompt        = item["prompt"],
                    strategy      = strategy,
                    ground_truth  = item.get("answer", ""),
                    task_type     = item.get("task_type", "general"),
                    gen_length    = args.gen_length,
                    num_steps     = args.num_steps,
                    block_length  = args.block_length,
                    temperature   = args.temperature,
                    eval_mode     = args.eval_mode,
                    model_id      = args.model_id,
                )
                store.add(ep)
                done += 1

                if ep.is_evaluated:
                    eval_count += 1
                    if ep.correct:
                        correct_count += 1

                acc_str = (
                    f"  acc={correct_count/eval_count:.1%}" if eval_count else ""
                )
                if args.verbose:
                    print(f"  [{done}/{total}] {ep}")
                else:
                    print(
                        f"\r  [{done}/{total}] strategy={strategy:<24} "
                        f"{ep.comment}{acc_str}",
                        end="",
                        flush=True,
                    )

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as exc:
                logger.warning(f"Error on prompt {item['prompt'][:40]!r}: {exc}")
                done += 1
        else:
            continue
        break  # KeyboardInterrupt inner loop propagated

    print()  # newline after \r progress
    store.print_stats()


if __name__ == "__main__":
    main()
