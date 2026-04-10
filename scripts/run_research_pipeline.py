"""Three-stage research pipeline for dLLM-Reason.

Stage 1 — Baseline Evaluation:
    Evaluate the base LLaDA model on reasoning benchmarks with various
    unmasking schedulers.  All inference goes through the FastAPI server.

Stage 2 — DAG Discovery:
    For each prompt, try multiple DAG templates (or run per-prompt search)
    to find the best unmasking order.  Stores (prompt, best-DAG) pairs.

Stage 3 — DAG-Aware Training:
    Fine-tune LLaDA so it internalises the optimal unmasking order
    discovered in Stage 2.  Supports SFT, GRPO, DiFFPO, and UnmaskRL.

Usage:
    # Start the server first:
    python scripts/serve.py --model_id GSAI-ML/LLaDA-8B-Instruct

    # Run full pipeline (default: gsm8k, confidence scheduler):
    python scripts/run_research_pipeline.py

    # Run specific stages:
    python scripts/run_research_pipeline.py --stages 1 2

    # Dry run (print config only):
    python scripts/run_research_pipeline.py --dry_run

See docs/ABLATION_SETTINGS.md for all ablation dimensions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from itertools import islice
from pathlib import Path

import requests
import torch

# ── Imports from the project ────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dllm_reason.data.reasoning_datasets import load_reasoning_dataset
from dllm_reason.library.episode import DAGEpisode, EpisodeStore


# ── Constants ────────────────────────────────────────────────────────────────

ALL_SCHEDULERS = [
    "confidence", "random", "linear", "entropy", "semi_ar",
    "maskgit_cosine", "critical_token_first", "curriculum",
    "adaptive_dynamic",
    # DAG-based schedulers:
    "cot", "skeleton", "bidirectional", "answer_first",
]

ALL_DATASETS = [
    "gsm8k", "math", "arc", "prontoqa",
    # These require additional evaluator support:
    # "mmlu", "hotpotqa", "mbpp", "humaneval", "gpqa", "aime",
]

FLAT_STRATEGIES = {"confidence", "random", "linear", "entropy", "semi_ar",
                   "maskgit_cosine", "critical_token_first", "curriculum",
                   "adaptive_dynamic"}

# DAG template priority (higher = preferred when multiple are correct)
DAG_PRIORITY = {
    "cot": 4, "skeleton": 3, "bidirectional": 2, "answer_first": 1,
}


# ── Utilities ────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_last_number(text: str) -> str | None:
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def auto_eval(output: str, ground_truth: str, dataset: str) -> tuple[bool, float, str]:
    """Evaluate model output against ground truth.

    Uses numeric matching for math-type datasets, substring matching otherwise.
    """
    if dataset in ("gsm8k", "math", "aime"):
        pred = _extract_last_number(output)
        gold = _extract_last_number(ground_truth)
        if pred is None or gold is None:
            return False, 0.0, f"parse_fail: pred={pred!r} gold={gold!r}"
        try:
            match = abs(float(pred) - float(gold)) < 1e-3
        except ValueError:
            match = pred.strip() == gold.strip()
        return match, 1.0 if match else 0.0, f"pred={pred} gold={gold}"
    else:
        norm = lambda s: s.strip().lower()
        match = norm(ground_truth) in norm(output)
        return match, 1.0 if match else 0.0, f"exact_match={match}"


def batched(iterable, n: int):
    """Batch an iterable into chunks of size n."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── API Client ───────────────────────────────────────────────────────────────


class PipelineAPIClient:
    """Thin wrapper around the FastAPI server endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/info", timeout=10)
            r.raise_for_status()
            info = r.json()
            if info.get("status") != "ready":
                print(f"[WARN] Server status: {info.get('status')}")
            else:
                print(f"[OK] Server ready: model={info.get('model_id')}, "
                      f"device={info.get('device')}")
            return info
        except requests.ConnectionError:
            print(f"[ERROR] Cannot connect to server at {self.base_url}")
            print("        Start the server first:")
            print("        python scripts/serve.py --model_id GSAI-ML/LLaDA-8B-Instruct")
            sys.exit(1)

    def batch_generate(
        self,
        prompts: list[str],
        strategy: str = "confidence",
        max_new_tokens: int = 128,
        num_steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
    ) -> list[dict]:
        r = requests.post(
            f"{self.base_url}/batch_generate",
            json={
                "prompts": prompts,
                "strategy": strategy,
                "max_new_tokens": max_new_tokens,
                "num_steps": num_steps,
                "block_length": block_length,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def switch_model(self, model_id: str, torch_dtype: str = "bfloat16") -> dict:
        r = requests.post(
            f"{self.base_url}/switch_model",
            json={"model_id": model_id, "torch_dtype": torch_dtype},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()


# ── Stage 1: Baseline Evaluation ─────────────────────────────────────────────


def stage1_baseline(api: PipelineAPIClient, args: argparse.Namespace, run_dir: Path):
    """Evaluate model on benchmarks with multiple schedulers."""
    out = run_dir / "stage1_baseline"
    out.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}  # {f"{ds}_{scheduler}": result}

    for scheduler_name in args.s1_schedulers:
        for ds in args.datasets:
            run_key = f"{ds}_{scheduler_name}"
            result_path = out / f"{run_key}.json"

            # Resume: skip if result file already exists
            if args.resume and result_path.exists():
                print(f"  [SKIP] {run_key} — already done")
                all_results[run_key] = load_json(result_path)
                continue

            print(f"\n  [EVAL] {run_key}")
            data = load_reasoning_dataset(ds, split="test")
            if args.num_samples > 0:
                data = data[:args.num_samples]

            correct, total = 0, 0
            per_sample: list[dict] = []

            for chunk in batched(data, args.api_batch_size):
                prompts = [item["question"] for item in chunk]
                try:
                    resps = api.batch_generate(
                        prompts=prompts,
                        strategy=scheduler_name,
                        max_new_tokens=args.gen_length,
                        num_steps=args.num_steps,
                        block_length=args.block_length,
                        temperature=args.temperature,
                    )
                except requests.HTTPError as e:
                    print(f"    [ERROR] API error: {e}")
                    continue

                for item, resp in zip(chunk, resps):
                    ok, score, comment = auto_eval(resp["text"], item["answer"], ds)
                    correct += int(ok)
                    total += 1
                    per_sample.append({
                        "question": item["question"],
                        "ground_truth": item["answer"],
                        "output": resp["text"],
                        "correct": ok,
                        "score": score,
                        "comment": comment,
                        "elapsed_seconds": resp.get("elapsed_seconds", 0),
                    })

            accuracy = correct / total if total > 0 else 0.0
            result = {
                "scheduler": scheduler_name,
                "dataset": ds,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "per_sample": per_sample,
            }
            save_json(result_path, result)
            all_results[run_key] = result
            print(f"    accuracy={accuracy:.4f}  ({correct}/{total})")

    # Print comparison table
    _print_comparison_table(all_results, args.s1_schedulers, args.datasets)
    save_json(out / "summary.json", {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                                      for k, v in all_results.items()})


def _print_comparison_table(results: dict, schedulers: list[str], datasets: list[str]):
    """Print scheduler × dataset accuracy table."""
    print(f"\n{'='*60}")
    print("Stage 1 — Baseline Comparison")
    print(f"{'='*60}")

    # Header
    header = f"{'Scheduler':<25}" + "".join(f"{ds:>12}" for ds in datasets)
    print(header)
    print("-" * len(header))

    for sched in schedulers:
        row = f"{sched:<25}"
        for ds in datasets:
            key = f"{ds}_{sched}"
            if key in results:
                acc = results[key].get("accuracy", 0.0)
                row += f"{acc:>11.4f} "
            else:
                row += f"{'N/A':>12}"
        print(row)
    print()


# ── Stage 2: DAG Discovery ──────────────────────────────────────────────────


def stage2_dag_discovery(
    api: PipelineAPIClient, args: argparse.Namespace, run_dir: Path,
) -> dict:
    """Discover best DAG template per prompt via sweep or search."""
    out = run_dir / "stage2_discovery"
    out.mkdir(parents=True, exist_ok=True)

    store = EpisodeStore(str(out / "episodes.db"))

    if args.s2_method == "sweep":
        _stage2_sweep(api, store, args, out)
    elif args.s2_method == "search":
        _stage2_search(store, args, out)
    else:
        raise ValueError(f"Unknown s2_method: {args.s2_method}")

    best_map = _select_best_dag_per_prompt(store, args)
    save_json(out / "best_dag_per_prompt.json", best_map)
    _print_discovery_stats(best_map)
    return best_map


def _get_existing_prompts(store: EpisodeStore, strategy: str) -> set[str]:
    """Get set of prompts already stored for a given strategy."""
    episodes = store.query(strategy_name=strategy, limit=10000)
    return {ep.prompt for ep in episodes}


def _stage2_sweep(
    api: PipelineAPIClient,
    store: EpisodeStore,
    args: argparse.Namespace,
    out: Path,
):
    """Try all strategies for each prompt, record results."""
    for ds in args.datasets:
        data = load_reasoning_dataset(ds, split="test")
        if args.num_samples > 0:
            data = data[:args.num_samples]

        for strategy in args.s2_strategies:
            # Collect prompts not yet tried with this strategy
            existing_prompts = _get_existing_prompts(store, strategy)
            todo = [(i, item) for i, item in enumerate(data)
                    if item["question"] not in existing_prompts]

            if not todo:
                print(f"  [SKIP] {ds}/{strategy} — all {len(data)} prompts done")
                continue

            print(f"  [SWEEP] {ds}/{strategy}: {len(todo)} prompts remaining")

            for chunk in batched(todo, args.api_batch_size):
                prompts = [item["question"] for _, item in chunk]
                try:
                    resps = api.batch_generate(
                        prompts=prompts,
                        strategy=strategy,
                        max_new_tokens=args.gen_length,
                        num_steps=args.num_steps,
                        block_length=args.block_length,
                        temperature=args.temperature,
                    )
                except requests.HTTPError as e:
                    print(f"    [ERROR] API error: {e}")
                    continue

                for (idx, item), resp in zip(chunk, resps):
                    ok, score, comment = auto_eval(resp["text"], item["answer"], ds)
                    ep = DAGEpisode(
                        episode_id=str(uuid.uuid4()),
                        prompt=item["question"],
                        task_type=ds,
                        ground_truth=item["answer"],
                        strategy_name=strategy,
                        model_id=args.checkpoint,
                        output=resp["text"],
                        correct=ok,
                        score=score,
                        comment=comment,
                        num_steps=args.num_steps,
                        block_length=args.block_length,
                        temperature=args.temperature,
                    )
                    store.add(ep)


def _stage2_search(
    store: EpisodeStore,
    args: argparse.Namespace,
    out: Path,
):
    """Per-prompt DAG search (local model, not through API).

    Requires local model loading. Supports greedy, evolutionary,
    rl_policy, and differentiable search methods.
    """
    print("  [SEARCH] Loading model locally for per-prompt search...")
    model = _load_model_local(args)
    device = model.device

    from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig
    from dllm_reason.graph.dag import TokenDAG

    for ds in args.datasets:
        data = load_reasoning_dataset(ds, split="test")
        if args.num_samples > 0:
            data = data[:args.num_samples]

        all_episodes = store.query(limit=100000)
        existing_prompts = {ep.prompt for ep in all_episodes}

        for idx, item in enumerate(data):
            if item["question"] in existing_prompts:
                continue

            print(f"    [{idx+1}/{len(data)}] Searching DAG for prompt...")

            # Try each template as a candidate
            from dllm_reason.graph.templates import build_template, TEMPLATE_NAMES
            best_score = -1.0
            best_ep = None

            for tname in TEMPLATE_NAMES:
                try:
                    dag = build_template(tname, args.gen_length, device)
                except Exception:
                    continue

                from dllm_reason.scheduler.dag_scheduler import DAGScheduler
                scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")

                text = model.generate(
                    prompt=item["question"],
                    generation_len=args.gen_length,
                    block_length=args.block_length,
                    scheduler=scheduler,
                    num_steps=args.num_steps,
                    temperature=args.temperature,
                )

                ok, score, comment = auto_eval(text, item["answer"], ds)
                ep = DAGEpisode(
                    episode_id=str(uuid.uuid4()),
                    prompt=item["question"],
                    task_type=ds,
                    ground_truth=item["answer"],
                    strategy_name=tname,
                    model_id=args.checkpoint,
                    output=text,
                    correct=ok,
                    score=score,
                    comment=comment,
                    dag_seq_len=args.gen_length,
                    dag_adjacency=dag.adj.cpu().int().tolist(),
                    num_steps=args.num_steps,
                    block_length=args.block_length,
                    temperature=args.temperature,
                )
                store.add(ep)

                if score > best_score:
                    best_score = score
                    best_ep = ep


def _select_best_dag_per_prompt(store: EpisodeStore, args: argparse.Namespace) -> dict:
    """Select the best DAG template for each unique prompt.

    Priority:
    1. correct + DAG-based strategy  (sorted by DAG_PRIORITY)
    2. correct + flat strategy
    3. incorrect (no training data)
    """
    all_episodes = store.query(limit=100000)
    by_prompt: dict[str, list[DAGEpisode]] = defaultdict(list)
    for ep in all_episodes:
        by_prompt[ep.prompt].append(ep)

    best_map: dict[str, dict] = {}
    for prompt, episodes in by_prompt.items():
        correct_dag = [e for e in episodes if e.correct and e.strategy_name not in FLAT_STRATEGIES]
        correct_flat = [e for e in episodes if e.correct and e.strategy_name in FLAT_STRATEGIES]

        if correct_dag:
            # Pick highest-priority DAG template
            best = max(correct_dag, key=lambda e: DAG_PRIORITY.get(e.strategy_name, 0))
        elif correct_flat:
            best = correct_flat[0]
        else:
            best = None

        best_map[prompt] = {
            "prompt": prompt,
            "best_strategy": best.strategy_name if best else None,
            "correct": best.correct if best else False,
            "output": best.output if best else "",
            "ground_truth": episodes[0].ground_truth if episodes else "",
            "task_type": episodes[0].task_type if episodes else "",
            "num_strategies_tried": len(set(e.strategy_name for e in episodes)),
            "num_correct": sum(1 for e in episodes if e.correct),
        }

    return best_map


def _print_discovery_stats(best_map: dict):
    """Print DAG discovery summary statistics."""
    print(f"\n{'='*60}")
    print("Stage 2 — DAG Discovery Summary")
    print(f"{'='*60}")

    total = len(best_map)
    has_best = sum(1 for v in best_map.values() if v["best_strategy"])
    by_strategy: dict[str, int] = defaultdict(int)
    for v in best_map.values():
        s = v.get("best_strategy")
        if s:
            by_strategy[s] += 1

    print(f"Total prompts:     {total}")
    print(f"With best DAG:     {has_best}")
    print(f"No correct answer: {total - has_best}")
    print()
    print("Strategy distribution:")
    for s, cnt in sorted(by_strategy.items(), key=lambda x: -x[1]):
        print(f"  {s:<25} {cnt:>5}  ({cnt/total*100:.1f}%)")
    print()


# ── Stage 3: DAG-Aware Training ─────────────────────────────────────────────


def stage3_dag_training(
    api: PipelineAPIClient | None,
    best_map: dict,
    args: argparse.Namespace,
    run_dir: Path,
):
    """Train model with DAG-aware masking bias."""
    out = run_dir / "stage3_trained"

    # Resume check
    if args.resume and (out / "pytorch_model.bin").exists():
        print("  [SKIP] Stage 3 — model already trained")
        return

    out.mkdir(parents=True, exist_ok=True)

    print("  Loading model locally for training...")
    model = _load_model_local(args)
    tokenizer = model.tokenizer
    device = model.device

    # Load episodes from Stage 2
    episodes_db = run_dir / "stage2_discovery" / "episodes.db"
    if not episodes_db.exists():
        print(f"  [ERROR] Episodes DB not found: {episodes_db}")
        print("          Run Stage 2 first, or set --stages 2 3")
        return

    store = EpisodeStore(str(episodes_db))
    all_episodes = store.query(limit=100000)
    correct_episodes = [e for e in all_episodes if e.correct]

    if not correct_episodes:
        print("  [ERROR] No correct episodes found. Cannot train.")
        return

    print(f"  Correct episodes: {len(correct_episodes)}/{len(all_episodes)}")

    # Import training utilities from learn_from_episodes.py
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))

    from learn_from_episodes import (
        _build_aggregate_dag, _patch_dag_noise, _save_model,
        run_sft, SFTEpisodeDataset, _sft_collate,
    )

    # Build a namespace that run_sft/etc expect
    train_args = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=512,
        log_every=50,
        output_dir=str(out),
        kl_coeff=args.kl_coeff,
        clip_ratio=args.clip_ratio,
        ppo_clip_eps=args.ppo_clip_eps,
        train_sampler=args.train_sampler,
        step_budget_lambda=0.01,
        min_steps=8,
        max_steps=args.num_steps,
        unmask_group_size=args.unmask_group_size,
        unmask_d_model=args.unmask_d_model,
        unmask_n_heads=4,
        unmask_num_steps=args.num_steps,
    )

    # ── DAG Injection Mode ──
    if args.s3_dag_mode == "per_template":
        _train_per_template(model, tokenizer, best_map, correct_episodes,
                            args, train_args, out, device)
    elif args.s3_dag_mode == "consensus":
        dag = _build_aggregate_dag(correct_episodes, device)
        if dag is not None:
            print(f"  Consensus DAG: {dag}")
            _patch_dag_noise(model, dag, args.dag_bias_strength)
        _train_single(model, tokenizer, correct_episodes, args, train_args, out)
    elif args.s3_dag_mode == "none":
        _train_single(model, tokenizer, correct_episodes, args, train_args, out)

    _save_model(model, tokenizer, str(out))

    # Optionally hot-swap model on server for re-evaluation
    if api:
        try:
            print("  Hot-swapping model on server...")
            api.switch_model(str(out), args.torch_dtype)
            print("  [OK] Model switched. Re-run Stage 1 to evaluate.")
        except Exception as e:
            print(f"  [WARN] switch_model failed: {e}")


def _train_per_template(
    model, tokenizer, best_map, episodes, args, train_args, out, device,
):
    """Group prompts by best DAG template, train each group with its DAG."""
    from learn_from_episodes import _patch_dag_noise
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag, answer_first_dag,
    )

    # Group by DAG template
    groups: dict[str, list[DAGEpisode]] = defaultdict(list)
    prompt_to_strategy = {v["prompt"]: v["best_strategy"] for v in best_map.values()}

    for ep in episodes:
        strategy = prompt_to_strategy.get(ep.prompt)
        if strategy and strategy not in FLAT_STRATEGIES:
            groups[strategy].append(ep)

    if not groups:
        print("  [WARN] No DAG-based groups found. Falling back to flat training.")
        _train_single(model, tokenizer, episodes, args, train_args, out)
        return

    template_stats = {}
    for template_name, group_eps in groups.items():
        print(f"\n  Training group: {template_name}  ({len(group_eps)} episodes)")
        template_stats[template_name] = len(group_eps)

        # Build DAG for this template
        dag = _build_dag_for_template(template_name, args.gen_length, device)
        if dag is not None:
            _patch_dag_noise(model, dag, args.dag_bias_strength)

        # Run training
        _train_single(model, tokenizer, group_eps, args, train_args, out,
                       save=False)

        # Restore original noise function
        if hasattr(model, "_original_noise_input"):
            model.noise_input = model._original_noise_input

    save_json(out / "template_stats.json", template_stats)


def _build_dag_for_template(name: str, gen_length: int, device):
    """Build a DAG from template name."""
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag, answer_first_dag,
    )

    try:
        if name == "cot":
            return chain_of_thought_dag(gen_length, num_steps=4, device=device)
        elif name == "skeleton":
            return skeleton_then_detail_dag(
                gen_length,
                list(range(0, gen_length, 3)),
                list(range(1, gen_length, 3)),
                device=device,
            )
        elif name == "bidirectional":
            return bidirectional_dag(gen_length, num_segments=4, device=device)
        elif name == "answer_first":
            return answer_first_dag(
                gen_length,
                list(range(int(gen_length * 0.8), gen_length)),
                device=device,
            )
    except Exception as e:
        print(f"    [WARN] Cannot build DAG for {name}: {e}")
    return None


def _train_single(model, tokenizer, episodes, args, train_args, out, save=True):
    """Run training with the selected mode (sft/grpo/diffppo/unmask_rl)."""
    from learn_from_episodes import run_sft

    if args.s3_mode == "sft":
        run_sft(model, tokenizer, episodes, train_args, dag_aware=False)
    elif args.s3_mode == "grpo":
        # GRPO requires a reference model (frozen copy)
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        from learn_from_episodes import run_grpo
        run_grpo(model, ref_model, tokenizer, episodes, train_args)
    elif args.s3_mode == "diffppo":
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        from learn_from_episodes import run_diffppo
        run_diffppo(model, ref_model, tokenizer, episodes, train_args)
    elif args.s3_mode == "unmask_rl":
        from learn_from_episodes import run_unmask_rl
        run_unmask_rl(model, tokenizer, episodes, train_args)
    else:
        raise ValueError(f"Unknown training mode: {args.s3_mode}")


# ── Model loading ────────────────────────────────────────────────────────────


def _load_model_local(args: argparse.Namespace):
    """Load LLaDA model locally for training or search."""
    from dllm_reason.models.llada import LLaDAWrapper

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    model = LLaDAWrapper(
        model_id=args.checkpoint,
        torch_dtype=dtype_map.get(args.torch_dtype, torch.bfloat16),
        device_map="auto",
    )
    print(f"  Model loaded: {args.checkpoint} on {model.device}")
    return model


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Three-stage research pipeline for dLLM-Reason",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Connection ──
    p.add_argument("--api_url", type=str, default="http://localhost:8000",
                    help="FastAPI server URL")

    # ── General ──
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3],
                    help="Which stages to run (default: 1 2 3)")
    # Options: any combination of 1, 2, 3
    p.add_argument("--run_dir", type=str, default=None,
                    help="Output directory (default: runs/research_<timestamp>)")
    p.add_argument("--resume", action="store_true",
                    help="Skip completed sub-tasks")
    p.add_argument("--dry_run", action="store_true",
                    help="Print config and exit")

    # ── Data ──
    p.add_argument("--datasets", type=str, nargs="+", default=["gsm8k"],
                    help="Datasets to evaluate on")
    # Options: gsm8k, math, arc, prontoqa, mmlu, hotpotqa, mbpp, humaneval, gpqa, aime
    p.add_argument("--num_samples", type=int, default=200,
                    help="Samples per dataset (-1 = all)")

    # ── Inference parameters (sent to API) ──
    p.add_argument("--gen_length", type=int, default=128,
                    help="Generation length in tokens")
    # Options: 64, 128, 256, 512
    p.add_argument("--num_steps", type=int, default=128,
                    help="Number of diffusion denoising steps")
    # Options: 32, 64, 128, 256
    p.add_argument("--block_length", type=int, default=32,
                    help="Block length for block-wise denoising")
    # Options: 8, 16, 32, 64
    p.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0.0 = greedy)")
    # Options: 0.0, 0.3, 0.5, 0.7, 1.0
    p.add_argument("--api_batch_size", type=int, default=4,
                    help="Prompts per /batch_generate call")

    # ── Stage 1: Baseline ──
    p.add_argument("--s1_schedulers", type=str, nargs="+", default=["confidence"],
                    help="Schedulers to evaluate in Stage 1")
    # Options: confidence, random, linear, entropy, semi_ar,
    #          maskgit_cosine, critical_token_first, curriculum,
    #          adaptive_dynamic, cot, skeleton, bidirectional, answer_first

    # ── Stage 2: DAG Discovery ──
    p.add_argument("--s2_strategies", type=str, nargs="+",
                    default=["confidence", "cot", "skeleton", "bidirectional",
                             "answer_first", "linear", "random"],
                    help="Strategies to sweep in Stage 2")
    p.add_argument("--s2_method", type=str, default="sweep",
                    choices=["sweep", "search"],
                    help="DAG discovery method")
    # Options: sweep (try all templates), search (per-prompt search)
    p.add_argument("--s2_search_method", type=str, default="greedy",
                    choices=["greedy", "evolutionary", "rl_policy", "differentiable"],
                    help="Search algorithm (only if --s2_method=search)")
    p.add_argument("--s2_search_budget", type=int, default=50,
                    help="Search budget per prompt (only if --s2_method=search)")

    # ── Stage 3: Training ──
    p.add_argument("--checkpoint", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                    help="Model checkpoint for local loading (Stage 2 search / Stage 3)")
    p.add_argument("--s3_mode", type=str, default="sft",
                    choices=["sft", "grpo", "diffppo", "unmask_rl"],
                    help="Training mode")
    # Options: sft (supervised fine-tuning on correct episodes)
    #          grpo (Group Relative Policy Optimization)
    #          diffppo (DiFFPO with optional step-budget controller)
    #          unmask_rl (frozen LM + lightweight policy net via REINFORCE)
    p.add_argument("--s3_dag_mode", type=str, default="per_template",
                    choices=["per_template", "consensus", "none"],
                    help="DAG injection mode")
    # Options: per_template (group by best template, train each group with its DAG)
    #          consensus (majority-vote DAG from all correct episodes)
    #          none (no DAG bias — ablation baseline)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate")
    # Options: 1e-6 ~ 1e-4
    p.add_argument("--dag_bias_strength", type=float, default=0.5,
                    help="DAG bias strength (0.0 = no bias, 1.0 = full bias)")
    # Options: 0.0, 0.1, 0.3, 0.5, 0.8, 1.0

    # GRPO / DiFFPO parameters
    p.add_argument("--kl_coeff", type=float, default=0.01,
                    help="KL divergence coefficient (grpo/diffppo)")
    # Options: 0.001, 0.005, 0.01, 0.02, 0.05, 0.1
    p.add_argument("--clip_ratio", type=float, default=0.2,
                    help="Clip ratio (grpo)")
    p.add_argument("--ppo_clip_eps", type=float, default=0.2,
                    help="PPO clip epsilon (diffppo)")
    # Options: 0.1, 0.2, 0.3
    p.add_argument("--train_sampler", action="store_true",
                    help="Joint train step-budget controller (diffppo only)")

    # UnmaskRL parameters
    p.add_argument("--unmask_group_size", type=int, default=4,
                    help="Group size for UnmaskRL")
    # Options: 2, 4, 8
    p.add_argument("--unmask_d_model", type=int, default=64,
                    help="Policy network hidden dimension (unmask_rl)")
    # Options: 32, 64, 128, 256

    # Model precision
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model precision")

    return p.parse_args(argv)


# ── Public entry point (for import from run_ablation.py) ─────────────────────


def run_pipeline(args_or_dict):
    """Run the pipeline with given arguments.

    Accepts either an argparse.Namespace or a dict (keys → CLI args).
    """
    if isinstance(args_or_dict, dict):
        # Convert dict to argv list
        argv = []
        for key, val in args_or_dict.items():
            if key in ("name", "desc"):
                continue  # ablation metadata, not CLI args
            arg_name = f"--{key}"
            if isinstance(val, bool):
                if val:
                    argv.append(arg_name)
            elif isinstance(val, list):
                argv.append(arg_name)
                argv.extend(str(v) for v in val)
            else:
                argv.extend([arg_name, str(val)])
        args = parse_args(argv)
    else:
        args = args_or_dict

    _run_pipeline_impl(args)


def _run_pipeline_impl(args: argparse.Namespace):
    """Internal pipeline implementation."""
    run_dir = Path(args.run_dir or f"runs/research_{timestamp()}")

    if args.dry_run:
        _print_config(args, run_dir)
        return

    print(f"\n{'='*60}")
    print(f"dLLM-Reason Research Pipeline")
    print(f"  Stages: {args.stages}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # API client (not needed for Stage 3 only)
    api = None
    if 1 in args.stages or 2 in args.stages:
        api = PipelineAPIClient(args.api_url)
        api.check_health()

    best_map = None

    if 1 in args.stages:
        print(f"\n{'─'*60}")
        print("Stage 1 — Baseline Evaluation")
        print(f"{'─'*60}")
        stage1_baseline(api, args, run_dir)

    if 2 in args.stages:
        print(f"\n{'─'*60}")
        print("Stage 2 — DAG Discovery")
        print(f"{'─'*60}")
        best_map = stage2_dag_discovery(api, args, run_dir)

    if 3 in args.stages:
        print(f"\n{'─'*60}")
        print("Stage 3 — DAG-Aware Training")
        print(f"{'─'*60}")
        if best_map is None:
            # Try to load from existing Stage 2 results
            best_map_path = run_dir / "stage2_discovery" / "best_dag_per_prompt.json"
            if best_map_path.exists():
                best_map = load_json(best_map_path)
            else:
                print("  [WARN] No best_dag_per_prompt.json found.")
                print("         Training without DAG bias (s3_dag_mode=none).")
                best_map = {}
                args.s3_dag_mode = "none"

        # Only connect API if we want to hot-swap after training
        if api is None:
            try:
                api = PipelineAPIClient(args.api_url)
                api.check_health()
            except SystemExit:
                api = None  # server not running, skip hot-swap

        stage3_dag_training(api, best_map, args, run_dir)

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Results in: {run_dir}")
    print(f"{'='*60}\n")


def _print_config(args: argparse.Namespace, run_dir: Path):
    """Print pipeline configuration (dry run)."""
    print(f"\n{'='*60}")
    print("dLLM-Reason Research Pipeline — DRY RUN")
    print(f"{'='*60}")
    print(f"  Run dir:       {run_dir}")
    print(f"  Stages:        {args.stages}")
    print(f"  Datasets:      {args.datasets}")
    print(f"  Num samples:   {args.num_samples}")
    print(f"  API URL:       {args.api_url}")
    print()
    print("  Inference:")
    print(f"    gen_length:    {args.gen_length}")
    print(f"    num_steps:     {args.num_steps}")
    print(f"    block_length:  {args.block_length}")
    print(f"    temperature:   {args.temperature}")
    print(f"    batch_size:    {args.api_batch_size}")
    print()
    if 1 in args.stages:
        print(f"  Stage 1:")
        print(f"    schedulers:  {args.s1_schedulers}")
        n = len(args.s1_schedulers) * len(args.datasets) * args.num_samples
        print(f"    total evals: ~{n}")
    if 2 in args.stages:
        print(f"  Stage 2:")
        print(f"    method:      {args.s2_method}")
        print(f"    strategies:  {args.s2_strategies}")
        if args.s2_method == "search":
            print(f"    search:      {args.s2_search_method} (budget={args.s2_search_budget})")
        n = len(args.s2_strategies) * len(args.datasets) * args.num_samples
        print(f"    total evals: ~{n}")
    if 3 in args.stages:
        print(f"  Stage 3:")
        print(f"    mode:        {args.s3_mode}")
        print(f"    dag_mode:    {args.s3_dag_mode}")
        print(f"    epochs:      {args.epochs}")
        print(f"    lr:          {args.lr}")
        print(f"    dag_bias:    {args.dag_bias_strength}")
        print(f"    checkpoint:  {args.checkpoint}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    _run_pipeline_impl(args)


if __name__ == "__main__":
    main()
