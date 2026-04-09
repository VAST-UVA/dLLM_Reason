"""Benchmark all unmasking schedulers on a reasoning dataset.

Runs each scheduler on the same prompts and reports accuracy, latency, and
token-efficiency in a single comparison table.

Usage
-----
# Compare all 11 schedulers on GSM8K (100 samples each)
python scripts/benchmark_schedulers.py \\
    --checkpoint GSAI-ML/LLaDA-8B-Instruct \\
    --model llada \\
    --dataset gsm8k \\
    --n_samples 100 \\
    --output_dir results/scheduler_bench

# Compare a subset of schedulers
python scripts/benchmark_schedulers.py \\
    --checkpoint checkpoints/llada-instruct \\
    --schedulers confidence entropy cot adaptive_dynamic \\
    --n_samples 50

# Export results as CSV
python scripts/benchmark_schedulers.py \\
    --checkpoint checkpoints/llada-instruct \\
    --n_samples 200 \\
    --save_csv results/scheduler_bench/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

ALL_SCHEDULERS = [
    "confidence",
    "random",
    "entropy",
    "linear",
    "curriculum",
    "critical_token_first",
    "maskgit_cosine",
    "semi_ar",
    "adaptive_dynamic",
    "cot",
    "skeleton",
    "bidirectional",
]


def build_scheduler(name: str, seq_len: int, device):
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
    from dllm_reason.scheduler.random_scheduler import RandomScheduler
    from dllm_reason.scheduler.entropy_scheduler import EntropyScheduler
    from dllm_reason.scheduler.linear_scheduler import LinearScheduler
    from dllm_reason.scheduler.curriculum_scheduler import CurriculumScheduler
    from dllm_reason.scheduler.critical_token_scheduler import CriticalTokenFirstScheduler
    from dllm_reason.scheduler.maskgit_scheduler import MaskGITCosineScheduler
    from dllm_reason.scheduler.semi_ar_scheduler import SemiAutoregressiveScheduler
    from dllm_reason.scheduler.adaptive_dynamic_scheduler import AdaptiveDynamicScheduler
    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.graph.templates import build_template

    if name == "confidence":      return ConfidenceScheduler()
    if name == "random":          return RandomScheduler()
    if name == "entropy":         return EntropyScheduler()
    if name == "linear":          return LinearScheduler()
    if name == "curriculum":      return CurriculumScheduler()
    if name == "critical_token_first":
        return CriticalTokenFirstScheduler()
    if name == "maskgit_cosine":  return MaskGITCosineScheduler()
    if name == "semi_ar":         return SemiAutoregressiveScheduler()
    if name == "adaptive_dynamic":return AdaptiveDynamicScheduler()
    if name in ("cot", "skeleton", "bidirectional"):
        dag = build_template(name, seq_len, device=device)
        return DAGScheduler(dag, "confidence_topk")
    raise ValueError(f"Unknown scheduler: {name!r}")


def load_model(args):
    dtype_map = {"bfloat16": torch.bfloat16,
                 "float16":  torch.float16,
                 "float32":  torch.float32}
    dtype = dtype_map[args.torch_dtype]
    print(f"Loading {args.model} from {args.checkpoint!r} ...")
    if args.model == "llada":
        from dllm_reason.models.llada import LLaDAWrapper
        model = LLaDAWrapper(model_id=args.checkpoint, torch_dtype=dtype, device_map="auto")
    else:
        from dllm_reason.data.tokenizer import get_tokenizer
        from dllm_reason.utils.registry import MODEL_REGISTRY
        import dllm_reason.models.mdlm, dllm_reason.models.sedd, dllm_reason.models.d3pm
        tokenizer = get_tokenizer(args.checkpoint if args.model == "llada" else "gpt2",
                                  add_mask_token=True)
        model_cls = MODEL_REGISTRY.get(args.model)
        model = model_cls(vocab_size=len(tokenizer), max_seq_len=args.max_seq_len)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model ready on {model.device}")
    return model


def run_one_scheduler(model, scheduler_name, dataset, args, device) -> dict:
    """Evaluate a single scheduler; return metrics dict."""
    from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig
    from dllm_reason.eval.metrics import extract_number

    scheduler = build_scheduler(scheduler_name, args.gen_length, device)
    sampler   = DiffusionSampler(
        model, scheduler,
        SamplingConfig(
            num_steps=args.num_steps,
            temperature=args.temperature,
            show_progress=False,
        ),
    )
    tokenizer = model.tokenizer

    correct = 0
    total   = 0
    latency_sum = 0.0

    for sample in dataset[: args.n_samples]:
        prompt = sample.get("question") or sample.get("prompt", "")
        answer = sample.get("answer") or sample.get("ground_truth", "")

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_ids  = enc["input_ids"]
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)

        t0 = time.monotonic()
        result = sampler.sample(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            gen_length=args.gen_length,
        )
        latency_sum += time.monotonic() - t0

        gen_text = tokenizer.decode(
            result.sequences[0, prompt_ids.shape[1]:],
            skip_special_tokens=True,
        )

        pred = extract_number(gen_text)
        gt   = extract_number(str(answer))
        if pred is not None and gt is not None and pred == gt:
            correct += 1
        elif str(pred).strip() == str(gt).strip():
            correct += 1
        total += 1

    n = max(total, 1)
    return {
        "scheduler":  scheduler_name,
        "accuracy":   correct / n,
        "n_correct":  correct,
        "n_total":    total,
        "avg_latency_s": latency_sum / n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark all unmasking schedulers side-by-side",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="llada",
                   choices=["llada", "mdlm", "sedd", "d3pm"])
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "math", "arc", "prontoqa"])
    p.add_argument("--schedulers", nargs="+", default=ALL_SCHEDULERS,
                   help="Schedulers to compare (default: all)")
    p.add_argument("--n_samples",  type=int, default=100)
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--num_steps",  type=int, default=128)
    p.add_argument("--temperature",type=float, default=0.0)
    p.add_argument("--max_seq_len",type=int, default=512)
    p.add_argument("--output_dir", default="results/scheduler_bench")
    p.add_argument("--save_csv",   default=None,
                   help="Also save a CSV file at this path")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args)

    from dllm_reason.data.reasoning_datasets import load_reasoning_dataset
    dataset = load_reasoning_dataset(args.dataset, split="test")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for name in args.schedulers:
        print(f"\n  [{name}] running {args.n_samples} samples ...", end="", flush=True)
        try:
            m = run_one_scheduler(model, name, dataset, args, device)
            results.append(m)
            print(f"  acc={m['accuracy']:.3f}  lat={m['avg_latency_s']:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"scheduler": name, "accuracy": float("nan"),
                            "n_correct": 0, "n_total": 0, "avg_latency_s": 0.0})

    # Print comparison table
    results.sort(key=lambda r: r["accuracy"] if r["accuracy"] == r["accuracy"] else -1,
                 reverse=True)
    w = 68
    print(f"\n{'═'*w}")
    print(f"  Scheduler Benchmark  —  {args.dataset}  ({args.n_samples} samples)")
    print(f"{'═'*w}")
    print(f"  {'RANK':<5}  {'SCHEDULER':<25}  {'ACCURACY':>10}  "
          f"{'CORRECT':>8}  {'LAT(s)':>8}")
    print(f"  {'─'*5}  {'─'*25}  {'─'*10}  {'─'*8}  {'─'*8}")
    for i, r in enumerate(results, 1):
        acc_str = f"{r['accuracy']:.3f}" if r['accuracy'] == r['accuracy'] else "ERROR"
        print(f"  {i:<5}  {r['scheduler']:<25}  {acc_str:>10}  "
              f"{r['n_correct']:>8}  {r['avg_latency_s']:>8.2f}")
    print(f"{'═'*w}\n")

    # Save JSON
    json_path = out_dir / f"benchmark_{args.dataset}.json"
    with open(json_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Results saved → {json_path}")

    # Save CSV
    csv_path = Path(args.save_csv) if args.save_csv else out_dir / f"benchmark_{args.dataset}.csv"
    if args.save_csv or True:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV saved      → {csv_path}")


if __name__ == "__main__":
    main()
