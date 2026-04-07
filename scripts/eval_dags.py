"""Main evaluation script: LLaDA + multiple DAG strategies on reasoning benchmarks.

Usage:
    python scripts/eval_dags.py \
        --benchmarks mbpp humaneval hotpotqa mmlu \
        --dags empty linear cot skeleton bidirectional \
        --model_id GSAI-ML/LLaDA-8B-Instruct \
        --output_dir results/ \
        --num_steps 128 \
        --num_samples 100

Results are saved as JSON files in output_dir/.
A summary table is printed at the end.
"""

import argparse
import json
import time
from pathlib import Path

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    """Load a YAML config file and return a flat dict of all values."""
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Flatten nested sections into a single dict
    flat = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def parse_args():
    # ── Pre-parse to get --config path before building the full parser ────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args()

    # Load config file defaults (if provided)
    cfg_defaults: dict = {}
    if pre_args.config:
        cfg_defaults = _load_config(pre_args.config)
        print(f"[config] Loaded: {pre_args.config}")
    elif Path("configs/eval_default.yaml").exists():
        cfg_defaults = _load_config("configs/eval_default.yaml")
        print("[config] Loaded: configs/eval_default.yaml (auto-detected)")

    # ── Build main parser — defaults come from config, CLI overrides them ─────
    parser = argparse.ArgumentParser(description="Evaluate LLaDA with DAG unmasking strategies")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (defaults to configs/eval_default.yaml)")

    def D(key, fallback):
        """Return config value if present, else fallback."""
        return cfg_defaults.get(key, fallback)

    # Model
    parser.add_argument("--model_id", type=str, default=D("model_id", "GSAI-ML/LLaDA-8B-Instruct"))
    parser.add_argument("--torch_dtype", type=str, default=D("torch_dtype", "bfloat16"),
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default=D("device_map", "auto"))

    # Benchmarks
    parser.add_argument("--benchmarks", nargs="+",
                        default=D("benchmarks", ["mbpp", "humaneval"]),
                        choices=["mbpp", "humaneval", "hotpotqa", "mmlu"])
    parser.add_argument("--num_samples", type=int, default=D("num_samples", None))

    # DAG strategies
    parser.add_argument("--dags", nargs="+",
                        default=D("dags", ["confidence"]),
                        choices=["empty", "random", "linear", "cot", "bidirectional",
                                 "confidence", "skeleton", "answer_first"])

    # Inference params
    parser.add_argument("--num_steps",    type=int,   default=D("num_steps", 128))
    parser.add_argument("--block_length", type=int,   default=D("block_length", 32))
    parser.add_argument("--temperature",  type=float, default=D("temperature", 0.0))
    parser.add_argument("--cfg_scale",    type=float, default=D("cfg_scale", 0.0))
    parser.add_argument("--remasking",    type=str,   default=D("remasking", "low_confidence"),
                        choices=["low_confidence", "random"])
    parser.add_argument("--max_new_tokens", type=int, default=D("max_new_tokens", 128))
    parser.add_argument("--generation_len", type=int, default=D("max_new_tokens", 128),
                        help="Alias for max_new_tokens")

    # CoT / MMLU
    parser.add_argument("--cot_steps",     type=int,  default=D("cot_steps", 4))
    parser.add_argument("--mmlu_subjects", nargs="+", default=D("mmlu_subjects", None))

    # Output / control
    parser.add_argument("--output_dir",   type=str,  default=D("output_dir", "results"))
    parser.add_argument("--resume",       action="store_true", default=D("resume", False))
    parser.add_argument("--no_run_tests", action="store_true",
                        default=not D("run_tests", True))
    parser.add_argument("--verbose_errors", action="store_true",
                        default=D("verbose_errors", False),
                        help="Print per-sample stderr/error/timeout on failure")

    # ── Detailed output saving ─────────────────────────────────────────────────
    parser.add_argument("--save_outputs", action="store_true",
                        default=D("save_outputs", False),
                        help="Save per-sample QA pairs / ground truth to JSON + Excel")
    parser.add_argument("--no_save_qa", action="store_true",
                        default=not D("save_qa", True),
                        help="Exclude prompt+generated output from saved files")
    parser.add_argument("--no_save_ground_truth", action="store_true",
                        default=not D("save_ground_truth", True),
                        help="Exclude reference answers from saved files")
    parser.add_argument("--record_trajectory", action="store_true",
                        default=D("record_trajectory", False),
                        help="Record per-step unmasking states (slow; saved to *_trajectory.json)")
    parser.add_argument("--output_formats", nargs="+",
                        default=D("output_formats", ["json", "xlsx"]),
                        choices=["json", "xlsx"],
                        help="File formats to write when --save_outputs is on")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# DAG factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dag_scheduler(dag_name: str, seq_len: int, args, device: torch.device):
    """Build a scheduler from a DAG name."""
    from dllm_reason.graph.dag import TokenDAG
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag,
    )
    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
    from dllm_reason.scheduler.random_scheduler import RandomScheduler
    from dllm_reason.scheduler.linear_scheduler import LinearScheduler

    if dag_name in ("empty", "random"):
        # No DAG constraints — pure random unmasking
        return RandomScheduler(), None

    elif dag_name == "confidence":
        # No DAG constraints — confidence-based (LLaDA default)
        return ConfidenceScheduler(), None

    elif dag_name == "linear":
        # Left-to-right chain (no DAG object needed, scheduler handles order)
        return LinearScheduler(), None

    elif dag_name == "cot":
        # Chain-of-thought: partition generation into reasoning steps
        dag = chain_of_thought_dag(
            seq_len=seq_len,
            num_steps=args.cot_steps,
            prompt_len=0,  # We handle prompt separately
            device=device,
        )
        scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
        return scheduler, dag

    elif dag_name == "skeleton":
        # Skeleton-then-detail: every 3rd token is "structural"
        structural = list(range(0, seq_len, 3))
        detail = list(range(1, seq_len, 3))
        dag = skeleton_then_detail_dag(
            seq_len=seq_len,
            skeleton_positions=structural,
            detail_positions=detail,
            device=device,
        )
        scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
        return scheduler, dag

    elif dag_name == "bidirectional":
        dag = bidirectional_dag(
            seq_len=seq_len,
            num_segments=4,
            device=device,
        )
        scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
        return scheduler, dag

    elif dag_name == "answer_first":
        # Last 20% of positions are the "answer" — unmask first
        answer_start = int(seq_len * 0.8)
        from dllm_reason.graph.templates import answer_first_dag
        dag = answer_first_dag(
            seq_len=seq_len,
            answer_positions=list(range(answer_start, seq_len)),
            reasoning_segments=3,
            device=device,
        )
        scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
        return scheduler, dag

    else:
        raise ValueError(f"Unknown DAG: {dag_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading LLaDA: {args.model_id}")
    print(f"{'='*60}\n")

    from dllm_reason.models.llada import LLaDAWrapper

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    model = LLaDAWrapper(
        model_id=args.model_id,
        max_seq_len=args.generation_len + 512,  # prompt + generation
        torch_dtype=dtype_map[args.torch_dtype],
        device_map=args.device_map,
    )

    device = model.device
    print(f"Model loaded on device: {device}")

    # ── Summary table setup ─────────────────────────────────────────────────
    all_results = {}  # {dag_name: {benchmark: metrics}}
    summary_rows = []

    # ── Evaluation loop ─────────────────────────────────────────────────────
    for dag_name in args.dags:
        all_results[dag_name] = {}

        for benchmark_name in args.benchmarks:
            run_key = f"{benchmark_name}_{dag_name}"
            result_path = output_dir / f"{run_key}.json"

            if args.resume and result_path.exists():
                print(f"[SKIP] {run_key} — already done")
                with open(result_path) as f:
                    result = json.load(f)
                all_results[dag_name][benchmark_name] = result
                continue

            print(f"\n{'─'*60}")
            print(f"Evaluating: benchmark={benchmark_name}  dag={dag_name}")
            print(f"{'─'*60}")

            # Build scheduler for this DAG
            scheduler, dag = build_dag_scheduler(
                dag_name, args.generation_len, args, device
            )

            # Build evaluator
            from dllm_reason.eval.benchmarks import BENCHMARK_REGISTRY
            evaluator_cls = BENCHMARK_REGISTRY[benchmark_name]

            eval_kwargs = {
                "model": model,
                "scheduler": scheduler,
                "num_steps": args.num_steps,
                "block_length": args.block_length,
                "temperature": args.temperature,
                "cfg_scale": args.cfg_scale,
                "remasking": args.remasking,
                "max_new_tokens": args.max_new_tokens,
                "num_samples": args.num_samples,
                "run_tests": not args.no_run_tests,
                "verbose_errors": args.verbose_errors,
                # detailed output saving
                "save_outputs": args.save_outputs,
                "save_dir": output_dir,
                "save_qa": not args.no_save_qa,
                "save_ground_truth": not args.no_save_ground_truth,
                "record_trajectory": args.record_trajectory,
                "output_formats": args.output_formats,
                "run_tag": dag_name,
            }

            if benchmark_name == "mmlu" and args.mmlu_subjects:
                eval_kwargs["subjects"] = args.mmlu_subjects

            evaluator = evaluator_cls(**eval_kwargs)

            # Run evaluation
            start_time = time.time()
            result = evaluator.evaluate()
            elapsed = time.time() - start_time

            result["dag"] = dag_name
            result["elapsed_seconds"] = elapsed
            result["model_id"] = args.model_id
            result["num_steps"] = args.num_steps

            # Save per-run result
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {result_path}")

            all_results[dag_name][benchmark_name] = result

            # Print quick summary
            metric = _get_primary_metric(result)
            print(f"  Result: {metric['name']} = {metric['value']:.4f} ({elapsed:.1f}s)")

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'DAG':<20}", end="")
    for bm in args.benchmarks:
        print(f"  {bm:>12}", end="")
    print()
    print("-" * (20 + 14 * len(args.benchmarks)))

    for dag_name in args.dags:
        print(f"{dag_name:<20}", end="")
        for bm in args.benchmarks:
            result = all_results.get(dag_name, {}).get(bm, {})
            metric = _get_primary_metric(result)
            print(f"  {metric['value']:>12.4f}", end="")
        print()

    # Save full summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


def _get_primary_metric(result: dict) -> dict:
    """Extract the primary metric from a result dict."""
    if not result:
        return {"name": "N/A", "value": 0.0}

    bm = result.get("benchmark", "")
    if bm == "mbpp":
        return {"name": "pass@1", "value": result.get("pass@1", 0.0)}
    elif bm == "humaneval":
        return {"name": "pass@1", "value": result.get("pass@1", 0.0)}
    elif bm == "hotpotqa":
        return {"name": "EM", "value": result.get("exact_match", 0.0)}
    elif bm == "mmlu":
        return {"name": "accuracy", "value": result.get("accuracy", 0.0)}
    else:
        # Try common keys
        for key in ["pass@1", "accuracy", "exact_match", "f1"]:
            if key in result:
                return {"name": key, "value": result[key]}
        return {"name": "N/A", "value": 0.0}


if __name__ == "__main__":
    main()
