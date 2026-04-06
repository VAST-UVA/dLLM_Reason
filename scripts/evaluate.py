"""Evaluation entry point: model + multiple schedulers on reasoning benchmarks.

Usage:
    # Evaluate all schedulers on GSM8K:
    python scripts/evaluate.py \
        --model mdlm \
        --checkpoint checkpoints/mdlm_gsm8k/best.pt \
        --dataset gsm8k \
        --schedulers random confidence linear cot skeleton \
        --output_dir results/mdlm_gsm8k

    # Evaluate LLaDA with specific DAG:
    python scripts/evaluate.py \
        --model llada \
        --checkpoint GSAI-ML/LLaDA-8B-Instruct \
        --dataset gsm8k \
        --schedulers cot \
        --num_samples 200
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dLLM with different unmasking strategies")

    parser.add_argument("--model", type=str, default="mdlm",
                        choices=["mdlm", "sedd", "d3pm", "llada"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint or HF model ID")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "arc", "prontoqa"])
    parser.add_argument("--schedulers", nargs="+",
                        default=["random", "confidence", "linear", "cot", "skeleton", "bidirectional"],
                        help="Schedulers to evaluate")

    # Inference params
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generation_len", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--cot_steps", type=int, default=4)

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def build_scheduler(name: str, seq_len: int, cot_steps: int, device):
    from dllm_reason.scheduler.random_scheduler import RandomScheduler
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
    from dllm_reason.scheduler.linear_scheduler import LinearScheduler
    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.graph.dag import TokenDAG
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag, bidirectional_dag,
    )

    if name == "random":
        return RandomScheduler()
    elif name == "confidence":
        return ConfidenceScheduler()
    elif name == "linear":
        return LinearScheduler()
    elif name == "cot":
        dag = chain_of_thought_dag(seq_len, cot_steps, device=device)
        return DAGScheduler(dag, "confidence_topk")
    elif name == "skeleton":
        structural = list(range(0, seq_len, 3))
        detail = list(range(1, seq_len, 3))
        dag = skeleton_then_detail_dag(seq_len, structural, detail, device=device)
        return DAGScheduler(dag, "confidence_topk")
    elif name == "bidirectional":
        dag = bidirectional_dag(seq_len, 4, device=device)
        return DAGScheduler(dag, "confidence_topk")
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    from dllm_reason.data.tokenizer import get_tokenizer
    tokenizer_name = args.checkpoint if args.model == "llada" else args.tokenizer
    tokenizer = get_tokenizer(tokenizer_name, add_mask_token=True)
    vocab_size = len(tokenizer)

    # ── Load model ───────────────────────────────────────────────────────────
    if args.model == "llada":
        from dllm_reason.models.llada import LLaDAWrapper
        model = LLaDAWrapper(
            model_id=args.checkpoint,
            max_seq_len=args.max_seq_len,
        )
    else:
        from dllm_reason.utils.registry import MODEL_REGISTRY
        import dllm_reason.models.mdlm, dllm_reason.models.sedd, dllm_reason.models.d3pm
        model_cls = MODEL_REGISTRY.get(args.model)
        model = model_cls(vocab_size=vocab_size, max_seq_len=args.max_seq_len)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)

    print(f"Model: {args.model} | Device: {device}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    from dllm_reason.data.reasoning_datasets import load_reasoning_dataset
    dataset = load_reasoning_dataset(args.dataset, split="test")
    if args.num_samples:
        dataset = dataset[:args.num_samples]
    print(f"Dataset: {args.dataset} ({len(dataset)} examples)")

    # ── Run evaluations ──────────────────────────────────────────────────────
    from dllm_reason.eval.reasoning_eval import ReasoningEvaluator, MultiSchedulerComparison

    evaluator = ReasoningEvaluator(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        generation_len=args.generation_len,
        num_steps=args.num_steps,
        temperature=args.temperature,
        num_samples=args.num_samples,
        device=device,
    )

    schedulers = {
        name: build_scheduler(name, args.generation_len, args.cot_steps, device)
        for name in args.schedulers
    }

    comparison = MultiSchedulerComparison(evaluator, output_dir=args.output_dir)
    results = comparison.run(
        schedulers=schedulers,
        datasets={args.dataset: dataset},
        model_name=args.model,
        resume=args.resume,
    )

    print(f"\nResults saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
