"""DAG structure search entry point.

Searches for optimal DAG structures that maximize reasoning accuracy
when used to guide the unmasking order of a dLLM.

Usage:
    # Greedy search with MDLM on GSM8K:
    python scripts/search_dag.py \
        --model mdlm \
        --checkpoint checkpoints/mdlm_gsm8k/best.pt \
        --dataset gsm8k \
        --method greedy \
        --budget 100 \
        --output_dir results/dag_search

    # Evolutionary search:
    python scripts/search_dag.py \
        --method evolutionary \
        --population_size 20 \
        --budget 200

    # RL policy search:
    python scripts/search_dag.py \
        --method rl_policy \
        --budget 300
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Search for optimal DAG structures")

    parser.add_argument("--model", type=str, default="mdlm",
                        choices=["mdlm", "sedd", "d3pm", "llada"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "arc", "prontoqa"])
    parser.add_argument("--tokenizer", type=str, default="gpt2")

    # Search method
    parser.add_argument("--method", type=str, default="evolutionary",
                        choices=["greedy", "evolutionary", "rl_policy", "differentiable"])
    parser.add_argument("--budget", type=int, default=100,
                        help="Number of DAG evaluations")

    # Evolutionary params
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--mutation_rate", type=float, default=0.3)

    # Initial DAG seed
    parser.add_argument("--init_dag", type=str, default="cot",
                        choices=["cot", "skeleton", "linear"],
                        help="Initial DAG to start search from")
    parser.add_argument("--init_cot_steps", type=int, default=4)

    # Fitness config
    parser.add_argument("--fitness", type=str, default="accuracy",
                        choices=["accuracy", "perplexity", "combined"])
    parser.add_argument("--fitness_samples", type=int, default=50,
                        help="Samples per DAG evaluation")
    parser.add_argument("--num_steps", type=int, default=32,
                        help="Inference steps for fitness evaluation")

    # Sequence length for DAG
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length (generation length) for DAG positions")
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--output_dir", type=str, default="results/dag_search")

    return parser.parse_args()


def build_initial_dag(name: str, seq_len: int, cot_steps: int, device):
    from dllm_reason.graph.dag import TokenDAG
    from dllm_reason.graph.templates import chain_of_thought_dag

    if name == "cot":
        return chain_of_thought_dag(seq_len, cot_steps, device=device)
    elif name == "linear":
        return TokenDAG.linear_chain(seq_len, device=device)
    elif name == "skeleton":
        from dllm_reason.graph.templates import skeleton_then_detail_dag
        return skeleton_then_detail_dag(seq_len, list(range(0, seq_len, 3)),
                                        list(range(1, seq_len, 3)), device=device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"DAG Search: method={args.method}, budget={args.budget}, dataset={args.dataset}")

    # ── Load tokenizer + data ─────────────────────────────────────────────────
    from dllm_reason.data.tokenizer import get_tokenizer
    from dllm_reason.data.reasoning_datasets import load_reasoning_dataset

    tokenizer_name = args.checkpoint if args.model == "llada" else args.tokenizer
    tokenizer = get_tokenizer(tokenizer_name, add_mask_token=True)

    dataset = load_reasoning_dataset(args.dataset, split="train")
    eval_dataset = dataset[:args.fitness_samples * 2]  # Extra for batching

    # ── Load model ────────────────────────────────────────────────────────────
    if args.model == "llada":
        from dllm_reason.models.llada import LLaDAWrapper
        model = LLaDAWrapper(model_id=args.checkpoint, max_seq_len=args.max_seq_len)
    else:
        from dllm_reason.utils.registry import MODEL_REGISTRY
        import dllm_reason.models.mdlm, dllm_reason.models.sedd, dllm_reason.models.d3pm
        model_cls = MODEL_REGISTRY.get(args.model)
        vocab_size = len(tokenizer)
        model = model_cls(vocab_size=vocab_size, max_seq_len=args.max_seq_len)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model = model.to(device)

    # ── Build fitness function ─────────────────────────────────────────────────
    from dllm_reason.data.reasoning_datasets import ReasoningDataset
    from dllm_reason.data.collator import DiffusionCollator
    from torch.utils.data import DataLoader

    eval_ds = ReasoningDataset(eval_dataset, tokenizer, max_seq_len=args.max_seq_len)
    eval_loader = DataLoader(
        eval_ds, batch_size=8, shuffle=True,
        collate_fn=DiffusionCollator(mask_token_id=model.mask_token_id),
    )

    from dllm_reason.eval.metrics import extract_number

    def answer_extractor(token_ids):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        num = extract_number(text)
        return num if num else text.strip()

    if args.fitness == "accuracy":
        from dllm_reason.search.fitness import accuracy_fitness
        def eval_fn(model, dag):
            return accuracy_fitness(
                model, dag, eval_loader, answer_extractor,
                max_samples=args.fitness_samples, num_steps=args.num_steps,
            )
    elif args.fitness == "perplexity":
        from dllm_reason.search.fitness import perplexity_fitness
        def eval_fn(model, dag):
            return perplexity_fitness(model, dag, eval_loader, max_samples=args.fitness_samples)
    else:
        from dllm_reason.search.fitness import combined_fitness
        def eval_fn(model, dag):
            return combined_fitness(model, dag, eval_loader, answer_extractor,
                                    max_samples=args.fitness_samples)

    # ── Initial DAG ───────────────────────────────────────────────────────────
    initial_dag = build_initial_dag(args.init_dag, args.seq_len, args.init_cot_steps, device)

    # ── Build searcher ────────────────────────────────────────────────────────
    from dllm_reason.utils.registry import SEARCH_REGISTRY
    import dllm_reason.search.greedy, dllm_reason.search.evolutionary
    import dllm_reason.search.rl_policy, dllm_reason.search.differentiable

    if args.method == "greedy":
        from dllm_reason.search.greedy import GreedyEdgeSearch
        searcher = GreedyEdgeSearch(initial_dag=initial_dag)
    elif args.method == "evolutionary":
        from dllm_reason.search.evolutionary import EvolutionarySearch
        searcher = EvolutionarySearch(
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            initial_dags=[initial_dag],
        )
    elif args.method == "rl_policy":
        from dllm_reason.search.rl_policy import RLPolicySearch
        searcher = RLPolicySearch(max_seq_len=args.seq_len)
    elif args.method == "differentiable":
        from dllm_reason.search.differentiable import DifferentiableDAGSearch
        searcher = DifferentiableDAGSearch()

    # ── Run search ────────────────────────────────────────────────────────────
    print(f"Starting search (budget={args.budget})...")
    result = searcher.search(
        model=model,
        eval_fn=eval_fn,
        seq_len=args.seq_len,
        budget=args.budget,
    )

    print(f"\nSearch complete!")
    print(f"Best fitness: {result.best_fitness:.4f}")
    print(f"Best DAG: {result.best_dag}")

    # ── Save results ──────────────────────────────────────────────────────────
    from dllm_reason.eval.dag_analysis import analyze_dag

    stats = analyze_dag(result.best_dag)
    output = {
        "method": args.method,
        "dataset": args.dataset,
        "best_fitness": result.best_fitness,
        "dag_stats": stats.to_dict(),
        "metadata": result.metadata,
        "history": result.history,
    }

    with open(output_dir / "search_result.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save adjacency matrix
    torch.save(
        result.best_dag.adjacency.cpu(),
        output_dir / "best_dag_adjacency.pt",
    )

    print(f"Results saved to {output_dir}/")

    # ── Visualize search progress ─────────────────────────────────────────────
    try:
        from dllm_reason.eval.dag_analysis import search_history_plot
        fig = search_history_plot(result.history, title=f"{args.method} Search on {args.dataset}")
        fig.savefig(output_dir / "search_progress.png", dpi=150, bbox_inches="tight")
        print(f"Search progress plot saved.")
    except Exception as e:
        print(f"[WARNING] Could not save plot: {e}")


if __name__ == "__main__":
    main()
