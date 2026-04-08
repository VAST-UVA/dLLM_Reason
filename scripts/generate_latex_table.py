"""Generate LaTeX comparison tables from evaluation results.

Usage:
    python scripts/generate_latex_table.py results/full_comparison_*/summary.json
    python scripts/generate_latex_table.py results/full_comparison_*/summary.json --output paper_table.tex
"""

import argparse
import json
from pathlib import Path


# Human-readable names for display
STRATEGY_NAMES = {
    "confidence": "Confidence",
    "random": "Random",
    "entropy": "Entropy",
    "semi_ar": "Semi-AR",
    "maskgit_cosine": "MaskGIT",
    "critical_token_first": "Critical-First",
    "curriculum": "Curriculum",
    "linear": "Linear (L→R)",
    "cot": "CoT DAG",
    "skeleton": "Skeleton",
    "bidirectional": "Bidirectional",
    "answer_first": "Answer-First",
    "adaptive_dynamic": "Adaptive (ours)",
}

BENCHMARK_NAMES = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "mbpp": "MBPP",
    "humaneval": "HumanEval",
    "arc": "ARC-C",
    "mmlu": "MMLU",
    "hotpotqa": "HotpotQA",
    "prontoqa": "ProntoQA",
    "gpqa": "GPQA",
    "aime": "AIME",
}

METRIC_KEYS = {
    "gsm8k": "accuracy",
    "math": "accuracy",
    "mbpp": "pass@1",
    "humaneval": "pass@1",
    "arc": "accuracy",
    "mmlu": "accuracy",
    "hotpotqa": "exact_match",
    "prontoqa": "accuracy",
    "gpqa": "accuracy",
    "aime": "accuracy",
}


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_metric(result: dict, benchmark: str) -> float | None:
    """Extract the primary metric value from a benchmark result."""
    if not result:
        return None
    key = METRIC_KEYS.get(benchmark, "accuracy")
    return result.get(key)


def generate_latex(
    summary: dict,
    strategies: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> str:
    """Generate a LaTeX booktabs table from summary results."""
    # Auto-detect strategies and benchmarks from data
    all_strategies = list(summary.keys())
    all_benchmarks = set()
    for dag_results in summary.values():
        if isinstance(dag_results, dict):
            all_benchmarks.update(dag_results.keys())

    strategies = strategies or all_strategies
    benchmarks = benchmarks or sorted(all_benchmarks)

    # Filter to only strategies/benchmarks with data
    strategies = [s for s in strategies if s in summary]
    benchmarks = [b for b in benchmarks if any(
        b in summary.get(s, {}) for s in strategies
    )]

    if not strategies or not benchmarks:
        return "% No data found in summary.json"

    # Build score matrix and find best per column
    scores: dict[str, dict[str, float | None]] = {}
    best_per_bm: dict[str, float] = {}

    for strat in strategies:
        scores[strat] = {}
        for bm in benchmarks:
            val = extract_metric(summary.get(strat, {}).get(bm, {}), bm)
            scores[strat][bm] = val
            if val is not None:
                if bm not in best_per_bm or val > best_per_bm[bm]:
                    best_per_bm[bm] = val

    # Generate LaTeX
    n_cols = len(benchmarks)
    col_spec = "l" + "c" * n_cols
    bm_headers = " & ".join(BENCHMARK_NAMES.get(b, b) for b in benchmarks)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of unmasking strategies across benchmarks. "
        r"Best results per benchmark are in \textbf{bold}.}",
        r"\label{tab:strategy_comparison}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Strategy & {bm_headers} \\\\",
        r"\midrule",
    ]

    for strat in strategies:
        name = STRATEGY_NAMES.get(strat, strat)
        cells = []
        for bm in benchmarks:
            val = scores[strat][bm]
            if val is None:
                cells.append("--")
            else:
                formatted = f"{val*100:.1f}"
                if bm in best_per_bm and abs(val - best_per_bm[bm]) < 1e-6:
                    formatted = rf"\textbf{{{formatted}}}"
                cells.append(formatted)
        row = f"{name} & {' & '.join(cells)} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from eval results")
    parser.add_argument("summary_json", type=str, help="Path to summary.json")
    parser.add_argument("--output", type=str, default=None, help="Output .tex file")
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    latex = generate_latex(summary)

    if args.output:
        Path(args.output).write_text(latex, encoding="utf-8")
        print(f"LaTeX table saved to {args.output}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
