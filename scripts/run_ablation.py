"""Ablation experiment runner for dLLM-Reason.

Defines a comprehensive set of ablation experiments and runs them
through the research pipeline (scripts/run_research_pipeline.py).

Each experiment is a dict that overrides specific pipeline parameters.
The runner iterates through selected experiments, calling
``run_pipeline()`` for each.

Usage:
    # List all experiments (dry run):
    python scripts/run_ablation.py --dry_run

    # Run all experiments:
    python scripts/run_ablation.py

    # Run specific experiments by name:
    python scripts/run_ablation.py --experiments scheduler_compare e2e_sft_per_template

    # Run all Stage-3 ablations reusing existing episodes:
    python scripts/run_ablation.py --experiments sft_per_template sft_consensus sft_no_dag \\
        --episodes_from runs/ablation/sweep_all_strategies

See docs/ABLATION_SETTINGS.md for all ablation dimensions and design rationale.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Experiment definitions ───────────────────────────────────────────────────

ABLATION_EXPERIMENTS = [

    # ═══════════════════════════════════════════════════════════════════════
    # A. Scheduler Comparison (Stage 1 only)
    # ═══════════════════════════════════════════════════════════════════════

    {
        "name": "scheduler_compare",
        "desc": "Compare all 13 unmasking schedulers on gsm8k + math",
        "stages": [1],
        "datasets": ["gsm8k", "math"],
        "s1_schedulers": [
            "confidence", "random", "linear", "entropy", "semi_ar",
            "maskgit_cosine", "critical_token_first", "curriculum",
            "adaptive_dynamic",
            "cot", "skeleton", "bidirectional", "answer_first",
        ],
        "num_samples": 500,
    },

    {
        "name": "scheduler_quick",
        "desc": "Quick scheduler comparison (3 schedulers, gsm8k only, 50 samples)",
        "stages": [1],
        "datasets": ["gsm8k"],
        "s1_schedulers": ["confidence", "cot", "skeleton"],
        "num_samples": 50,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # B. DAG Discovery Methods (Stage 1+2)
    # ═══════════════════════════════════════════════════════════════════════

    {
        "name": "sweep_all_strategies",
        "desc": "Template sweep: 7 strategies on gsm8k (200 samples)",
        "stages": [1, 2],
        "datasets": ["gsm8k"],
        "s2_method": "sweep",
        "s2_strategies": [
            "confidence", "cot", "skeleton", "bidirectional",
            "answer_first", "linear", "random",
        ],
        "num_samples": 200,
    },

    {
        "name": "search_greedy",
        "desc": "Per-prompt greedy DAG search (50 samples)",
        "stages": [2],
        "datasets": ["gsm8k"],
        "s2_method": "search",
        "s2_search_method": "greedy",
        "s2_search_budget": 30,
        "num_samples": 50,
    },

    {
        "name": "search_evolutionary",
        "desc": "Per-prompt evolutionary DAG search (50 samples)",
        "stages": [2],
        "datasets": ["gsm8k"],
        "s2_method": "search",
        "s2_search_method": "evolutionary",
        "s2_search_budget": 50,
        "num_samples": 50,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # C. Training Mode x DAG Mode (Stage 3)
    # ═══════════════════════════════════════════════════════════════════════

    # — SFT variants —
    {
        "name": "sft_per_template",
        "desc": "SFT with per-template DAG bias",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
    },
    {
        "name": "sft_consensus",
        "desc": "SFT with consensus DAG (majority vote)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "consensus",
    },
    {
        "name": "sft_no_dag",
        "desc": "SFT without DAG bias (ablation baseline)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "none",
    },

    # — GRPO variants —
    {
        "name": "grpo_per_template",
        "desc": "GRPO with per-template DAG bias (kl=0.01)",
        "stages": [3],
        "s3_mode": "grpo",
        "s3_dag_mode": "per_template",
        "kl_coeff": 0.01,
    },
    {
        "name": "grpo_consensus",
        "desc": "GRPO with consensus DAG (kl=0.02)",
        "stages": [3],
        "s3_mode": "grpo",
        "s3_dag_mode": "consensus",
        "kl_coeff": 0.02,
    },
    {
        "name": "grpo_no_dag",
        "desc": "GRPO without DAG bias (ablation baseline)",
        "stages": [3],
        "s3_mode": "grpo",
        "s3_dag_mode": "none",
    },

    # — DiFFPO variants —
    {
        "name": "diffppo_default",
        "desc": "DiFFPO with per-template DAG bias",
        "stages": [3],
        "s3_mode": "diffppo",
        "s3_dag_mode": "per_template",
    },
    {
        "name": "diffppo_train_sampler",
        "desc": "DiFFPO with joint step-budget controller training",
        "stages": [3],
        "s3_mode": "diffppo",
        "s3_dag_mode": "per_template",
        "train_sampler": True,
        "ppo_clip_eps": 0.1,
    },

    # — UnmaskRL variants —
    {
        "name": "unmask_rl_small",
        "desc": "UnmaskRL with small policy net (d=64, group=4)",
        "stages": [3],
        "s3_mode": "unmask_rl",
        "unmask_d_model": 64,
        "unmask_group_size": 4,
    },
    {
        "name": "unmask_rl_large",
        "desc": "UnmaskRL with large policy net (d=128, group=8)",
        "stages": [3],
        "s3_mode": "unmask_rl",
        "unmask_d_model": 128,
        "unmask_group_size": 8,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # D. DAG Bias Strength Sweep (Stage 3)
    # ═══════════════════════════════════════════════════════════════════════

    {
        "name": "bias_0.0",
        "desc": "DAG bias = 0.0 (no bias, ablation control)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
        "dag_bias_strength": 0.0,
    },
    {
        "name": "bias_0.1",
        "desc": "DAG bias = 0.1 (very weak bias)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
        "dag_bias_strength": 0.1,
    },
    {
        "name": "bias_0.3",
        "desc": "DAG bias = 0.3 (moderate bias)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
        "dag_bias_strength": 0.3,
    },
    {
        "name": "bias_0.5",
        "desc": "DAG bias = 0.5 (default)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
        "dag_bias_strength": 0.5,
    },
    {
        "name": "bias_0.8",
        "desc": "DAG bias = 0.8 (strong bias)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
        "dag_bias_strength": 0.8,
    },
    {
        "name": "bias_1.0",
        "desc": "DAG bias = 1.0 (full bias)",
        "stages": [3],
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
        "dag_bias_strength": 1.0,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # E. End-to-End Pipeline Experiments
    # ═══════════════════════════════════════════════════════════════════════

    {
        "name": "e2e_sft_per_template",
        "desc": "Full 3-stage: eval -> discover -> SFT per_template",
        "stages": [1, 2, 3],
        "datasets": ["gsm8k"],
        "num_samples": 200,
        "s3_mode": "sft",
        "s3_dag_mode": "per_template",
    },

    {
        "name": "e2e_grpo_consensus",
        "desc": "Full 3-stage: eval -> discover -> GRPO consensus",
        "stages": [1, 2, 3],
        "datasets": ["gsm8k"],
        "num_samples": 200,
        "s3_mode": "grpo",
        "s3_dag_mode": "consensus",
        "kl_coeff": 0.02,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # F. Multi-Dataset Evaluation
    # ═══════════════════════════════════════════════════════════════════════

    {
        "name": "multi_dataset_eval",
        "desc": "Evaluate 3 schedulers on 4 reasoning datasets",
        "stages": [1],
        "datasets": ["gsm8k", "math", "arc", "prontoqa"],
        "s1_schedulers": ["confidence", "cot", "skeleton"],
        "num_samples": 200,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # G. Inference Parameter Sensitivity
    # ═══════════════════════════════════════════════════════════════════════

    {
        "name": "steps_32",
        "desc": "Baseline with num_steps=32 (fast, lower quality)",
        "stages": [1],
        "datasets": ["gsm8k"],
        "s1_schedulers": ["confidence"],
        "num_steps": 32,
        "num_samples": 200,
    },
    {
        "name": "steps_256",
        "desc": "Baseline with num_steps=256 (slow, higher quality)",
        "stages": [1],
        "datasets": ["gsm8k"],
        "s1_schedulers": ["confidence"],
        "num_steps": 256,
        "num_samples": 200,
    },
    {
        "name": "temp_0.5",
        "desc": "Baseline with temperature=0.5",
        "stages": [1],
        "datasets": ["gsm8k"],
        "s1_schedulers": ["confidence"],
        "temperature": 0.5,
        "num_samples": 200,
    },
    {
        "name": "temp_1.0",
        "desc": "Baseline with temperature=1.0",
        "stages": [1],
        "datasets": ["gsm8k"],
        "s1_schedulers": ["confidence"],
        "temperature": 1.0,
        "num_samples": 200,
    },
]

# Build name → experiment lookup
_EXP_BY_NAME = {exp["name"]: exp for exp in ABLATION_EXPERIMENTS}


# ── Runner ───────────────────────────────────────────────────────────────────


def run_experiments(
    experiments: list[dict],
    base_run_dir: Path,
    common_overrides: dict,
    resume: bool = False,
    episodes_from: str | None = None,
):
    """Run a list of ablation experiments sequentially."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_research_pipeline import run_pipeline

    results_summary: list[dict] = []
    total = len(experiments)

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        desc = exp.get("desc", "")
        run_dir = base_run_dir / name

        print(f"\n{'#'*60}")
        print(f"# Experiment [{i}/{total}]: {name}")
        print(f"# {desc}")
        print(f"# Output: {run_dir}")
        print(f"{'#'*60}\n")

        # Build config: defaults + experiment overrides + common overrides
        config = {**exp, "run_dir": str(run_dir)}
        if resume:
            config["resume"] = True
        config.update(common_overrides)

        # For Stage 3 experiments that reuse episodes from another run
        if episodes_from and 3 in exp.get("stages", [1, 2, 3]):
            src_db = Path(episodes_from) / "stage2_discovery" / "episodes.db"
            dst_dir = run_dir / "stage2_discovery"
            if src_db.exists() and not (dst_dir / "episodes.db").exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_db, dst_dir / "episodes.db")
                print(f"  Copied episodes.db from {episodes_from}")
            # Also copy best_dag_per_prompt.json
            src_best = Path(episodes_from) / "stage2_discovery" / "best_dag_per_prompt.json"
            if src_best.exists() and not (dst_dir / "best_dag_per_prompt.json").exists():
                shutil.copy2(src_best, dst_dir / "best_dag_per_prompt.json")

        t0 = time.time()
        try:
            run_pipeline(config)
            elapsed = time.time() - t0
            status = "OK"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"FAILED: {e}"
            print(f"\n  [ERROR] Experiment {name} failed: {e}")

        results_summary.append({
            "name": name,
            "desc": desc,
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
            "run_dir": str(run_dir),
        })

    # Write summary
    summary_path = base_run_dir / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    _print_ablation_summary(results_summary)
    _collect_accuracy_table(experiments, base_run_dir)


def _print_ablation_summary(results: list[dict]):
    """Print experiment execution summary."""
    print(f"\n{'='*60}")
    print("Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Experiment':<30} {'Status':<10} {'Time':>8}")
    print("-" * 50)
    for r in results:
        status = "OK" if r["status"] == "OK" else "FAIL"
        elapsed = f"{r['elapsed_seconds']:.0f}s"
        print(f"{r['name']:<30} {status:<10} {elapsed:>8}")
    print()


def _collect_accuracy_table(experiments: list[dict], base_run_dir: Path):
    """Collect accuracy from all Stage 1 results into a comparison table."""
    rows = []
    for exp in experiments:
        name = exp["name"]
        s1_dir = base_run_dir / name / "stage1_baseline"
        if not s1_dir.exists():
            continue

        summary_path = s1_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            for key, info in summary.items():
                rows.append({
                    "experiment": name,
                    "run_key": key,
                    "accuracy": info.get("accuracy", "N/A"),
                    "correct": info.get("correct", "N/A"),
                    "total": info.get("total", "N/A"),
                })

    if rows:
        table_path = base_run_dir / "accuracy_table.json"
        with open(table_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Accuracy table written to: {table_path}")

        # Also write CSV
        csv_path = base_run_dir / "accuracy_table.csv"
        with open(csv_path, "w") as f:
            f.write("experiment,run_key,accuracy,correct,total\n")
            for r in rows:
                f.write(f"{r['experiment']},{r['run_key']},{r['accuracy']},"
                        f"{r['correct']},{r['total']}\n")
        print(f"Accuracy CSV written to:   {csv_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="Run ablation experiments for dLLM-Reason research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiments", type=str, nargs="+", default=["all"],
                    help="Experiment names to run (default: all)")
    # Options: "all", or select experiment names from ABLATION_EXPERIMENTS
    p.add_argument("--base_run_dir", type=str, default="runs/ablation",
                    help="Base directory for all ablation outputs")
    p.add_argument("--resume", action="store_true",
                    help="Resume incomplete experiments")
    p.add_argument("--dry_run", action="store_true",
                    help="Print experiment list and exit")
    p.add_argument("--episodes_from", type=str, default=None,
                    help="Copy episodes.db from this run_dir for Stage 3 experiments")

    # Common overrides (applied to all experiments)
    p.add_argument("--api_url", type=str, default="http://localhost:8000")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Override model checkpoint for all experiments")

    args = p.parse_args()

    # Select experiments
    if "all" in args.experiments:
        selected = ABLATION_EXPERIMENTS
    else:
        selected = []
        for name in args.experiments:
            if name in _EXP_BY_NAME:
                selected.append(_EXP_BY_NAME[name])
            else:
                print(f"[WARN] Unknown experiment: {name}")
                print(f"       Available: {', '.join(_EXP_BY_NAME.keys())}")

    if not selected:
        print("No experiments selected.")
        return

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"Ablation Experiments — DRY RUN")
        print(f"{'='*60}")
        print(f"  Base dir:       {args.base_run_dir}")
        print(f"  Total:          {len(selected)} experiments")
        if args.episodes_from:
            print(f"  Episodes from:  {args.episodes_from}")
        print()
        print(f"{'Name':<30} {'Stages':<12} {'Description'}")
        print("-" * 70)
        for exp in selected:
            stages = str(exp.get("stages", [1, 2, 3]))
            print(f"{exp['name']:<30} {stages:<12} {exp.get('desc', '')}")
        print()
        return

    # Build common overrides from CLI
    common = {"api_url": args.api_url}
    if args.checkpoint:
        common["checkpoint"] = args.checkpoint

    run_experiments(
        experiments=selected,
        base_run_dir=Path(args.base_run_dir),
        common_overrides=common,
        resume=args.resume,
        episodes_from=args.episodes_from,
    )


if __name__ == "__main__":
    main()
