"""End-to-end pipeline: data → collect → search → learn → eval.

Stages
------
  download   Download the target dataset to local datasets/
  collect    Run the dLLM on the dataset with multiple strategies;
             store episodes in an EpisodeStore (SQLite)
  search     Search for the best DAG structure (greedy / evolutionary)
  learn      Fine-tune from episodes: sft / grpo / diffppo / unmask_rl
  eval       Benchmark the fine-tuned model on the target dataset

All outputs are written under a single run directory:
    runs/<run_name>/
        episodes.db               ← EpisodeStore (collect stage)
        dag_search/               ← search results + best_dag_adjacency.pt
        finetuned/                ← fine-tuned model checkpoint(s)
        eval/                     ← evaluation results
        pipeline.log              ← combined stdout/stderr of every stage
        pipeline_manifest.json    ← config + timing + exit codes

Usage
-----
# Full pipeline, LLaDA on GSM8K, evolutionary search + GRPO fine-tuning
python scripts/run_pipeline.py \\
    --checkpoint GSAI-ML/LLaDA-8B-Instruct \\
    --dataset gsm8k \\
    --stages download collect search learn eval \\
    --rl_mode grpo \\
    --run_name gsm8k_grpo

# Skip download (data already present), run collect + learn only
python scripts/run_pipeline.py \\
    --checkpoint checkpoints/llada-instruct \\
    --dataset gsm8k \\
    --stages collect learn \\
    --rl_mode diffppo

# Dry-run: print every command without executing
python scripts/run_pipeline.py \\
    --checkpoint checkpoints/llada-instruct \\
    --dataset gsm8k \\
    --dry_run

# Resume a previous run (skip stages whose output already exists)
python scripts/run_pipeline.py \\
    --checkpoint checkpoints/llada-instruct \\
    --dataset gsm8k \\
    --run_dir runs/gsm8k_grpo \\
    --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Colour helpers (no deps) ──────────────────────────────────────────────────

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[32m"
_CYAN  = "\033[36m"
_YELLOW= "\033[33m"
_RED   = "\033[31m"
_RESET = "\033[0m"

def _c(text: str, code: str) -> str:
    """Wrap text in ANSI colour (no-op on Windows without VT mode)."""
    return f"{code}{text}{_RESET}"

def banner(title: str, index: int, total: int) -> None:
    width = 68
    line  = "─" * width
    print(f"\n{_c(line, _CYAN)}")
    print(f"{_c(f'  Stage {index}/{total}: {title}', _BOLD + _CYAN)}")
    print(f"{_c(line, _CYAN)}")

def ok(msg: str)   -> None: print(_c(f"  ✓  {msg}", _GREEN))
def warn(msg: str) -> None: print(_c(f"  ⚠  {msg}", _YELLOW))
def err(msg: str)  -> None: print(_c(f"  ✗  {msg}", _RED))
def info(msg: str) -> None: print(_c(f"     {msg}", _DIM))


# ── Helpers ───────────────────────────────────────────────────────────────────

PYTHON = sys.executable  # same interpreter that launched this script
SCRIPTS_DIR = Path(__file__).resolve().parent

ALL_STAGES = ["download", "collect", "search", "learn", "eval"]


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _run(
    cmd: list[str],
    log_file: Path,
    dry_run: bool = False,
    env: dict | None = None,
) -> int:
    """Run a subprocess, tee output to console + log_file.  Returns exit code."""
    cmd_str = " ".join(str(c) for c in cmd)
    info(f"$ {cmd_str}")

    if dry_run:
        return 0

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*70}\n{cmd_str}\n{'='*70}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            print(f"  {_DIM}{line.rstrip()}{_RESET}")
            lf.write(line)
        proc.wait()
    return proc.returncode


# ── Stage implementations ─────────────────────────────────────────────────────

def stage_download(args: argparse.Namespace, run_dir: Path, log: Path) -> int:
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "download_datasets.py"),
        "--datasets", args.dataset,
        "--output_dir", str(args.data_dir),
    ]
    if args.hf_mirror:
        cmd += ["--mirror", args.hf_mirror]
    return _run(cmd, log, args.dry_run)


def stage_collect(args: argparse.Namespace, run_dir: Path, log: Path) -> int:
    db_path = run_dir / "episodes.db"
    strategies = args.collect_strategies

    cmd = [
        PYTHON, str(SCRIPTS_DIR / "collect_episodes.py"),
        "--model_id",    args.checkpoint,
        "--model_type",  args.model_type,
        "--torch_dtype", args.torch_dtype,
        "--dataset_path", str(args.data_dir / f"{args.dataset}_train.jsonl"),
        "--strategy",    *strategies,
        "--n_samples",   str(args.collect_n_samples),
        "--gen_length",  str(args.gen_length),
        "--num_steps",   str(args.num_steps),
        "--temperature", str(args.temperature),
        "--eval_mode",   "auto",
        "--db_path",     str(db_path),
    ]
    return _run(cmd, log, args.dry_run)


def stage_search(args: argparse.Namespace, run_dir: Path, log: Path) -> int:
    out = run_dir / "dag_search"
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "search_dag.py"),
        "--model",        args.model_type,
        "--checkpoint",   args.checkpoint,
        "--dataset",      args.dataset,
        "--method",       args.search_method,
        "--budget",       str(args.search_budget),
        "--fitness",      args.search_fitness,
        "--fitness_samples", str(args.search_fitness_samples),
        "--num_steps",    str(args.num_steps),
        "--seq_len",      str(args.gen_length),
        "--init_dag",     args.search_init_dag,
        "--output_dir",   str(out),
    ]
    if args.search_method == "evolutionary":
        cmd += [
            "--population_size", str(args.population_size),
            "--mutation_rate",   str(args.mutation_rate),
        ]
    return _run(cmd, log, args.dry_run)


def stage_learn(args: argparse.Namespace, run_dir: Path, log: Path) -> int:
    db_path   = run_dir / "episodes.db"
    out_dir   = run_dir / "finetuned"
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "learn_from_episodes.py"),
        "--db_path",     str(db_path),
        "--model_id",    args.checkpoint,
        "--model_type",  args.model_type,
        "--torch_dtype", args.torch_dtype,
        "--mode",        args.rl_mode,
        "--epochs",      str(args.epochs),
        "--batch_size",  str(args.batch_size),
        "--lr",          str(args.lr),
        "--max_length",  str(args.gen_length + 64),
        "--log_every",   str(args.log_every),
        "--kl_coeff",    str(args.kl_coeff),
        "--output_dir",  str(out_dir),
    ]
    if args.dag_aware:
        cmd.append("--dag_aware")

    if args.rl_mode == "diffppo":
        cmd += [
            "--ppo_clip_eps",        str(args.ppo_clip_eps),
            "--min_steps",           str(args.min_steps),
            "--max_steps",           str(args.num_steps),
            "--step_budget_lambda",  str(args.step_budget_lambda),
        ]
        if args.train_sampler:
            cmd.append("--train_sampler")

    if args.rl_mode == "unmask_rl":
        cmd += [
            "--unmask_group_size", str(args.unmask_group_size),
            "--unmask_num_steps",  str(args.num_steps),
            "--unmask_d_model",    str(args.unmask_d_model),
            "--unmask_n_heads",    str(args.unmask_n_heads),
        ]

    return _run(cmd, log, args.dry_run)


def stage_eval(args: argparse.Namespace, run_dir: Path, log: Path) -> int:
    finetuned = run_dir / "finetuned"
    out_dir   = run_dir / "eval"
    # Use finetuned checkpoint if it exists, else fall back to base
    ckpt = str(finetuned) if (finetuned.exists() or args.dry_run) else args.checkpoint
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "evaluate.py"),
        "--model",       args.model_type,
        "--checkpoint",  ckpt,
        "--dataset",     args.dataset,
        "--schedulers",  *args.eval_schedulers,
        "--num_steps",   str(args.num_steps),
        "--num_samples", str(args.eval_n_samples),
        "--generation_len", str(args.gen_length),
        "--output_dir",  str(out_dir),
    ]
    return _run(cmd, log, args.dry_run)


# ── Stage registry ────────────────────────────────────────────────────────────

_STAGE_FN = {
    "download": stage_download,
    "collect":  stage_collect,
    "search":   stage_search,
    "learn":    stage_learn,
    "eval":     stage_eval,
}

_SKIP_SENTINEL = {
    "download": lambda d: (d.parent / "datasets" / "gsm8k").exists() or
                          Path("datasets").exists(),
    "collect":  lambda d: (d / "episodes.db").exists(),
    "search":   lambda d: (d / "dag_search" / "search_result.json").exists(),
    "learn":    lambda d: (d / "finetuned").exists(),
    "eval":     lambda d: (d / "eval").exists(),
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end dLLM-Reason pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Run control ──────────────────────────────────────────────────────
    p.add_argument("--stages", nargs="+", default=ALL_STAGES,
                   choices=ALL_STAGES, metavar="STAGE",
                   help="Stages to run: download collect search learn eval")
    p.add_argument("--run_name", default=None,
                   help="Friendly name; auto-generated if omitted")
    p.add_argument("--run_dir", default=None,
                   help="Root output directory (overrides --run_name)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--resume", action="store_true",
                   help="Skip stages whose output already exists in run_dir")
    p.add_argument("--stop_on_error", action="store_true", default=True,
                   help="Abort pipeline if any stage returns non-zero exit code")

    # ── Model ────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint", required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--model_type", default="llada",
                   choices=["llada", "mdlm", "sedd", "d3pm"])
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    # ── Data ─────────────────────────────────────────────────────────────
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "math", "arc", "prontoqa"])
    p.add_argument("--data_dir", default=None,
                   help="Dataset directory (default: datasets/<dataset>)")
    p.add_argument("--hf_mirror", default=None,
                   help="HuggingFace mirror URL for download stage")

    # ── Collect stage ────────────────────────────────────────────────────
    p.add_argument("--collect_strategies", nargs="+",
                   default=["confidence", "cot", "entropy"],
                   help="Unmasking strategies to collect episodes for")
    p.add_argument("--collect_n_samples", type=int, default=500,
                   help="Number of dataset samples to collect per strategy")

    # ── Inference / generation ───────────────────────────────────────────
    p.add_argument("--gen_length",  type=int,   default=128,
                   help="Generation length (tokens)")
    p.add_argument("--num_steps",   type=int,   default=128,
                   help="Diffusion denoising steps")
    p.add_argument("--temperature", type=float, default=0.0)

    # ── Search stage ─────────────────────────────────────────────────────
    p.add_argument("--search_method", default="evolutionary",
                   choices=["greedy", "evolutionary", "rl_policy", "differentiable"])
    p.add_argument("--search_budget", type=int, default=100)
    p.add_argument("--search_fitness", default="accuracy",
                   choices=["accuracy", "perplexity", "combined"])
    p.add_argument("--search_fitness_samples", type=int, default=50)
    p.add_argument("--search_init_dag", default="cot",
                   choices=["cot", "skeleton", "linear"])
    p.add_argument("--population_size", type=int, default=20)
    p.add_argument("--mutation_rate",   type=float, default=0.3)

    # ── Learn stage ──────────────────────────────────────────────────────
    p.add_argument("--rl_mode", default="grpo",
                   choices=["sft", "grpo", "diffppo", "unmask_rl"],
                   help="Learning algorithm for the learn stage")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--kl_coeff",   type=float, default=0.01)
    p.add_argument("--log_every",  type=int,   default=20)
    p.add_argument("--dag_aware",  action="store_true",
                   help="Use DAG-biased masking during training")

    # DiFFPO-specific
    p.add_argument("--ppo_clip_eps",       type=float, default=0.2)
    p.add_argument("--train_sampler",      action="store_true")
    p.add_argument("--min_steps",          type=int,   default=8)
    p.add_argument("--step_budget_lambda", type=float, default=0.1)

    # UnmaskRL-specific
    p.add_argument("--unmask_group_size", type=int, default=4)
    p.add_argument("--unmask_d_model",    type=int, default=64)
    p.add_argument("--unmask_n_heads",    type=int, default=4)

    # ── Eval stage ───────────────────────────────────────────────────────
    p.add_argument("--eval_schedulers", nargs="+",
                   default=["confidence", "cot", "random"],
                   help="Schedulers to benchmark in the eval stage")
    p.add_argument("--eval_n_samples", type=int, default=200)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve run directory ─────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.run_name or f"{args.dataset}_{args.rl_mode}_{ts}"
        run_dir = Path("runs") / name

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "pipeline.log"

    # Default data_dir
    if args.data_dir is None:
        args.data_dir = Path("datasets") / args.dataset
    else:
        args.data_dir = Path(args.data_dir)

    # ── Print run summary ─────────────────────────────────────────────────
    width = 68
    print(f"\n{_c('═' * width, _BOLD + _CYAN)}")
    print(_c("  dLLM-Reason Pipeline", _BOLD + _CYAN))
    print(f"{_c('═' * width, _BOLD + _CYAN)}")
    print(f"  run dir   : {run_dir}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  dataset   : {args.dataset}")
    print(f"  stages    : {', '.join(args.stages)}")
    print(f"  rl_mode   : {args.rl_mode}")
    print(f"  search    : {args.search_method}  budget={args.search_budget}")
    if args.dry_run:
        print(_c("  [DRY RUN — no commands will be executed]", _YELLOW))
    if args.resume:
        print(_c("  [RESUME — completed stages will be skipped]", _YELLOW))
    print()

    # ── Manifest ──────────────────────────────────────────────────────────
    manifest: dict = {
        "run_dir":    str(run_dir),
        "checkpoint": args.checkpoint,
        "dataset":    args.dataset,
        "stages":     args.stages,
        "rl_mode":    args.rl_mode,
        "search_method": args.search_method,
        "start_time": datetime.now().isoformat(),
        "stages_result": {},
    }

    active_stages = [s for s in ALL_STAGES if s in args.stages]
    total = len(active_stages)
    pipeline_ok = True

    for idx, stage_name in enumerate(active_stages, start=1):
        banner(stage_name.upper(), idx, total)

        # Resume check
        if args.resume and not args.dry_run:
            sentinel_fn = _SKIP_SENTINEL.get(stage_name)
            if sentinel_fn and sentinel_fn(run_dir):
                warn(f"Output already exists — skipping (--resume)")
                manifest["stages_result"][stage_name] = {
                    "status": "skipped", "duration_s": 0
                }
                continue

        stage_fn = _STAGE_FN[stage_name]
        t0 = time.monotonic()
        rc = stage_fn(args, run_dir, log_file)
        elapsed = time.monotonic() - t0

        result = {
            "status":     "ok" if rc == 0 else "failed",
            "exit_code":  rc,
            "duration_s": round(elapsed, 1),
        }
        manifest["stages_result"][stage_name] = result

        if rc == 0:
            ok(f"Done in {_fmt_duration(elapsed)}")
        else:
            err(f"Stage exited with code {rc} after {_fmt_duration(elapsed)}")
            if args.stop_on_error:
                err("Aborting pipeline (--stop_on_error is set).")
                pipeline_ok = False
                break
            else:
                warn("Continuing despite error (--stop_on_error not set).")

    # ── Final summary ─────────────────────────────────────────────────────
    manifest["end_time"] = datetime.now().isoformat()
    manifest_path = run_dir / "pipeline_manifest.json"
    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n{_c('─' * 68, _CYAN)}")
    print(_c("  Pipeline Summary", _BOLD))
    print(f"{_c('─' * 68, _CYAN)}")

    total_time = 0.0
    for stage_name in active_stages:
        res = manifest["stages_result"].get(stage_name)
        if res is None:
            continue
        status  = res["status"]
        dur     = res["duration_s"]
        total_time += dur
        colour  = _GREEN if status == "ok" else (_YELLOW if status == "skipped" else _RED)
        icon    = "✓" if status == "ok" else ("⊘" if status == "skipped" else "✗")
        dur_str = _fmt_duration(dur) if dur else "—"
        print(f"  {_c(icon, colour)}  {stage_name:<12} "
              f"{_c(status, colour):<20}  {dur_str}")

    print(f"{_c('─' * 68, _CYAN)}")
    print(f"  Total time : {_fmt_duration(total_time)}")
    print(f"  Run dir    : {run_dir}")
    if not args.dry_run:
        print(f"  Log        : {log_file}")
        print(f"  Manifest   : {manifest_path}")

    if pipeline_ok:
        print(_c("\n  Pipeline completed successfully.", _GREEN + _BOLD))
    else:
        print(_c("\n  Pipeline finished with errors.", _RED + _BOLD))
        sys.exit(1)


if __name__ == "__main__":
    main()
