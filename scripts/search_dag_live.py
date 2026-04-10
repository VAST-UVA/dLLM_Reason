"""DAG structure search with real-time web visualization.

Opens a local web page (http://localhost:<port>) that streams live search
progress via Server-Sent Events.  No external JS build step required —
Chart.js is loaded from CDN.

Usage:
    # Evolutionary search, live web view on port 8765
    python scripts/search_dag_live.py \\
        --model llada --checkpoint checkpoints/llada-instruct \\
        --dataset gsm8k --method evolutionary --budget 100

    # Greedy search, custom port
    python scripts/search_dag_live.py \\
        --method greedy --budget 50 --port 8765

    # Optional matplotlib save at the end (no live window)
    python scripts/search_dag_live.py --method greedy --budget 50 --save_plot
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import threading
import time
import webbrowser
from math import inf
from pathlib import Path

import torch

# ── HTML dashboard ─────────────────────────────────────────────────────────────
LIVE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DAG Search — Live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Segoe UI", system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
  header { padding: 16px 24px; background: #1e293b; border-bottom: 1px solid #334155;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.3rem; font-weight: 700; color: #38bdf8; }
  #status-badge { padding: 4px 12px; border-radius: 99px; font-size: .8rem; font-weight: 600; }
  .running  { background: #0ea5e9; color: #fff; }
  .done     { background: #22c55e; color: #fff; }
  .waiting  { background: #64748b; color: #fff; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; }
  .card h2 { font-size: .95rem; color: #94a3b8; margin-bottom: 14px; text-transform: uppercase; letter-spacing: .06em; }
  #stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat { background: #0f172a; border-radius: 8px; padding: 12px 16px; }
  .stat .label { font-size: .75rem; color: #64748b; margin-bottom: 4px; }
  .stat .value { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
  .stat .value.green { color: #4ade80; }
  #chart-wrap { position: relative; height: 280px; }
  #dag-wrap  { position: relative; text-align: center; }
  #dag-img   { max-width: 100%; border-radius: 8px; background: #0f172a; min-height: 120px; }
  #dag-placeholder { color: #475569; font-size: .9rem; padding: 40px 0; }
  #log-box { height: 180px; overflow-y: auto; font-family: monospace;
             font-size: .78rem; color: #94a3b8; background: #0f172a;
             border-radius: 8px; padding: 10px 14px; }
  #log-box p { margin: 0; line-height: 1.6; }
  #log-box p.best { color: #4ade80; }
</style>
</head>
<body>
<header>
  <h1>DAG Search &mdash; Live</h1>
  <span id="status-badge" class="waiting">Waiting…</span>
</header>
<main>
  <!-- Stats -->
  <div class="card">
    <h2>Progress</h2>
    <div id="stats-grid">
      <div class="stat"><div class="label">Step</div><div class="value" id="s-step">—</div></div>
      <div class="stat"><div class="label">Budget</div><div class="value" id="s-budget">—</div></div>
      <div class="stat"><div class="label">Current Fitness</div><div class="value" id="s-cur">—</div></div>
      <div class="stat"><div class="label">Best Fitness</div><div class="value green" id="s-best">—</div></div>
      <div class="stat"><div class="label">Edges</div><div class="value" id="s-edges">—</div></div>
      <div class="stat"><div class="label">Generation</div><div class="value" id="s-gen">—</div></div>
    </div>
  </div>

  <!-- Log -->
  <div class="card">
    <h2>Event Log</h2>
    <div id="log-box"></div>
  </div>

  <!-- Fitness chart -->
  <div class="card">
    <h2>Fitness Curve</h2>
    <div id="chart-wrap"><canvas id="fitness-chart"></canvas></div>
  </div>

  <!-- DAG heatmap -->
  <div class="card">
    <h2>Best DAG (adjacency heatmap)</h2>
    <div id="dag-wrap">
      <div id="dag-placeholder">Waiting for first heatmap update…</div>
      <img id="dag-img" src="" alt="" style="display:none">
    </div>
  </div>
</main>

<script>
const cfg = {steps: [], fitnesses: [], bestFitnesses: []};

// Chart.js setup
const ctx = document.getElementById('fitness-chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Fitness', data: [], borderColor: '#38bdf8', borderWidth: 1.5,
        pointRadius: 0, tension: 0.2, fill: false },
      { label: 'Best',    data: [], borderColor: '#4ade80', borderWidth: 2,
        borderDash: [5,3], pointRadius: 0, tension: 0, fill: false },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { labels: { color: '#94a3b8' } } },
    scales: {
      x: { ticks: { color: '#64748b', maxTicksLimit: 10 }, grid: { color: '#1e293b' } },
      y: { ticks: { color: '#64748b' }, grid: { color: '#334155' }, min: 0, max: 1 }
    }
  }
});

function addLog(msg, cls='') {
  const box = document.getElementById('log-box');
  const p = document.createElement('p');
  if (cls) p.className = cls;
  p.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  box.appendChild(p);
  box.scrollTop = box.scrollHeight;
  // Keep at most 200 lines
  while (box.children.length > 200) box.removeChild(box.firstChild);
}

function update(d) {
  document.getElementById('s-step').textContent  = d.step;
  document.getElementById('s-budget').textContent= d.budget ?? '—';
  document.getElementById('s-cur').textContent   = d.fitness.toFixed(4);
  document.getElementById('s-best').textContent  = d.best_fitness.toFixed(4);
  document.getElementById('s-edges').textContent = d.edges ?? '—';
  document.getElementById('s-gen').textContent   = d.generation ?? '—';

  chart.data.labels.push(d.step);
  chart.data.datasets[0].data.push(d.fitness);
  chart.data.datasets[1].data.push(d.best_fitness);
  chart.update('none');

  const logMsg = `step=${d.step}  fit=${d.fitness.toFixed(4)}  best=${d.best_fitness.toFixed(4)}  edges=${d.edges??'?'}`;
  addLog(logMsg, d.is_new_best ? 'best' : '');
}

function showHeatmap(b64) {
  const img = document.getElementById('dag-img');
  const ph  = document.getElementById('dag-placeholder');
  img.src = 'data:image/png;base64,' + b64;
  img.style.display = 'block';
  ph.style.display  = 'none';
}

// SSE
const sse = new EventSource('/stream');
const badge = document.getElementById('status-badge');

sse.onopen = () => {
  badge.textContent = 'Running…';
  badge.className = 'running';
  addLog('Connected to search stream.');
};

sse.onmessage = (e) => {
  const d = JSON.parse(e.data);
  if (d.heatmap_b64) { showHeatmap(d.heatmap_b64); return; }
  if (d.done) {
    badge.textContent = 'Done ✓';
    badge.className = 'done';
    addLog('Search complete!  best=' + d.best_fitness.toFixed(4), 'best');
    sse.close();
    return;
  }
  update(d);
};

sse.onerror = () => {
  badge.textContent = 'Disconnected';
  badge.className = 'waiting';
  addLog('Stream disconnected.');
};
</script>
</body>
</html>
"""


# ── Progress queue (thread-safe bridge between search and HTTP) ────────────────

_progress_q: queue.Queue = queue.Queue()


# ── Heatmap rendering ──────────────────────────────────────────────────────────

def _dag_to_b64_png(dag, max_size: int = 64) -> str:
    """Render DAG adjacency matrix as a base64-encoded PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import numpy as np

        n = min(dag.seq_len, max_size)
        adj = dag.adjacency[:n, :n].float().cpu().numpy()

        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.imshow(adj, cmap="Blues", aspect="auto", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"edges={dag.num_edges()}  (first {n} pos)", fontsize=8)
        ax.set_xlabel("dst", fontsize=7)
        ax.set_ylabel("src", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


# ── FastAPI app ────────────────────────────────────────────────────────────────

def build_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="DAG Search Live")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return LIVE_HTML

    @app.get("/stream")
    def stream():
        def event_generator():
            while True:
                try:
                    item = _progress_q.get(timeout=30)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue

                if item is None:          # sentinel — search done
                    yield "data: {\"done\": true, \"best_fitness\": 0}\n\n"
                    break

                # Large heatmap blobs get their own SSE message type
                if "heatmap_b64" in item:
                    yield f"data: {json.dumps({'heatmap_b64': item['heatmap_b64']})}\n\n"
                else:
                    yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# ── eval_fn wrapper ────────────────────────────────────────────────────────────

class LiveTracker:
    """Wraps eval_fn calls; pushes progress events to the SSE queue."""

    def __init__(self, budget: int, plot_interval: int = 5):
        self.budget = budget
        self.plot_interval = plot_interval
        self.step = 0
        self.best_fitness = -inf
        self.best_dag = None
        self.generation = 0

    def wrap(self, eval_fn):
        def wrapped(model, dag):
            fitness = eval_fn(model, dag)
            self._update(fitness, dag)
            return fitness
        return wrapped

    def _update(self, fitness: float, dag) -> None:
        self.step += 1
        is_new_best = fitness > self.best_fitness
        if is_new_best:
            self.best_fitness = fitness
            self.best_dag = dag

        # Terminal progress
        filled = int(20 * self.step / max(self.budget, 1))
        bar = "█" * filled + "░" * (20 - filled)
        tag = "  ★ NEW BEST" if is_new_best else ""
        print(
            f"\r[{bar}] {self.step:>4}/{self.budget}"
            f"  cur={fitness:.4f}  best={self.best_fitness:.4f}"
            f"  edges={dag.num_edges():>4}{tag}",
            end="", flush=True,
        )

        # Push stats event
        _progress_q.put({
            "step": self.step,
            "budget": self.budget,
            "fitness": fitness,
            "best_fitness": self.best_fitness,
            "edges": dag.num_edges(),
            "generation": self.generation,
            "is_new_best": is_new_best,
            "done": False,
        })

        # Push heatmap on new best OR every plot_interval steps
        if is_new_best or self.step % self.plot_interval == 0:
            b64 = _dag_to_b64_png(dag if is_new_best else (self.best_dag or dag))
            if b64:
                _progress_q.put({"heatmap_b64": b64})

    def finalize(self, best_fitness: float) -> None:
        print()  # newline after progress bar
        # Final heatmap of best DAG
        if self.best_dag is not None:
            b64 = _dag_to_b64_png(self.best_dag)
            if b64:
                _progress_q.put({"heatmap_b64": b64})
        # Done sentinel
        _progress_q.put({"done": True, "best_fitness": best_fitness})
        _progress_q.put(None)   # causes generator to close


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DAG structure search with real-time web visualization"
    )

    p.add_argument("--model", default="mdlm",
                   choices=["mdlm", "sedd", "d3pm", "llada"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "math", "arc", "prontoqa"])
    p.add_argument("--tokenizer", default="gpt2")

    p.add_argument("--method", default="evolutionary",
                   choices=["greedy", "evolutionary", "rl_policy", "differentiable"])
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--population_size", type=int, default=20)
    p.add_argument("--mutation_rate", type=float, default=0.3)

    p.add_argument("--init_dag", default=None,
                   choices=["cot", "skeleton", "linear",
                            "bidirectional", "answer_first", "interleaved",
                            "random_low", "random_high"],
                   help="Single explicit seed DAG (legacy; superseded by --init_templates)")
    p.add_argument("--init_cot_steps", type=int, default=4,
                   help="num_steps for the cot template when --init_dag=cot")
    p.add_argument(
        "--init_templates", nargs="*",
        metavar="TEMPLATE",
        default=None,
        help=(
            "Template names to use as search initialization seeds.  "
            "For evolutionary: added to initial population.  "
            "For greedy: evaluated first, best one chosen as start.  "
            "Available: cot skeleton bidirectional answer_first interleaved "
            "linear random_low random_high.  "
            "Pass with no names (--init_templates) to use the default set, "
            "or list specific names.  "
            "Overrides --init_dag."
        ),
    )

    p.add_argument("--fitness", default="accuracy",
                   choices=["accuracy", "perplexity", "combined"])
    p.add_argument("--fitness_samples", type=int, default=50)
    p.add_argument("--num_steps", type=int, default=32)

    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=512)

    p.add_argument("--output_dir", default="results/dag_search_live")

    # Web server
    p.add_argument("--port", type=int, default=8765,
                   help="Port for the live web dashboard (default: 8765)")
    p.add_argument("--no_browser", action="store_true",
                   help="Do not auto-open the browser tab")
    p.add_argument("--plot_interval", type=int, default=5,
                   help="Refresh heatmap every N evaluations (default: 5)")

    # Post-search static plot
    p.add_argument("--save_plot", action="store_true",
                   help="Save search_progress.png after search finishes")

    return p.parse_args()


# ── Helpers (same as search_dag.py) ───────────────────────────────────────────

def build_initial_dag(name, seq_len, cot_steps, device):
    """Build a single named template (legacy helper for --init_dag)."""
    from dllm_reason.graph.templates import build_template
    from dllm_reason.graph.dag import TokenDAG
    if name is None:
        return None
    if name == "cot":
        # cot_steps is the only non-default parameter
        from dllm_reason.graph.templates import chain_of_thought_dag
        return chain_of_thought_dag(seq_len, cot_steps, device=device)
    return build_template(name, seq_len, device=device)


# ── Search thread ──────────────────────────────────────────────────────────────

def run_search(args, tracker: LiveTracker):
    """Runs in a background thread.  Pushes events via tracker → _progress_q."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer + data
    from dllm_reason.data.tokenizer import get_tokenizer
    from dllm_reason.data.reasoning_datasets import load_reasoning_dataset

    tokenizer_name = args.checkpoint if args.model == "llada" else args.tokenizer
    tokenizer = get_tokenizer(tokenizer_name, add_mask_token=True)
    dataset = load_reasoning_dataset(args.dataset, split="train")
    eval_dataset = dataset[:args.fitness_samples * 2]

    # Model
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

    # Fitness function
    from dllm_reason.data.reasoning_datasets import ReasoningDataset
    from dllm_reason.data.collator import DiffusionCollator
    from torch.utils.data import DataLoader
    from dllm_reason.eval.metrics import extract_number

    eval_ds = ReasoningDataset(eval_dataset, tokenizer, max_seq_len=args.max_seq_len)
    eval_loader = DataLoader(
        eval_ds, batch_size=8, shuffle=True,
        collate_fn=DiffusionCollator(mask_token_id=model.mask_token_id),
    )

    def answer_extractor(token_ids):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        num = extract_number(text)
        return num if num else text.strip()

    if args.fitness == "accuracy":
        from dllm_reason.search.fitness import accuracy_fitness
        def eval_fn(model, dag):
            return accuracy_fitness(model, dag, eval_loader, answer_extractor,
                                    max_samples=args.fitness_samples,
                                    num_steps=args.num_steps)
    elif args.fitness == "perplexity":
        from dllm_reason.search.fitness import perplexity_fitness
        def eval_fn(model, dag):
            return perplexity_fitness(model, dag, eval_loader,
                                      max_samples=args.fitness_samples)
    else:
        from dllm_reason.search.fitness import combined_fitness
        def eval_fn(model, dag):
            return combined_fitness(model, dag, eval_loader, answer_extractor,
                                    max_samples=args.fitness_samples)

    # Wrap with live tracker
    live_eval_fn = tracker.wrap(eval_fn)

    # ── Resolve init templates / seed DAG ────────────────────────────────
    # --init_templates takes priority; --init_dag is the legacy single-DAG path.
    # --init_templates with no names → use searcher default set (pass None)
    # --init_templates with names   → use those names (pass list)
    # neither flag set              → init_templates=None (searcher decides)
    if args.init_templates is not None:
        # user passed --init_templates (possibly with zero or more names)
        init_templates = args.init_templates if args.init_templates else None
        initial_dag = None
        if init_templates:
            print(f"Template seeds: {init_templates}")
        else:
            print("Template seeds: default set")
    else:
        init_templates = None
        initial_dag = build_initial_dag(args.init_dag, args.seq_len,
                                        args.init_cot_steps, device)
        if initial_dag is not None:
            print(f"Single seed DAG: {args.init_dag} ({initial_dag.num_edges()} edges)")

    import dllm_reason.search.greedy, dllm_reason.search.evolutionary
    import dllm_reason.search.rl_policy, dllm_reason.search.differentiable

    if args.method == "greedy":
        from dllm_reason.search.greedy import GreedyEdgeSearch
        searcher = GreedyEdgeSearch(
            initial_dag=initial_dag,
            init_templates=init_templates,
        )
    elif args.method == "evolutionary":
        from dllm_reason.search.evolutionary import EvolutionarySearch
        searcher = EvolutionarySearch(
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            initial_dags=[initial_dag] if initial_dag is not None else [],
            init_templates=init_templates,
        )
    elif args.method == "rl_policy":
        from dllm_reason.search.rl_policy import RLPolicySearch
        searcher = RLPolicySearch(max_seq_len=args.seq_len)
    elif args.method == "differentiable":
        from dllm_reason.search.differentiable import DifferentiableDAGSearch
        searcher = DifferentiableDAGSearch()

    print(f"\nStarting {args.method} search (budget={args.budget})...")
    result = searcher.search(
        model=model,
        eval_fn=live_eval_fn,
        seq_len=args.seq_len,
        budget=args.budget,
    )

    tracker.finalize(result.best_fitness)
    print(f"Search complete! Best fitness: {result.best_fitness:.4f}")

    # Save results
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
    torch.save(result.best_dag.adjacency.cpu(), output_dir / "best_dag_adjacency.pt")
    print(f"Results saved to {output_dir}/")

    if args.save_plot:
        try:
            from dllm_reason.eval.dag_analysis import search_history_plot
            fig = search_history_plot(result.history,
                                       title=f"{args.method} on {args.dataset}")
            fig.savefig(output_dir / "search_progress.png", dpi=150,
                        bbox_inches="tight")
            print(f"Static plot saved to {output_dir}/search_progress.png")
        except Exception as e:
            print(f"[WARNING] Could not save plot: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    tracker = LiveTracker(budget=args.budget, plot_interval=args.plot_interval)

    # Start search in a background thread (so we can serve HTTP immediately)
    search_thread = threading.Thread(
        target=run_search, args=(args, tracker), daemon=True
    )
    search_thread.start()

    # Give search thread a moment to import / load model before opening browser
    if not args.no_browser:
        def _open_after_delay():
            time.sleep(2)
            url = f"http://localhost:{args.port}"
            print(f"\n  ✦ Live dashboard: {url}\n")
            webbrowser.open(url)
        threading.Thread(target=_open_after_delay, daemon=True).start()

    # Start FastAPI server (blocking — keeps the process alive)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is required for the web UI.  "
            "Install it with: pip install uvicorn  (or: pip install 'dllm-reason[serve]')"
        )

    app = build_app()
    print(f"Starting web server on http://localhost:{args.port} ...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

    search_thread.join()


if __name__ == "__main__":
    main()
