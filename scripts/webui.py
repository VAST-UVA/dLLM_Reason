"""Interactive Web UI for dLLM-Reason.

Provides a browser-based dashboard for:
1. Interactive generation with strategy switching
2. DAG visualization (real-time unmasking progression)
3. Benchmark results viewer
4. Strategy comparison side-by-side

Usage:
    python scripts/webui.py --port 7860
    python scripts/webui.py --model_id checkpoints/llada-instruct --port 7860
    dllm-webui --port 7860

Requires: pip install "dllm-reason[serve]"  (fastapi, uvicorn)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="dLLM-Reason Web UI", version="1.4.0")

# Global model reference
_model = None
_model_id = ""

# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

AVAILABLE_STRATEGIES = [
    "confidence", "random", "entropy", "semi_ar",
    "maskgit_cosine", "critical_token_first", "curriculum",
    "linear", "cot", "skeleton", "bidirectional", "answer_first",
    "adaptive_dynamic",
]


class GenerateRequest(BaseModel):
    prompt: str
    strategy: str = "confidence"
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    num_steps: int = Field(default=128, ge=1, le=1024)
    block_length: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    cfg_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    remasking: str = "low_confidence"
    record_trajectory: bool = False


class CompareRequest(BaseModel):
    prompt: str
    strategies: list[str]
    system_prompt: str | None = None
    max_new_tokens: int = 128
    num_steps: int = 128
    block_length: int = 32
    temperature: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler builder (same as serve.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(strategy: str, gen_len: int, block_length: int, device):
    """Build a scheduler from strategy name."""
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
    from dllm_reason.scheduler.random_scheduler import RandomScheduler
    from dllm_reason.scheduler.linear_scheduler import LinearScheduler
    from dllm_reason.scheduler.entropy_scheduler import EntropyScheduler
    from dllm_reason.scheduler.semi_ar_scheduler import SemiAutoregressiveScheduler
    from dllm_reason.scheduler.maskgit_scheduler import MaskGITCosineScheduler
    from dllm_reason.scheduler.critical_token_scheduler import CriticalTokenFirstScheduler
    from dllm_reason.scheduler.curriculum_scheduler import CurriculumScheduler
    from dllm_reason.scheduler.adaptive_dynamic_scheduler import AdaptiveDynamicScheduler

    schedulers = {
        "confidence": lambda: ConfidenceScheduler(),
        "random": lambda: RandomScheduler(),
        "linear": lambda: LinearScheduler(),
        "entropy": lambda: EntropyScheduler(),
        "semi_ar": lambda: SemiAutoregressiveScheduler(block_size=block_length),
        "maskgit_cosine": lambda: MaskGITCosineScheduler(),
        "critical_token_first": lambda: CriticalTokenFirstScheduler(),
        "curriculum": lambda: CurriculumScheduler(),
        "adaptive_dynamic": lambda: AdaptiveDynamicScheduler(),
    }

    if strategy in schedulers:
        return schedulers[strategy]()

    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag, answer_first_dag,
    )

    if strategy == "cot":
        dag = chain_of_thought_dag(gen_len, num_steps=4, device=device)
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "skeleton":
        dag = skeleton_then_detail_dag(
            gen_len, list(range(0, gen_len, 3)), list(range(1, gen_len, 3)), device=device,
        )
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "bidirectional":
        dag = bidirectional_dag(gen_len, num_segments=4, device=device)
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "answer_first":
        dag = answer_first_dag(
            gen_len, list(range(int(gen_len * 0.8), gen_len)), device=device,
        )
        return DAGScheduler(dag, sub_strategy="confidence_topk")

    raise ValueError(f"Unknown strategy: {strategy}")


# ──────────────────────────────────────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the main Web UI page."""
    return MAIN_HTML


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": _model_id,
        "device": str(_model.device if _model else "none"),
        "strategies": AVAILABLE_STRATEGIES,
    }


@app.get("/api/strategies")
def strategies():
    return {"strategies": AVAILABLE_STRATEGIES}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    if req.strategy not in AVAILABLE_STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy}")

    scheduler = build_scheduler(req.strategy, req.max_new_tokens, req.block_length, _model.device)

    t0 = time.time()
    result = _model.generate(
        prompt=req.prompt,
        generation_len=req.max_new_tokens,
        block_length=req.block_length,
        scheduler=scheduler,
        num_steps=req.num_steps,
        temperature=req.temperature,
        cfg_scale=req.cfg_scale,
        remasking=req.remasking,
        system_prompt=req.system_prompt,
        record_trajectory=req.record_trajectory,
    )
    elapsed = time.time() - t0

    if req.record_trajectory:
        text, trajectory = result
    else:
        text = result
        trajectory = []

    return {
        "text": text,
        "strategy": req.strategy,
        "elapsed_seconds": round(elapsed, 3),
        "num_tokens": len(text.split()),
        "trajectory": trajectory,
    }


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Generate with multiple strategies for side-by-side comparison."""
    if _model is None:
        raise HTTPException(500, "Model not loaded")

    results = []
    for strategy in req.strategies:
        if strategy not in AVAILABLE_STRATEGIES:
            results.append({"strategy": strategy, "error": f"Unknown strategy"})
            continue

        scheduler = build_scheduler(strategy, req.max_new_tokens, req.block_length, _model.device)

        t0 = time.time()
        text = _model.generate(
            prompt=req.prompt,
            generation_len=req.max_new_tokens,
            block_length=req.block_length,
            scheduler=scheduler,
            num_steps=req.num_steps,
            temperature=req.temperature,
            cfg_scale=req.cfg_scale,
            system_prompt=req.system_prompt,
        )
        elapsed = time.time() - t0

        results.append({
            "strategy": strategy,
            "text": text,
            "elapsed_seconds": round(elapsed, 3),
            "num_tokens": len(text.split()),
        })

    return {"prompt": req.prompt, "results": results}


@app.get("/api/results")
def list_results():
    """List available benchmark result files."""
    results_dir = Path("results")
    if not results_dir.exists():
        return {"results": []}

    files = []
    for f in sorted(results_dir.glob("**/summary.json")):
        files.append(str(f.relative_to(results_dir)))
    return {"results": files}


@app.get("/api/results/{path:path}")
def get_result(path: str):
    """Read a specific result file.

    Security: the requested path must resolve inside the ``results``
    directory. Any attempt to escape it via ``..`` or absolute paths is
    rejected (bug C15 — path-traversal).
    """
    results_root = Path("results").resolve()
    try:
        result_path = (results_root / path).resolve()
    except (OSError, ValueError):
        raise HTTPException(400, "Invalid path")
    # Ensure the resolved path is inside results_root
    try:
        result_path.relative_to(results_root)
    except ValueError:
        raise HTTPException(403, "Forbidden")
    if not result_path.exists() or not result_path.is_file():
        raise HTTPException(404, f"Not found: {path}")
    with open(result_path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Embedded HTML (single-file UI — no external dependencies)
# ──────────────────────────────────────────────────────────────────────────────

MAIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dLLM-Reason Web UI</title>
<style>
:root { --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a; --text: #e0e0e0;
        --accent: #6366f1; --accent2: #818cf8; --green: #22c55e; --red: #ef4444; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
header { display: flex; justify-content: space-between; align-items: center;
         padding: 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
header h1 { font-size: 1.5rem; color: var(--accent2); }
.status { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.ok { background: var(--green); }
.status-dot.err { background: var(--red); }

.tabs { display: flex; gap: 4px; margin-bottom: 20px; }
.tab { padding: 8px 20px; border: 1px solid var(--border); border-radius: 6px 6px 0 0;
       background: transparent; color: var(--text); cursor: pointer; font-size: 0.9rem; }
.tab.active { background: var(--card); border-bottom-color: var(--card); color: var(--accent2); }

.panel { display: none; }
.panel.active { display: block; }

.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
        padding: 20px; margin-bottom: 16px; }
label { display: block; margin-bottom: 4px; font-size: 0.85rem; color: #999; }
textarea, input, select { width: 100%; padding: 10px; border: 1px solid var(--border);
                          border-radius: 6px; background: var(--bg); color: var(--text);
                          font-family: inherit; font-size: 0.9rem; margin-bottom: 12px; }
textarea { min-height: 100px; resize: vertical; }
button { padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer;
         font-size: 0.9rem; font-weight: 600; }
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: var(--accent2); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }

.output { white-space: pre-wrap; font-family: 'Fira Code', monospace; font-size: 0.85rem;
          padding: 16px; background: var(--bg); border-radius: 6px; min-height: 60px;
          border: 1px solid var(--border); }
.meta { font-size: 0.8rem; color: #888; margin-top: 8px; }

.compare-grid { display: grid; gap: 16px; }
.compare-item { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }
.compare-item h4 { color: var(--accent2); margin-bottom: 8px; }

.trajectory-step { padding: 8px; margin: 4px 0; background: var(--bg); border-radius: 4px;
                   font-family: monospace; font-size: 0.8rem; border-left: 3px solid var(--accent); }

.results-table { width: 100%; border-collapse: collapse; }
.results-table th, .results-table td { padding: 8px 12px; text-align: left;
                                        border-bottom: 1px solid var(--border); }
.results-table th { color: var(--accent2); font-weight: 600; }
.best { color: var(--green); font-weight: 700; }

.checkbox-group { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.checkbox-group label { display: flex; align-items: center; gap: 4px; cursor: pointer;
                        padding: 4px 10px; border: 1px solid var(--border); border-radius: 4px;
                        font-size: 0.85rem; color: var(--text); }
.checkbox-group label:has(input:checked) { border-color: var(--accent); background: rgba(99,102,241,0.15); }
.checkbox-group input[type="checkbox"] { width: auto; margin: 0; }

.loading { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border);
           border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>dLLM-Reason</h1>
    <div class="status">
      <span class="status-dot" id="statusDot"></span>
      <span id="statusText">Connecting...</span>
    </div>
  </header>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('generate')">Generate</button>
    <button class="tab" onclick="switchTab('compare')">Compare</button>
    <button class="tab" onclick="switchTab('trajectory')">Trajectory</button>
    <button class="tab" onclick="switchTab('results')">Results</button>
  </div>

  <!-- Generate Tab -->
  <div id="generate" class="panel active">
    <div class="card">
      <label>Prompt</label>
      <textarea id="prompt" placeholder="Enter your prompt here...">Solve step by step: What is 15 * 23?</textarea>
      <div class="grid-3">
        <div>
          <label>Strategy</label>
          <select id="strategy"></select>
        </div>
        <div>
          <label>Max Tokens</label>
          <input type="number" id="maxTokens" value="128" min="1" max="2048">
        </div>
        <div>
          <label>Temperature</label>
          <input type="number" id="temperature" value="0.0" min="0" max="2" step="0.1">
        </div>
      </div>
      <div class="grid-3">
        <div>
          <label>Num Steps</label>
          <input type="number" id="numSteps" value="128" min="1" max="1024">
        </div>
        <div>
          <label>Block Length</label>
          <input type="number" id="blockLength" value="32" min="1" max="512">
        </div>
        <div>
          <label>System Prompt</label>
          <input type="text" id="systemPrompt" placeholder="(optional)">
        </div>
      </div>
      <button class="btn-primary" id="generateBtn" onclick="doGenerate()">Generate</button>
    </div>
    <div class="card">
      <label>Output</label>
      <div class="output" id="output">Press Generate to start...</div>
      <div class="meta" id="outputMeta"></div>
    </div>
  </div>

  <!-- Compare Tab -->
  <div id="compare" class="panel">
    <div class="card">
      <label>Prompt</label>
      <textarea id="comparePrompt">Solve step by step: What is 15 * 23?</textarea>
      <label>Select strategies to compare:</label>
      <div class="checkbox-group" id="compareStrategies"></div>
      <button class="btn-primary" id="compareBtn" onclick="doCompare()">Compare</button>
    </div>
    <div class="compare-grid" id="compareResults"></div>
  </div>

  <!-- Trajectory Tab -->
  <div id="trajectory" class="panel">
    <div class="card">
      <label>Prompt</label>
      <textarea id="trajPrompt">What is 7 * 8?</textarea>
      <div class="grid-3">
        <div>
          <label>Strategy</label>
          <select id="trajStrategy"></select>
        </div>
        <div>
          <label>Max Tokens</label>
          <input type="number" id="trajMaxTokens" value="64" min="1" max="512">
        </div>
        <div>
          <label>Num Steps</label>
          <input type="number" id="trajNumSteps" value="64" min="1" max="512">
        </div>
      </div>
      <button class="btn-primary" id="trajBtn" onclick="doTrajectory()">Generate with Trajectory</button>
    </div>
    <div class="card">
      <label>Final Output</label>
      <div class="output" id="trajOutput">Press Generate to start...</div>
      <label style="margin-top:12px">Unmasking Trajectory (step by step)</label>
      <div id="trajSteps"></div>
    </div>
  </div>

  <!-- Results Tab -->
  <div id="results" class="panel">
    <div class="card">
      <label>Available Result Files</label>
      <select id="resultFile" onchange="loadResult()">
        <option value="">Select a result file...</option>
      </select>
    </div>
    <div class="card">
      <div id="resultsContent">Select a result file to view.</div>
    </div>
  </div>
</div>

<script>
const API = '';
let strategies = [];

async function init() {
  try {
    const resp = await fetch(API + '/api/health');
    const data = await resp.json();
    document.getElementById('statusDot').className = 'status-dot ok';
    document.getElementById('statusText').textContent =
      data.model + ' on ' + data.device;
    strategies = data.strategies;
    populateStrategies();
    loadResultFiles();
  } catch(e) {
    document.getElementById('statusDot').className = 'status-dot err';
    document.getElementById('statusText').textContent = 'Disconnected';
  }
}

function populateStrategies() {
  const selects = ['strategy', 'trajStrategy'];
  selects.forEach(id => {
    const el = document.getElementById(id);
    el.innerHTML = strategies.map(s =>
      '<option value="' + s + '">' + s + '</option>'
    ).join('');
  });
  const cg = document.getElementById('compareStrategies');
  cg.innerHTML = strategies.map(s =>
    '<label><input type="checkbox" value="' + s + '"' +
    (s === 'confidence' || s === 'adaptive_dynamic' ? ' checked' : '') +
    '>' + s + '</label>'
  ).join('');
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(name).classList.add('active');
}

async function doGenerate() {
  const btn = document.getElementById('generateBtn');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Generating...';
  document.getElementById('output').textContent = 'Generating...';
  document.getElementById('outputMeta').textContent = '';

  try {
    const resp = await fetch(API + '/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: document.getElementById('prompt').value,
        strategy: document.getElementById('strategy').value,
        max_new_tokens: parseInt(document.getElementById('maxTokens').value),
        num_steps: parseInt(document.getElementById('numSteps').value),
        block_length: parseInt(document.getElementById('blockLength').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        system_prompt: document.getElementById('systemPrompt').value || null,
      })
    });
    const data = await resp.json();
    document.getElementById('output').textContent = data.text || data.detail || 'No output';
    document.getElementById('outputMeta').textContent =
      'Strategy: ' + data.strategy + ' | ' +
      data.num_tokens + ' tokens | ' +
      data.elapsed_seconds + 's';
  } catch(e) {
    document.getElementById('output').textContent = 'Error: ' + e.message;
  }
  btn.disabled = false; btn.textContent = 'Generate';
}

async function doCompare() {
  const btn = document.getElementById('compareBtn');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Comparing...';
  const container = document.getElementById('compareResults');
  container.innerHTML = '<div class="card">Generating...</div>';

  const selected = Array.from(
    document.querySelectorAll('#compareStrategies input:checked')
  ).map(c => c.value);

  if (selected.length === 0) {
    container.innerHTML = '<div class="card">Select at least one strategy.</div>';
    btn.disabled = false; btn.textContent = 'Compare';
    return;
  }

  try {
    const resp = await fetch(API + '/api/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: document.getElementById('comparePrompt').value,
        strategies: selected,
        max_new_tokens: 128, num_steps: 128, block_length: 32, temperature: 0.0,
      })
    });
    const data = await resp.json();
    // Escape all user / model generated fields to prevent HTML injection
    // from prompts or model outputs rendering executable markup (bug C14).
    container.innerHTML = data.results.map(r =>
      '<div class="compare-item"><h4>' + escapeHtml(String(r.strategy || '')) + '</h4>' +
      '<div class="output">' + escapeHtml(String(r.text || r.error || 'No output')) + '</div>' +
      '<div class="meta">' + (r.elapsed_seconds || 0) + 's | ' +
      (r.num_tokens || 0) + ' tokens</div></div>'
    ).join('');
    container.style.gridTemplateColumns = 'repeat(' + Math.min(selected.length, 3) + ', 1fr)';
  } catch(e) {
    container.innerHTML = '<div class="card">Error: ' + e.message + '</div>';
  }
  btn.disabled = false; btn.textContent = 'Compare';
}

async function doTrajectory() {
  const btn = document.getElementById('trajBtn');
  btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Generating...';
  document.getElementById('trajOutput').textContent = 'Generating...';
  document.getElementById('trajSteps').innerHTML = '';

  try {
    const resp = await fetch(API + '/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: document.getElementById('trajPrompt').value,
        strategy: document.getElementById('trajStrategy').value,
        max_new_tokens: parseInt(document.getElementById('trajMaxTokens').value),
        num_steps: parseInt(document.getElementById('trajNumSteps').value),
        block_length: 32, temperature: 0.0,
        record_trajectory: true,
      })
    });
    const data = await resp.json();
    document.getElementById('trajOutput').textContent = data.text || 'No output';

    const steps = data.trajectory || [];
    const container = document.getElementById('trajSteps');
    if (steps.length === 0) {
      container.innerHTML = '<div class="meta">No trajectory recorded.</div>';
    } else {
      // Show every N-th step for readability
      const maxShow = 20;
      const stride = Math.max(1, Math.floor(steps.length / maxShow));
      let html = '';
      for (let i = 0; i < steps.length; i += stride) {
        html += '<div class="trajectory-step"><strong>Step ' + i + ':</strong> ' +
                escapeHtml(steps[i]) + '</div>';
      }
      html += '<div class="trajectory-step"><strong>Final:</strong> ' +
              escapeHtml(steps[steps.length - 1]) + '</div>';
      container.innerHTML = html;
    }
  } catch(e) {
    document.getElementById('trajOutput').textContent = 'Error: ' + e.message;
  }
  btn.disabled = false; btn.textContent = 'Generate with Trajectory';
}

async function loadResultFiles() {
  try {
    const resp = await fetch(API + '/api/results');
    const data = await resp.json();
    const sel = document.getElementById('resultFile');
    data.results.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f; opt.textContent = f;
      sel.appendChild(opt);
    });
  } catch(e) {}
}

async function loadResult() {
  const file = document.getElementById('resultFile').value;
  if (!file) return;
  try {
    const resp = await fetch(API + '/api/results/' + file);
    const data = await resp.json();
    renderResults(data);
  } catch(e) {
    document.getElementById('resultsContent').textContent = 'Error loading: ' + e.message;
  }
}

function renderResults(data) {
  const container = document.getElementById('resultsContent');
  // data is {strategy: {benchmark: {metric: value}}}
  const strats = Object.keys(data);
  if (strats.length === 0) { container.textContent = 'No data.'; return; }

  const benchmarks = new Set();
  strats.forEach(s => {
    if (typeof data[s] === 'object') Object.keys(data[s]).forEach(b => benchmarks.add(b));
  });
  const bms = Array.from(benchmarks).sort();

  let html = '<table class="results-table"><tr><th>Strategy</th>';
  bms.forEach(b => { html += '<th>' + b + '</th>'; });
  html += '</tr>';

  // Find best per benchmark
  const best = {};
  bms.forEach(b => {
    let maxVal = -Infinity;
    strats.forEach(s => {
      const r = data[s]?.[b];
      if (r) {
        const v = r.accuracy || r['pass@1'] || r.exact_match || 0;
        if (v > maxVal) maxVal = v;
      }
    });
    best[b] = maxVal;
  });

  strats.forEach(s => {
    html += '<tr><td><strong>' + s + '</strong></td>';
    bms.forEach(b => {
      const r = data[s]?.[b];
      if (r) {
        const v = r.accuracy || r['pass@1'] || r.exact_match || 0;
        const cls = Math.abs(v - best[b]) < 0.0001 ? ' class="best"' : '';
        html += '<td' + cls + '>' + (v * 100).toFixed(1) + '</td>';
      } else {
        html += '<td>--</td>';
      }
    });
    html += '</tr>';
  });
  html += '</table>';
  container.innerHTML = html;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

init();
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global _model, _model_id

    parser = argparse.ArgumentParser(description="dLLM-Reason Interactive Web UI")
    parser.add_argument("--model_id", type=str, default="checkpoints/llada-instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--quantize", type=str, default=None, choices=["4bit", "8bit"])
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no_model", action="store_true",
                        help="Start UI without loading model (results viewer only)")
    args = parser.parse_args()

    if not args.no_model:
        import torch

        _model_id = args.model_id
        dtype_map = {
            "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
        }

        quant_config = None
        if args.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype_map[args.torch_dtype],
                bnb_4bit_quant_type="nf4",
            )
        elif args.quantize == "8bit":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        from dllm_reason.models.llada import LLaDAWrapper
        print(f"Loading model: {args.model_id}")
        _model = LLaDAWrapper(
            model_id=args.model_id,
            torch_dtype=dtype_map[args.torch_dtype],
            device_map="auto",
            quantization_config=quant_config,
        )
        print(f"Model loaded on {_model.device}")
    else:
        print("Starting in results-viewer-only mode (no model loaded)")

    print(f"Web UI: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
