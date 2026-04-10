# dLLM-Reason

**DAG-Guided Discrete Diffusion Language Models for Reasoning**

[![PyPI version](https://img.shields.io/pypi/v/dllm-reason)](https://pypi.org/project/dllm-reason/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/BDeMo/dLLM_Reason/actions/workflows/ci.yml/badge.svg)](https://github.com/BDeMo/dLLM_Reason/actions/workflows/ci.yml)

## Overview

dLLM-Reason is a research framework that enhances reasoning in discrete diffusion language models (dLLMs) by controlling the token unmasking order via DAG (Directed Acyclic Graph) topological structures.

**Core idea**: dLLMs generate text by iteratively unmasking tokens. We impose a DAG on unmasking order — edges encode reasoning dependencies — so prerequisite steps are generated before downstream conclusions.

```
Model Layer          Scheduler Layer          DAG Layer
(what to predict) <-> (where to unmask)  <-> (dependency structure)
MDLM|SEDD|D3PM|LLaDA  13 schedulers          TokenDAG / SpanDAG

Episode Pipeline
  prompt → strategy → generate → evaluate → EpisodeStore → SFT / GRPO / DiFFPO
```

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CUDA recommended for model inference)

### From PyPI

```bash
# Core package
pip install dllm-reason

# With DAG Library retrieval (FAISS + sentence-transformers)
pip install "dllm-reason[library]"

# With REST API server
pip install "dllm-reason[serve]"

# Full install (all extras)
pip install "dllm-reason[dev,library,serve]"
```

### From Source (recommended for research)

```bash
git clone https://github.com/BDeMo/dLLM_Reason.git
cd dLLM_Reason
pip install -e ".[dev,library,serve]"
```

### China mirror (HuggingFace models blocked)

```bash
pip install dllm-reason -i https://pypi.tuna.tsinghua.edu.cn/simple
# or set mirror for model downloads:
python scripts/download_models.py --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com
```

### Verify installation

```bash
python -c "import dllm_reason; print(dllm_reason.__version__)"
dllm-pipeline --help
```

### Optional Extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `library` | faiss-cpu, sentence-transformers, scikit-learn | DAG Library retrieval & semantic search |
| `serve` | fastapi, uvicorn, bitsandbytes | REST API server + 4/8-bit quantization |
| `dev` | pytest, pytest-cov, ruff | Testing and linting |

### CLI Commands

After `pip install dllm-reason`, the following commands are available globally:

**Pipeline & Training**

| Command | Script | Description |
|---------|--------|-------------|
| `dllm-pipeline` | `run_pipeline.py` | End-to-end: download → collect → search → learn → eval |
| `dllm-train` | `train.py` | Pretrain / fine-tune a dLLM |
| `dllm-collect` | `collect_episodes.py` | Collect episodes with multiple strategies |
| `dllm-learn` | `learn_from_episodes.py` | SFT / GRPO / DiFFPO / UnmaskRL from episodes |

**Evaluation**

| Command | Script | Description |
|---------|--------|-------------|
| `dllm-eval` | `evaluate.py` | Single-model evaluation on a benchmark |
| `dllm-eval-dags` | `eval_dags.py` | Multi-strategy × multi-benchmark evaluation |
| `dllm-bench-schedulers` | `benchmark_schedulers.py` | Compare all 11 schedulers side-by-side |
| `dllm-episodes` | `inspect_episodes.py` | Inspect / export EpisodeStore |

**DAG Search & Management**

| Command | Script | Description |
|---------|--------|-------------|
| `dllm-search` | `search_dag.py` | DAG structure search (greedy / evolutionary) |
| `dllm-analyze-dag` | `analyze_dag.py` | DAG structural statistics + visualizations |
| `dllm-templates` | `generate_templates.py` | Generate and save named DAG templates |
| `dllm-merge-dags` | `merge_dags.py` | Merge multiple DAGs (union / intersection / weighted) |

**DAG Library**

| Command | Script | Description |
|---------|--------|-------------|
| `dllm-library` | `manage_library.py` | CRUD + retrieval for the DAG Library |
| `dllm-feedback` | `add_feedback.py` | Add benchmark scores / human ratings / Elo |

**Serving & Visualization**

| Command | Script | Description |
|---------|--------|-------------|
| `dllm-serve` | `serve.py` | REST API server with hot-switching strategies |
| `dllm-webui` | `webui.py` | Interactive Web UI dashboard |
| `dllm-viz` | `visualize_dag.py` | Render a DAG to PNG / interactive plot |
| — | `search_dag_live.py` | DAG search with real-time web dashboard |

---

## Quick Start

```bash
# ── Step 0: Install ───────────────────────────────────────────────
pip install "dllm-reason[library,serve]"

# ── Step 1: Download model and datasets ──────────────────────────
python scripts/download_models.py              # → checkpoints/llada-instruct/
python scripts/download_datasets.py            # → datasets/

# China mirror
python scripts/download_models.py   --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com

# ── Step 2: Run the full pipeline (one command) ───────────────────
dllm-pipeline \
    --checkpoint checkpoints/llada-instruct \
    --dataset gsm8k \
    --stages download collect search learn eval \
    --rl_mode grpo

# ── Step 3: Smoke-test evaluation ────────────────────────────────
dllm-eval-dags --dags confidence --benchmarks gsm8k --num_samples 5

# ── Step 4: Start the REST API server ────────────────────────────
dllm-serve --model_id checkpoints/llada-instruct --quantize 4bit
```

---

## Usage

All parameters live in `configs/eval_default.yaml`. CLI flags always override the config.

### Evaluation

```bash
# Default run (LLaDA + confidence)
bash scripts/run_eval.sh

# Direct CLI — pick strategies and benchmarks
dllm-eval-dags \
    --dags confidence entropy adaptive_dynamic cot \
    --benchmarks gsm8k math mbpp humaneval \
    --num_steps 128 --num_samples 100 \
    --save_outputs \
    --output_dir results/my_run

# Per-strategy convenience scripts
bash scripts/runs/confidence.sh       # highest-confidence first (LLaDA default)
bash scripts/runs/random.sh           # uniform random
bash scripts/runs/entropy.sh          # lowest-entropy first
bash scripts/runs/semi_ar.sh          # block-by-block L->R
bash scripts/runs/linear.sh           # strict left-to-right
bash scripts/runs/cot.sh              # Chain-of-Thought DAG
bash scripts/runs/skeleton.sh         # skeleton-then-detail DAG
bash scripts/runs/bidirectional.sh    # both ends toward center
bash scripts/runs/answer_first.sh     # answer region first
bash scripts/runs/all_strategies.sh   # all 13 strategies in one run
bash scripts/runs/full_comparison.sh  # 13 strategies x 10 benchmarks

# All scripts pass extra args through:
bash scripts/runs/cot.sh --benchmarks mbpp humaneval --num_samples 100 --cot_steps 6
```

### Single-Prompt Inference

```bash
python scripts/infer_llada.py \
    --model_id checkpoints/llada-instruct \
    --prompt "What is 7 * 8?" \
    --num_steps 128 --block_length 32 --temperature 0.0
```

### REST API Serving

```bash
# Install serving extras
pip install "dllm-reason[serve]"

# Start server (bfloat16 / port 8000)
dllm-serve --model_id checkpoints/llada-instruct

# With 4-bit quantization (~5 GB VRAM)
dllm-serve --model_id checkpoints/llada-instruct --quantize 4bit

# Generate with any strategy — hot-switchable, no model reload
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 7*8?", "strategy": "adaptive_dynamic", "max_new_tokens": 256}'

# List strategies
curl http://localhost:8000/strategies

# Health check
curl http://localhost:8000/health
```

### Save Per-Sample Outputs

```bash
dllm-eval-dags --dags confidence cot adaptive_dynamic \
    --save_outputs \
    --output_dir results/detailed

# Output files per (benchmark, strategy):
#   {bench}_{dag}_samples.json   — prompt, generated, ground truth, pass/fail
#   {bench}_{dag}_samples.xlsx   — same in spreadsheet format
#   {bench}_{dag}_trajectory.json — per-step unmasking states (with --record_trajectory)
```

### Web UI (Interactive Dashboard)

```bash
# Install serving extras
pip install "dllm-reason[serve]"

# Launch Web UI (loads model + opens browser dashboard)
dllm-webui --model_id checkpoints/llada-instruct --port 7860

# With quantization
dllm-webui --model_id checkpoints/llada-instruct --quantize 4bit

# Results viewer only (no model, no GPU needed)
dllm-webui --no_model --port 7860
```

Features:
- **Generate**: Interactive text generation with strategy selector
- **Compare**: Side-by-side comparison of multiple strategies on the same prompt
- **Trajectory**: Visualize step-by-step unmasking progression
- **Results**: Browse and compare benchmark results (reads `results/` directory)

### DAG Search with Live Web Dashboard

```bash
# Evolutionary search — opens http://localhost:8765 automatically
python scripts/search_dag_live.py \
    --model llada \
    --checkpoint checkpoints/llada-instruct \
    --dataset gsm8k \
    --method evolutionary \       # greedy | evolutionary | rl_policy | differentiable
    --budget 100 \
    --fitness accuracy \          # accuracy | perplexity | combined
    --seq_len 256 \
    --port 8765

# Seed population from templates (new)
python scripts/search_dag_live.py --method evolutionary \
    --init_templates              # default set: cot skeleton bidirectional answer_first

python scripts/search_dag_live.py --method evolutionary \
    --init_templates cot skeleton bidirectional answer_first

# Greedy warm-start: evaluate all templates, start from best
python scripts/search_dag_live.py --method greedy \
    --init_templates cot skeleton answer_first

# 8 available template names:
# cot  skeleton  bidirectional  answer_first  interleaved
# linear  random_low  random_high
```

### Episode Collection

Collect `(prompt, strategy, output, evaluation)` records into a persistent SQLite store.

```bash
# Single prompt
python scripts/collect_episodes.py \
    --model_id checkpoints/llada-instruct \
    --prompt "What is 12 * 15?" \
    --ground_truth "180" \
    --task_type math \
    --strategy cot confidence \   # collect for BOTH strategies in one run
    --db_path episodes/gsm8k.db

# Dataset (JSONL)
python scripts/collect_episodes.py \
    --dataset_path data/gsm8k_test.jsonl \
    --prompt_field problem --answer_field answer \
    --n_samples 500 \
    --strategy confidence cot adaptive_dynamic \
    --eval_mode auto              # auto | manual | none
```

### Learning from Episodes

Fine-tune the model on stored episodes.

```bash
# Print stats only
python scripts/learn_from_episodes.py --db_path episodes/gsm8k.db \
    --model_id none --mode stats

# SFT on correct episodes
python scripts/learn_from_episodes.py \
    --db_path episodes/gsm8k.db \
    --model_id checkpoints/llada-instruct \
    --mode sft --task_type math --dag_aware \
    --epochs 3 --lr 1e-5 \
    --output_dir checkpoints/sft-math

# GRPO (group relative policy optimisation)
python scripts/learn_from_episodes.py \
    --mode grpo --kl_coeff 0.01 \
    --output_dir checkpoints/grpo-math

# DiFFPO — PPO with importance-ratio clipping + joint sampler training
# Reference: Zhao et al. 2024  https://arxiv.org/abs/2510.02212
python scripts/learn_from_episodes.py \
    --mode diffppo \
    --ppo_clip_eps 0.2 \
    --train_sampler \             # jointly train adaptive step-budget controller
    --min_steps 8 --max_steps 128 \
    --step_budget_lambda 0.1 \
    --output_dir checkpoints/diffppo-math
```

| Mode | Data used | Algorithm | Best for |
|------|-----------|-----------|----------|
| `sft` | correct only | Cross-entropy | Fast, clean data |
| `grpo` | all evaluated | GRPO group advantage | Contrastive (correct vs wrong) |
| `diffppo` | all evaluated | PPO clip + step controller | Accuracy + inference speed Pareto |

### LaTeX Table Generation

```bash
# Generate publication-ready comparison table from results
python scripts/generate_latex_table.py results/summary.json --output paper_table.tex
```

### Config File

```yaml
# configs/eval_default.yaml
model:
  model_id: "checkpoints/llada-instruct"
  torch_dtype: "bfloat16"

inference:
  num_steps: 128
  block_length: 32
  temperature: 0.0
  cfg_scale: 0.0
  remasking: "low_confidence"
  max_new_tokens: 128

benchmarks:
  benchmarks: ["mbpp", "humaneval"]
  num_samples: null       # null = full dataset

dags:
  dags: ["confidence"]
  # choices: confidence | random | entropy | semi_ar | maskgit_cosine
  #          | critical_token_first | curriculum | linear | cot
  #          | skeleton | bidirectional | answer_first | adaptive_dynamic
```

---

## 13 Unmasking Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| `confidence` | Flat | Unmask highest model confidence first (LLaDA default) |
| `random` | Flat | Uniform random unmasking |
| `entropy` | Flat | Lowest-entropy (most certain by distribution) first |
| `semi_ar` | Flat | Block-by-block left-to-right, confidence within block |
| `maskgit_cosine` | Flat | MaskGIT cosine schedule: more tokens early, fewer later |
| `critical_token_first` | Flat | Highest KL divergence from uniform (most influential) first |
| `curriculum` | Flat | Easy tokens first (high confidence + low entropy) |
| `linear` | Flat | Strict left-to-right sequential |
| `cot` | DAG | Chain-of-Thought: reasoning segments before answer |
| `skeleton` | DAG | Structural tokens first, then fill details |
| `bidirectional` | DAG | Both ends toward center |
| `answer_first` | DAG | Answer region unmasked before reasoning |
| `adaptive_dynamic` | Dynamic | **Dynamic soft DAG** — constructs pairwise influence graph at runtime (ours) |

---

## 10 Benchmarks

| Benchmark | Type | Metric | Dataset |
|-----------|------|--------|---------|
| `mbpp` | Code generation | pass@1 | Google MBPP (Python) |
| `humaneval` | Code generation | pass@1 | OpenAI HumanEval (Python) |
| `gsm8k` | Math reasoning | exact match | Grade school math |
| `math` | Competition math | exact match | MATH (extracts `\boxed{}`) |
| `mmlu` | Knowledge | accuracy | 57-subject multitask |
| `hotpotqa` | Multi-hop QA | EM / F1 | Multi-hop reasoning |
| `arc` | Science reasoning | accuracy | ARC-Challenge |
| `prontoqa` | Logic reasoning | accuracy | Formal logic |
| `gpqa` | PhD-level science | accuracy | GPQA Diamond subset |
| `aime` | Competition math | accuracy | AMC/AIME (integer 000-999) |

---

## Project Structure

```
src/dllm_reason/
  models/          MDLM, SEDD, D3PM, LLaDA (4 dLLMs)
  graph/           TokenDAG, SpanDAG, 9 templates + registry, constraints, visualization
  scheduler/       13 unmasking strategies (8 flat + 4 DAG + 1 adaptive dynamic)
  search/          Evolutionary, Greedy, RL Policy, NOTEARS, E2E DAG, NAS (6 search methods)
  inference/       DiffusionSampler (auto-pad, early-stop), DAGSampler
  training/        Pretrain, DAG-aware, Fine-tune, DiffuGRPO, DiFFPO
  eval/            10 benchmark evaluators, metrics, DAG analysis
  library/         DAGEntry + DAGStore (retrieval/fusion/feedback/merge)
                   DAGEpisode + EpisodeStore  ← episode pipeline
  data/            Dataset loaders (GSM8K, MATH, ARC, ProntoQA, ...)
  utils/           Registry, logging, distributed

configs/           31 YAML configs (model, graph, search, task, eval, experiment, library)
scripts/           serve.py  search_dag_live.py  collect_episodes.py
                   learn_from_episodes.py  + eval / train / viz scripts
                   runs/  16 shell convenience scripts
tests/             DAG, schedulers, models, library (4 test suites)
notebooks/         DAG exploration, results analysis
docs/              API_REFERENCE.md  REFERENCES.md  deployment.md  tutorial  V1.0_RELEASE.md
REFERENCES.md      All cited papers (root-level, kept in sync with docs/)
```

---

## Key Components

### TokenDAG

Boolean adjacency matrix on GPU. Edge `(i, j)` = "position `i` must unmask before `j`".

```python
from dllm_reason.graph.dag import TokenDAG

dag = TokenDAG.linear_chain(seq_len=256)
ready = dag.ready_positions(is_unmasked)  # one batched GPU op
```

**8 named templates** accessible via unified registry:

```python
from dllm_reason.graph.templates import build_all_templates, build_template, TEMPLATE_NAMES
# TEMPLATE_NAMES = ['cot','answer_first','skeleton','bidirectional',
#                   'interleaved','linear','random_low','random_high']

templates = build_all_templates(seq_len=128, device="cuda")
dag = build_template("cot", seq_len=128)
```

### SpanDAG

Coarse-grained DAG over token spans — reduces search space by `span_size^2`.

```python
from dllm_reason.graph.span_dag import SpanDAG

sdag = SpanDAG.cot(num_spans=8, span_size=32, num_reasoning_steps=4)
token_dag = sdag.to_token_dag()  # expand for scheduler
```

### DAGScheduler

DAG constraints inject at scheduler layer — models need zero modification.

```python
from dllm_reason.scheduler.dag_scheduler import DAGScheduler
scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
```

### Adaptive Dynamic DAG (Novel)

Constructs soft dependency graph at runtime based on pairwise influence between masked positions.

```python
from dllm_reason.scheduler.adaptive_dynamic_scheduler import AdaptiveDynamicScheduler
scheduler = AdaptiveDynamicScheduler(influence_threshold=0.3, momentum=0.5)
```

### DAG Search (6 Methods)

Automatically discover optimal DAG structures.

```python
# Evolutionary search — templates seed the initial population automatically
from dllm_reason.search.evolutionary import EvolutionarySearch
searcher = EvolutionarySearch(
    population_size=20,
    init_templates=["cot", "skeleton", "bidirectional"],  # None = default set
    library=dag_store,
)
result = searcher.search(model, eval_fn, seq_len=256, budget=200)

# Greedy search — warm-start from the best template
from dllm_reason.search.greedy import GreedyEdgeSearch
searcher = GreedyEdgeSearch(init_templates=["cot", "skeleton", "answer_first"])
result = searcher.search(model, eval_fn, seq_len=256, budget=100)

# End-to-end DAG learning (differentiable, jointly with task loss)
from dllm_reason.search.e2e_dag_learner import E2EDAGLearner, E2EConfig
learner = E2EDAGLearner(config=E2EConfig(lr_dag=3e-3))
result = learner.search(model, eval_fn, seq_len=256, budget=200)

# NAS-style search (DARTS supernet or ENAS controller)
from dllm_reason.search.nas_search import NASDAGSearch, NASConfig
searcher = NASDAGSearch(config=NASConfig(mode="supernet", span_size=16))
result = searcher.search(model, eval_fn, seq_len=256, budget=200)
```

| Method | Type | Description |
|--------|------|-------------|
| Greedy | Black-box | Add/remove edges iteratively |
| Evolutionary | Black-box | Population-based with tournament selection |
| RL Policy | Black-box | GRU + REINFORCE for edge construction |
| Differentiable | Gradient | NOTEARS continuous relaxation |
| E2E DAG Learning | Gradient | Joint DAG + task loss optimization |
| NAS SuperNet/Controller | Gradient/RL | DARTS or ENAS-style architecture search |

### DAG Library

Persistent storage + retrieval + feedback for DAG structures.

- **DAGStore**: SQLite + FAISS vector index for `DAGEntry` records
- **Retrieval**: 3 channels (semantic, structural, performance)
- **Fusion**: 4 strategies (weighted, RRF, max, voting)
- **Feedback**: 3 sources (auto benchmark, human rating, Elo tournament)
- **Merge**: 3 strategies (union, intersection, weighted)

All components independently toggleable for ablation. 7 preset configs in `configs/library/`.

### Episode Pipeline

Full loop: collect interaction records → store → learn from them.

```python
from dllm_reason.library.episode import DAGEpisode, EpisodeStore

store = EpisodeStore("episodes/gsm8k.db")
ep = DAGEpisode(prompt="...", task_type="math", strategy_name="cot",
                output="...", ground_truth="42")
store.add(ep)
store.update_eval(ep.episode_id, correct=True, score=1.0)

# Paginated training iterator (memory-safe)
for ep in store.iter_for_training(task_type="math", correct_only=True):
    ...

store.print_stats()
```

Scripts: `collect_episodes.py` → `learn_from_episodes.py` (modes: `sft` / `grpo` / `diffppo`)

### RL Training

Three algorithms in `src/dllm_reason/training/rl_train.py`:

| Class | Algorithm | Key feature |
|-------|-----------|-------------|
| `DiffuGRPO` | GRPO | Group-relative advantage, no importance weights |
| `DiFFPO` | PPO + joint sampler | Importance-ratio clipping + `StepBudgetController` |

`DiFFPO` (Zhao et al. 2024, [arXiv:2510.02212](https://arxiv.org/abs/2510.02212)) jointly trains a lightweight MLP controller that predicts the optimal number of denoising steps per prompt, improving the accuracy–compute Pareto frontier.

---

## Models

| Model | Type | Reference |
|-------|------|-----------|
| LLaDA | LLaMA-3 masked diffusion (8B) | [Nie et al., 2025](https://arxiv.org/abs/2502.09992) |
| MDLM | Absorbing-state continuous-time | [Sahoo et al., 2024](https://arxiv.org/abs/2406.07524) |
| SEDD | Score-entropy discrete diffusion | [Lou et al., 2024](https://arxiv.org/abs/2310.16834) |
| D3PM | Discrete-time structured transitions | [Austin et al., 2021](https://arxiv.org/abs/2107.03006) |

---

## Configuration

All configs use YAML + Hydra/OmegaConf.

| Directory | Contents |
|-----------|----------|
| `configs/model/` | Model hyperparameters (mdlm, sedd, d3pm, llada) |
| `configs/graph/` | DAG template parameters |
| `configs/search/` | Search algorithm settings |
| `configs/task/` | Dataset configs |
| `configs/eval/` | Benchmark settings |
| `configs/experiment/` | End-to-end experiment combinations |
| `configs/library/` | DAG Library ablation variants |
| `configs/eval_default.yaml` | Default evaluation config |

---

## Documentation

- **[Tutorial: pip install + all-strategies evaluation](docs/tutorial_eval_all_strategies.md)**
- **[Deployment Guide: REST API, Docker, quantization](docs/deployment.md)**
- **[API Reference](docs/API_REFERENCE.md)**
- **[References: all cited papers](docs/REFERENCES.md)** · also at [REFERENCES.md](REFERENCES.md)
- **[Version History](docs/V1.0_RELEASE.md)**

---

## License

MIT License. See [LICENSE](LICENSE) for details.
