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
```

---

## Installation

```bash
# From PyPI
pip install dllm-reason

# From GitHub (latest dev)
pip install "git+https://github.com/BDeMo/dLLM_Reason.git"

# With all extras
pip install "dllm-reason[dev,library,serve]"

# Editable (development)
git clone https://github.com/BDeMo/dLLM_Reason.git
cd dLLM_Reason
pip install -e ".[dev,library,serve]"
```

### Optional Extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `dev` | pytest, pytest-cov, ruff | Testing and linting |
| `library` | faiss-cpu, sentence-transformers, scikit-learn | DAG Library retrieval |
| `serve` | fastapi, uvicorn, bitsandbytes | REST API serving + quantization |

### CLI Commands

After installation, the following commands are available globally:

| Command | Equivalent | Description |
|---------|-----------|-------------|
| `dllm-eval-dags` | `python scripts/eval_dags.py` | Multi-strategy x multi-benchmark evaluation |
| `dllm-serve` | `python scripts/serve.py` | REST API server with hot-switching strategies |
| `dllm-train` | `python scripts/train.py` | Model training |
| `dllm-eval` | `python scripts/evaluate.py` | Single-model evaluation |
| `dllm-search` | `python scripts/search_dag.py` | DAG structure search |
| `dllm-viz` | `python scripts/visualize_dag.py` | DAG visualization |
| `dllm-webui` | `python scripts/webui.py` | Interactive Web UI dashboard |

---

## Quick Start

```bash
# 1. Download model & datasets
python scripts/download_models.py              # -> checkpoints/llada-instruct/
python scripts/download_datasets.py            # -> datasets/

# China HuggingFace mirror
python scripts/download_models.py --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com

# 2. Smoke test (5 samples, confidence strategy)
dllm-eval-dags --dags confidence --benchmarks mbpp --num_samples 5

# 3. Full comparison (all 13 strategies x all 10 benchmarks)
bash scripts/runs/full_comparison.sh

# 4. Start REST API server
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
  graph/           TokenDAG, SpanDAG, 6 templates, constraints, visualization
  scheduler/       13 unmasking strategies (8 flat + 4 DAG + 1 adaptive dynamic)
  search/          Evolutionary, Greedy, RL Policy, NOTEARS, E2E DAG, NAS (6 search methods)
  inference/       DiffusionSampler (auto-pad, early-stop), DAGSampler
  training/        Pretrain, DAG-aware, Fine-tune, Diffu-GRPO
  eval/            10 benchmark evaluators, metrics, DAG analysis
  library/         DAG Library (store, retrieval, fusion, feedback, merge)
  data/            Dataset loaders (GSM8K, MATH, ARC, ProntoQA, ...)
  utils/           Registry, logging, distributed

configs/           31 YAML configs (model, graph, search, task, eval, experiment, library)
scripts/           8 Python scripts + 16 shell run scripts
tests/             DAG, schedulers, models, library (4 test suites)
notebooks/         DAG exploration, results analysis
docs/              Version history, API reference, deployment guide, tutorial, references
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

6 templates: Chain-of-Thought, Answer-First, Skeleton-Detail, Bidirectional, Interleaved, Random.

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
# Evolutionary search
from dllm_reason.search.evolutionary import EvolutionarySearch
searcher = EvolutionarySearch(population_size=20, library=dag_store)
result = searcher.search(model, eval_fn, seq_len=256, budget=200)

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

- **Store**: SQLite + FAISS vector index
- **Retrieval**: 3 channels (semantic, structural, performance)
- **Fusion**: 4 strategies (weighted, RRF, max, voting)
- **Feedback**: 3 sources (auto benchmark, human rating, Elo tournament)
- **Merge**: 3 strategies (union, intersection, weighted)

All components independently toggleable for ablation. 7 preset configs in `configs/library/`.

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
- **[References: paper citations for all components](docs/REFERENCES.md)**
- **[Version History](docs/V1.0_RELEASE.md)**

---

## License

MIT License. See [LICENSE](LICENSE) for details.
