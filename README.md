# dLLM-Reason

**DAG-Guided Discrete Diffusion Language Models for Reasoning**

[![PyPI version](https://img.shields.io/badge/pip%20install-dllm--reason-blue)](https://github.com/BDeMo/dLLM_Reason)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/BDeMo/dLLM_Reason/actions/workflows/ci.yml/badge.svg)](https://github.com/BDeMo/dLLM_Reason/actions/workflows/ci.yml)

## Overview

dLLM-Reason is a research framework that enhances reasoning in discrete diffusion language models (dLLMs) by controlling the token unmasking order via DAG (Directed Acyclic Graph) topological structures.

**Core idea**: dLLMs generate text by iteratively unmasking tokens. We impose a DAG on unmasking order -- edges encode reasoning dependencies -- so prerequisite steps are generated before downstream conclusions.

```
Model Layer          Scheduler Layer         DAG Layer
(what to predict) <-> (where to unmask) <-> (dependency structure)
MDLM|SEDD|D3PM|LLaDA   DAGScheduler          TokenDAG + Templates
```

## Installation

```bash
pip install dllm-reason
```

Or install from GitHub (latest dev):

```bash
pip install "git+https://github.com/BDeMo/dLLM_Reason.git"
```

With optional extras (FAISS, sentence-transformers, dev tools):

```bash
pip install "dllm-reason[dev,library]"
```

For development (editable install):

```bash
git clone https://github.com/BDeMo/dLLM_Reason.git
cd dLLM_Reason
pip install -e ".[dev,library]"
```

After installation the following CLI commands are available globally:

| Command | Equivalent |
|---------|-----------|
| `dllm-eval-dags` | `python scripts/eval_dags.py` |
| `dllm-train` | `python scripts/train.py` |
| `dllm-eval` | `python scripts/evaluate.py` |
| `dllm-search` | `python scripts/search_dag.py` |
| `dllm-viz` | `python scripts/visualize_dag.py` |

### Quick Start

```bash
# Download models & datasets
python scripts/download_models.py              # -> checkpoints/llada-instruct/
python scripts/download_datasets.py            # -> datasets/

# HF mirror (China)
python scripts/download_models.py --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com
```

## Usage

All parameters live in `configs/eval_default.yaml`. CLI flags always override the config.

### 1. Default run (LLaDA + confidence scheduler)

```bash
bash scripts/run_eval.sh
# pass any CLI overrides after the script:
bash scripts/run_eval.sh --benchmarks mbpp --num_samples 50
bash scripts/run_eval.sh --dags cot skeleton --num_steps 64
bash scripts/run_eval.sh --verbose_errors
```

Results are written to `results/eval_<timestamp>/`.

### 2. Per-strategy scripts

`scripts/runs/` contains one script per unmasking strategy:

```bash
bash scripts/runs/confidence.sh    # highest-confidence first (LLaDA default)
bash scripts/runs/random.sh        # uniform random
bash scripts/runs/linear.sh        # left-to-right
bash scripts/runs/empty.sh         # no constraint
bash scripts/runs/cot.sh           # Chain-of-Thought DAG
bash scripts/runs/skeleton.sh      # Skeleton-then-Detail DAG
bash scripts/runs/bidirectional.sh # bidirectional DAG
bash scripts/runs/answer_first.sh  # answer region first
bash scripts/runs/all_strategies.sh  # all 8 strategies in one run
```

All scripts pass extra args through to `eval_dags.py`:

```bash
bash scripts/runs/cot.sh --benchmarks mbpp humaneval --num_samples 100 --cot_steps 6
```

### 3. Direct CLI

```bash
python scripts/eval_dags.py \
    --dags confidence cot skeleton \
    --benchmarks mbpp humaneval \
    --num_steps 64 --temperature 0.5 \
    --num_samples 100 \
    --output_dir results/my_run
```

### 4. Config file

Edit `configs/eval_default.yaml` to change defaults:

```yaml
model:
  model_id: "checkpoints/llada-instruct"
  torch_dtype: "bfloat16"        # bfloat16 | float16 | float32

inference:
  num_steps: 128
  block_length: 32               # max_new_tokens must be divisible
  temperature: 0.0               # 0 = greedy argmax
  cfg_scale: 0.0                 # 0 = disabled
  remasking: "low_confidence"    # low_confidence | random
  max_new_tokens: 128

benchmarks:
  benchmarks: ["mbpp", "humaneval"]
  num_samples: null              # null = full dataset
  run_tests: true                # false = skip code execution
  verbose_errors: false          # --verbose_errors to enable

dags:
  dags: ["confidence"]
  cot_steps: 4

output:
  output_dir: "results"
  resume: false
```

### 5. Save per-sample outputs (QA pairs, ground truth, trajectory)

Add `--save_outputs` to any run to write per-sample files alongside the summary JSON:

```bash
# Default: writes both JSON and Excel
bash scripts/run_eval.sh --save_outputs

# Use the dedicated script (has comments explaining every option)
bash scripts/runs/save_outputs.sh --benchmarks mbpp --num_samples 50

# Also record unmasking trajectory (one entry per diffusion step per sample)
bash scripts/runs/save_outputs.sh --record_trajectory --num_samples 10
```

Output files written to `results/<run>/`:

| File | Contents |
|------|----------|
| `{bench}_{dag}_samples.json` | Full per-sample records: prompt, generated, ground truth, pass/fail |
| `{bench}_{dag}_samples.xlsx` | Same data as a spreadsheet (one row per sample) |
| `{bench}_{dag}_trajectory.json` | *(only with `--record_trajectory`)* Decoded token states at each diffusion step |

Control what is included:

```bash
--save_outputs             # master switch (required)
--no_save_qa               # omit prompt + generated answer
--no_save_ground_truth     # omit reference answers
--record_trajectory        # add per-step unmasking states (large; keep off for big runs)
--output_formats json      # write only JSON (skip Excel)
--output_formats xlsx      # write only Excel (skip JSON)
```

Config file equivalents (`configs/eval_default.yaml`):

```yaml
save:
  save_outputs: false       # master switch
  save_qa: true
  save_ground_truth: true
  record_trajectory: false
  output_formats: ["json", "xlsx"]
```

### 6. Single-prompt inference

```bash
python scripts/infer_llada.py \
    --model_id checkpoints/llada-instruct \
    --prompt "What is 7 * 8?" \
    --num_steps 128 --block_length 32 --temperature 0.0
```

### Available strategies

| Strategy | Description |
|----------|-------------|
| `confidence` | Unmask highest-confidence tokens first |
| `random` | Uniform random unmasking |
| `linear` | Left-to-right sequential |
| `empty` | No constraint (pure random) |
| `cot` | Chain-of-Thought segment DAG |
| `skeleton` | Structural tokens first, then detail |
| `bidirectional` | Both ends toward center |
| `answer_first` | Answer region unmasked before reasoning |

### Available benchmarks

| Benchmark | Type | Metric |
|-----------|------|--------|
| `mbpp` | Python code generation | pass@1 |
| `humaneval` | Python code generation | pass@1 |
| `gsm8k` | Math reasoning | exact match |
| `math` | Competition math | exact match |
| `mmlu` | Knowledge (multi-subject) | accuracy |
| `hotpotqa` | Multi-hop QA | EM / F1 |
| `arc` | Science reasoning | accuracy |
| `prontoqa` | Logic reasoning | accuracy |

## Project Structure

```
src/dllm_reason/
  models/          MDLM, SEDD, D3PM, LLaDA (4 dLLMs)
  graph/           TokenDAG, 6 templates, constraints, visualization
  scheduler/       Random, Confidence, Linear, DAGScheduler, Adaptive (5 schedulers)
  search/          Evolutionary, Greedy, RL Policy, NOTEARS (4 search methods)
  inference/       DiffusionSampler, DAGSampler
  training/        Pretrain, DAG-aware, Fine-tune, Diffu-GRPO
  eval/            Metrics, 4 benchmark evaluators, DAG analysis
  library/         DAG Library (store, retrieval, fusion, feedback, merge)
  data/            GSM8K, MATH, ARC, ProntoQA loaders
  utils/           Registry, logging, distributed

configs/           31 YAML configs (model, graph, search, task, eval, experiment, library)
scripts/           Train, evaluate, search, visualize, download, server setup
tests/             DAG, schedulers, models, library (4 test suites)
notebooks/         DAG exploration, results analysis
docs/              V1.0 release notes, API reference, presentation
```

## Key Components

### TokenDAG

The core data structure. A boolean adjacency matrix on GPU where edge `(i, j)` means "position `i` must unmask before position `j`".

```python
from dllm_reason.graph.dag import TokenDAG

dag = TokenDAG.linear_chain(seq_len=256)
ready = dag.ready_positions(is_unmasked)  # one batched GPU op
```

**6 templates**: Chain-of-Thought, Answer-First, Skeleton-Detail, Bidirectional, Interleaved, Random.

### DAGScheduler

Injects DAG constraints at the scheduler layer -- models need zero modification.

```python
from dllm_reason.scheduler.dag_scheduler import DAGScheduler

scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
# sub_strategies: all_ready, confidence_topk, proportional
```

### DAG Search

Automatically discover optimal DAG structures.

```python
from dllm_reason.search.evolutionary import EvolutionarySearch

searcher = EvolutionarySearch(
    population_size=20,
    library=dag_store,           # seed from library
    task_description="math",
)
result = searcher.search(model, eval_fn, seq_len=256, budget=200)
# result.best_dag auto-written back to library
```

4 methods: Evolutionary, Greedy, RL Policy, Differentiable (NOTEARS).

### DAG Library

Persistent storage + retrieval + feedback for DAG structures.

- **Store**: SQLite + FAISS vector index
- **Retrieval**: 3 channels (semantic, structural, performance) -- independently toggleable
- **Fusion**: 4 strategies (weighted, RRF, max, voting)
- **Feedback**: 3 sources (auto benchmark, human rating, Elo tournament)
- **Merge**: 3 strategies (union, intersection, weighted)

All components independently toggleable for ablation experiments. 7 preset configs in `configs/library/`.

### Models

| Model | Type | Reference |
|-------|------|-----------|
| MDLM | Absorbing-state continuous-time diffusion | [Sahoo et al., 2024](https://github.com/kuleshov-group/mdlm) |
| SEDD | Score-entropy discrete diffusion | [Lou et al., 2024](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) |
| D3PM | Discrete-time structured transitions | [Austin et al., 2021](https://arxiv.org/abs/2107.03006) |
| LLaDA | LLaMA-3 based masked diffusion (8B) | [GSAI-ML](https://github.com/ML-GSAI/LLaDA) |

### Benchmarks

| Benchmark | Type | Metric |
|-----------|------|--------|
| GSM8K | Math reasoning | Exact match |
| MATH | Competition math | Exact match |
| MBPP | Code generation | pass@1 |
| HumanEval | Code generation | pass@1 |
| HotpotQA | Multi-hop QA | EM, F1 |
| MMLU | Knowledge | Accuracy |
| ARC | Science reasoning | Accuracy |
| ProntoQA | Logic | Accuracy |

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
| `configs/eval_default.yaml` | Default evaluation config (used by run_eval.sh) |

## Documentation

- **[Tutorial: pip install + all-strategies evaluation](docs/tutorial_eval_all_strategies.md)** -- Step-by-step guide to install via pip and run all 8 DAG strategies
- [`docs/V1.0_RELEASE.md`](docs/V1.0_RELEASE.md) -- Full version history and feature details
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) -- Complete API reference with code examples
- [`docs/dLLM_Reason_V1.2.3.pptx`](docs/dLLM_Reason_V1.2.3.pptx) -- Project presentation (12 slides, v1.2.3)

## License

MIT License. See [LICENSE](LICENSE) for details.
