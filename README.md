# dLLM-Reason

**DAG-Guided Discrete Diffusion Language Models for Reasoning**

## Overview

dLLM-Reason is a research framework that enhances reasoning in discrete diffusion language models (dLLMs) by controlling the token unmasking order via DAG (Directed Acyclic Graph) topological structures.

**Core idea**: dLLMs generate text by iteratively unmasking tokens. We impose a DAG on unmasking order -- edges encode reasoning dependencies -- so prerequisite steps are generated before downstream conclusions.

```
Model Layer          Scheduler Layer         DAG Layer
(what to predict) <-> (where to unmask) <-> (dependency structure)
MDLM|SEDD|D3PM|LLaDA   DAGScheduler          TokenDAG + Templates
```

## Quick Start

```bash
# Install
git clone https://github.com/BDeMo/dLLM_Reason.git
cd dLLM_Reason
pip install -e ".[dev,library]"

# Download models & datasets
python scripts/download_models.py              # -> checkpoints/llada-instruct/
python scripts/download_datasets.py            # -> datasets/

# HF mirror (China)
python scripts/download_models.py --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com
```

## Usage

### 1. Run evaluation (recommended entry point)

Edit the configuration block at the top of `scripts/run_eval.sh`, then:

```bash
bash scripts/run_eval.sh
```

Key variables to configure:

```bash
# scripts/run_eval.sh  -- configuration block
MODEL_ID="checkpoints/llada-instruct"  # local path to LLaDA weights
NUM_STEPS=128         # diffusion denoising steps
BLOCK_LENGTH=32       # tokens per block (MAX_NEW_TOKENS must be divisible by this)
TEMPERATURE=0.0       # 0 = greedy argmax, >0 = sampling
CFG_SCALE=0.0         # classifier-free guidance scale (0 = disabled)
REMASKING="low_confidence"  # low_confidence | random
MAX_NEW_TOKENS=128    # total generation length

BENCHMARKS="mbpp humaneval"   # space-separated list
NUM_SAMPLES=200               # samples per benchmark (leave empty for all)
DAGS="confidence"             # scheduler strategies (see below)

RUN_TESTS=true        # false = skip code execution, only inspect outputs
VERBOSE_ERRORS=false  # true = print per-sample error logs
```

Results and a full log are written to `results/llada_eval_<timestamp>/`.

### 2. Config file + CLI overrides

All parameters can also be set via `configs/eval_default.yaml` and overridden on the command line:

```bash
# Use config file defaults
python scripts/eval_dags.py --config configs/eval_default.yaml

# Override specific values
python scripts/eval_dags.py \
    --config configs/eval_default.yaml \
    --num_steps 64 \
    --temperature 0.5 \
    --benchmarks mbpp \
    --num_samples 50 \
    --verbose_errors
```

`configs/eval_default.yaml` sections:

```yaml
model:
  model_id: "checkpoints/llada-instruct"
  torch_dtype: "bfloat16"   # bfloat16 | float16 | float32

inference:
  num_steps: 128
  block_length: 32
  temperature: 0.0          # 0 = greedy argmax
  cfg_scale: 0.0
  remasking: "low_confidence"
  max_new_tokens: 128

benchmarks:
  benchmarks: ["mbpp", "humaneval"]
  num_samples: null         # null = all samples
  run_tests: true
  verbose_errors: false     # --verbose_errors to enable

dags:
  dags: ["confidence"]
  cot_steps: 4
```

### 3. Standalone single-prompt inference

```bash
python scripts/infer_llada.py \
    --model_id checkpoints/llada-instruct \
    --prompt "What is 7 * 8?" \
    --num_steps 128 \
    --block_length 32 \
    --temperature 0.0
```

### Available DAG / scheduler strategies

| Strategy | Description |
|----------|-------------|
| `confidence` | Unmask highest-confidence tokens first (strong baseline) |
| `random` | Uniform random unmasking |
| `linear` | Left-to-right sequential unmasking |
| `empty` | Standard LLaDA (no DAG constraint) |
| `cot` | Chain-of-Thought: segment-level dependencies |
| `skeleton` | Skeleton-then-Detail: structural tokens first |
| `bidirectional` | Bidirectional chains from both ends |
| `answer_first` | Answer token unmasked first, reasoning filled in |

Pass multiple strategies to compare in one run:

```bash
DAGS="confidence empty linear cot skeleton"
```

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

- [`docs/V1.0_RELEASE.md`](docs/V1.0_RELEASE.md) -- Full version history and feature details
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) -- Complete API reference with code examples
- [`docs/dLLM_Reason_V1.0.pptx`](docs/dLLM_Reason_V1.0.pptx) -- Project presentation (10 slides)

## License

Research use only.
