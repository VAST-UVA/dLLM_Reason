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
python scripts/download_models.py              # -> checkpoints/
python scripts/download_datasets.py            # -> datasets/

# Or use HF mirror (China)
python scripts/download_models.py --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com

# Server one-shot setup
bash scripts/setup_server.sh
bash scripts/setup_server.sh --mirror https://hf-mirror.com  # with mirror
```

## CLI

```bash
dllm-train       # Train dLLM models
dllm-eval        # Evaluate with multiple schedulers
dllm-eval-dags   # LLaDA + DAG strategies on benchmarks
dllm-search      # Search optimal DAG structures
dllm-viz         # Visualize DAG structures
```

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

## Example Workflows

```bash
# Train MDLM on GSM8K
dllm-train --model mdlm --dataset gsm8k --mode pretrain --output_dir checkpoints/mdlm_gsm8k

# Evaluate LLaDA with multiple DAG strategies
dllm-eval-dags --benchmarks mbpp humaneval mmlu --dags cot skeleton bidirectional --num_steps 128

# Search optimal DAG
dllm-search --method evolutionary --population_size 20 --budget 200

# Visualize DAG templates
dllm-viz --mode templates --seq_len 32 --output_dir figures/dags
```

## Configuration

All configs use YAML + Hydra/OmegaConf.

| Directory | Count | Contents |
|-----------|-------|----------|
| `configs/model/` | 4 | Model hyperparameters |
| `configs/graph/` | 5 | DAG template parameters |
| `configs/search/` | 4 | Search algorithm settings |
| `configs/task/` | 4 | Dataset configs |
| `configs/eval/` | 4 | Benchmark settings |
| `configs/experiment/` | 3 | End-to-end experiments |
| `configs/library/` | 7 | Library ablation variants |

## Documentation

- [`docs/V1.0_RELEASE.md`](docs/V1.0_RELEASE.md) -- Full version history and feature details
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) -- Complete API reference with code examples
- [`docs/dLLM_Reason_V1.0.pptx`](docs/dLLM_Reason_V1.0.pptx) -- Project presentation (10 slides)

## License

Research use only.
