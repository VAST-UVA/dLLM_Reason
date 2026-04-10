# dLLM-Reason Feature Manual

> Version: v1.5.0  |  Last updated: 2026-04-09  |  Language: English  |  中文: [FEATURE_MANUAL.zh.md](FEATURE_MANUAL.zh.md)

## 0. Project Overview

**dLLM-Reason** is a research framework for reasoning on top of discrete diffusion language models (dLLMs). Its core contribution is using a **TokenDAG** to constrain the partial order in which tokens are unmasked, explicitly injecting *reasoning dependencies* into the diffusion generation process.

The framework enables:

- Hot-swapping **unmasking schedulers** on top of any dLLM backend (MDLM / SEDD / D3PM / LLaDA);
- **Searching for optimal DAG structures** on reasoning tasks (6 search methods);
- Persisting sampled rollouts as reusable "reasoning experience" via the **Episode Pipeline**;
- Learning from the episode library with **SFT / GRPO / DiFFPO / UnmaskRL**;
- Unified evaluation on 10 benchmarks;
- Interactive debugging via a **FastAPI service** and **Web UI**.

---

## 1. Top-Level Architecture

```
                     ┌──────────────────┐
                     │   DiffusionLM    │
                     │ (MDLM/SEDD/D3PM) │
                     └────────┬─────────┘
                              │ logits + confidences
                              ▼
 ┌──────────────┐    ┌────────────────────┐    ┌──────────────┐
 │  TokenDAG    │───>│ UnmaskingScheduler │───>│  Sampled     │
 │  (structure) │    │ (DAGScheduler)     │    │  Sequence    │
 └──────────────┘    └────────────────────┘    └──────────────┘
         ▲                                            │
         │           ┌────────────────────┐           │
         └───────────│  DAGSearcher       │<──────────┘
                     │  (eval feedback)   │   accuracy / reward
                     └────────────────────┘
```

Key insight: **The DAG constrains ORDER, the dLLM supplies TOKEN PREDICTIONS, and the two are decoupled.** Any dLLM composes freely with any scheduler.

---

## 2. Models Layer — `dllm_reason.models`

Unified `DiffusionLM` interface (`models/base.py`):

```python
class DiffusionLM(nn.Module):
    def forward(self, x_t, t) -> Tensor: ...
    def compute_loss(self, x_0) -> Tensor: ...
    def noise_input(self, x_0, t) -> Tensor: ...
    def sample(self, scheduler, ...) -> Tensor: ...
```

| Model | File | Description |
|-------|------|-------------|
| **MDLM** | `models/mdlm.py` | Masked diffusion LM; simplest baseline |
| **SEDD** | `models/sedd.py` | Score-Entropy Discrete Diffusion |
| **D3PM** | `models/d3pm.py` | Discrete diffusion prior; supports absorbing / uniform |
| **LLaDA** | `models/llada.py` | Wrapper around HuggingFace `GSAI-ML/LLaDA-8B` |

Backbone transformer (`models/backbone/transformer.py`):
- Pre-LN, RMSNorm, SwiGLU, Rotary Position Embedding
- Timestep embeddings via AdaLN / FiLM

---

## 3. Graph Layer — `dllm_reason.graph`

### 3.1 `TokenDAG` (`graph/dag.py`)

- Representation: a `(seq_len, seq_len)` boolean adjacency tensor where `adj[i, j] = True` means **position i must be unmasked before position j**.
- Core operation `ready_positions(unmasked)` determines which positions are unblocked in a single GPU matrix op:

```python
ready = (adj.logical_not() | unmasked.unsqueeze(-1)).all(dim=0)
```

- API:
  - `from_edges(edges, seq_len)`
  - `from_levels(levels, seq_len)`
  - `linear_chain(seq_len)`
  - `no_edges(seq_len)`
  - `topological_levels()`
  - `ready_positions(unmasked_mask)`
  - `is_valid()` / `num_edges()`
  - `mutate(p_add, p_remove)`

### 3.2 Predefined templates (`graph/templates.py`)

| Name | Meaning | Use case |
|------|---------|----------|
| `cot` | Output split into segments; each segment depends on previous, parallel within | Multi-step math reasoning |
| `answer_first` | Answer tokens are roots and unmask first; reasoning filled after | Verification-style reasoning |
| `skeleton` | L0 = structural tokens, L1 = operands, L2 = filler | Structured reasoning |
| `bidirectional` | Unmask inward from both ends | Boundary-constrained tasks |
| `interleaved` | Interleaved groups, sequential across, parallel within | Interleaved compute / narration |
| `linear` | Strict left-to-right (equivalent to AR) | Sanity check |
| `random_low` | Random DAG, density = 0.05 | Search populations |
| `random_high` | Random DAG, density = 0.15 | Search populations |

Constructors: `build_all_templates(seq_len, device)` returns every template at once; `build_template(name, seq_len, device)` returns a single one.

### 3.3 Utilities

- `graph/constraints.py` — acyclicity validation and safe mutate
- `graph/viz.py` — matplotlib + networkx visualization

---

## 4. Scheduler Layer — `dllm_reason.scheduler`

Unified interface:

```python
class UnmaskingScheduler(ABC):
    def select_positions(
        self,
        step: int,
        mask: Tensor,        # current mask
        logits: Tensor,      # (B, L, V)
        confidences: Tensor, # (B, L)
    ) -> Tensor: ...
```

| Name | File | Mechanism |
|------|------|-----------|
| `random` | `random_scheduler.py` | Uniform sampling |
| `confidence` | `confidence.py` | Top-k max-prob |
| `linear` | `linear_scheduler.py` | Strict left-to-right |
| `maskgit_cosine` | `maskgit_cosine.py` | Cosine schedule |
| `low_confidence_remask` | `low_conf_remask.py` | Re-mask low-confidence tokens and re-predict |
| `entropy` | `entropy_scheduler.py` | Top-k by negative entropy |
| `stochastic_confidence` | `stoch_conf.py` | Gumbel-top-k sampling |
| `adaptive_dynamic` | `adaptive_dynamic.py` | Confidence + DAG-readiness weighted |
| `semi_ar` | `semi_ar.py` | Parallel within blocks, sequential across |
| `curriculum` | `curriculum.py` | Hard-to-easy progression |
| `dag` | `dag_scheduler.py` | **DAG-guided ★ core contribution** |

**DAGScheduler per-step flow:**

1. Call `dag.ready_positions(already_unmasked)` for all positions whose dependencies are satisfied.
2. Intersect with the still-masked positions → eligible positions.
3. A sub-policy selects k from eligible (`all_ready` / `confidence_topk` / `proportional`).
4. Return the selected positions.

---

## 5. Search Layer — `dllm_reason.search`

Unified interface:

```python
class DAGSearcher:
    def search(self, model, dataset, fitness_fn, budget) -> TokenDAG: ...
```

| Method | File | When to use |
|--------|------|-------------|
| Greedy | `greedy.py` | Fast baseline; add/remove one edge at a time |
| Evolutionary | `evolutionary.py` | Population + crossover + mutation |
| RL Policy | `rl_policy.py` | REINFORCE over a DAG-editing policy |
| Differentiable | `differentiable.py` | NOTEARS-style continuous relaxation |
| End-to-End | `e2e_search.py` | End-to-end gradient |
| NAS Controller | `nas_search.py` | LSTM controller that emits DAGs |

**Fitness signals** (`search/fitness.py`):
- `accuracy`, `reward`, `log_likelihood`, `composite`

Population initialization defaults to `build_all_templates(seq_len)` as seeds.

---

## 6. Training Layer — `dllm_reason.training`

| Mode | File | Description |
|------|------|-------------|
| `pretrain` | `pretrain.py` | Standard MDM pretraining |
| `finetune` | `finetune.py` | Supervised finetune |
| `dag_aware` | `dag_aware_train.py` | Training with a DAG-biased masking distribution |
| `rl` | `rl_train.py` | diffu-GRPO / DiFFPO / UnmaskingPolicyRL |

### 6.1 RL details

`rl_train.py` contains:

- **`DiffuGRPO`** — GRPO adapted to diffusion LMs, referencing the `d1` repository.
- **`DiFFPO`** (Diffusion Fast-Forward Policy Optimization):
  - PPO-clip advantage
  - `StepBudgetController`: dynamically caps the number of diffusion steps per episode.
- **`UnmaskingPolicyNet` + `UnmaskingPolicyRL`**:
  - Freeze the LM, train only a small policy net that acts as the scheduler.
  - Process-level REINFORCE.
  - The DAG structure can be fed in as a policy input feature.

### 6.2 Training CLI

```bash
# Basic
dllm-train --model mdlm --dataset gsm8k --mode pretrain

# DAG-aware
dllm-train --model mdlm --dataset gsm8k --dag_aware --dag cot

# RL
dllm-train --model mdlm --dataset gsm8k --mode rl --rl_algo diffu_grpo

# Custom run name (v1.4.1+)
dllm-train --model mdlm --dataset gsm8k --name my_experiment
```

Auto-generated checkpoint directory: `checkpoints/<name>_<timestamp>/`

---

## 7. Episode Pipeline & Library — `dllm_reason.library`

Core idea: persist dLLM rollout traces (prompt, DAG, output, correctness, score) as "experience" for offline or online learning.

### 7.1 `DAGEpisode` dataclass

Fields: `episode_id, prompt, task_type, ground_truth, strategy_name, dag_seq_len, dag_json, output, correct, score, reward, meta, created_at`.

### 7.2 `EpisodeStore` (SQLite, WAL mode)

API:
- `add(episode)` / `add_many(episodes)`
- `query(task_type=..., strategy=..., min_score=..., limit=...)`
- `delete(episode_id)`
- `stats()`
- `close()`

### 7.3 Scripts

| Command | Purpose |
|---------|---------|
| `dllm-collect-episodes` | Concurrent rollouts → write to store |
| `dllm-learn-from-episodes` | Read from store → SFT / GRPO / DiFFPO / UnmaskRL |
| `dllm-inspect-episodes` | CLI browser |
| `dllm-manage-library` | Clean up, deduplicate, export |
| `dllm-add-feedback` | Push human feedback scores into the store |

---

## 8. Evaluation Layer — `dllm_reason.eval`

`reasoning_eval.py` supports 10 benchmarks:

| Category | Benchmarks |
|----------|------------|
| Code | `mbpp`, `humaneval` |
| Math | `gsm8k`, `math`, `aime` |
| Multiple choice | `arc`, `mmlu`, `gpqa` |
| Multi-hop | `hotpotqa` |
| Logic | `prontoqa` |

Metrics: `exact_match`, `accuracy`, `pass@k`, `rouge`, `reasoning_score` (custom).

```bash
dllm-evaluate --model mdlm --ckpt <path> --benchmark gsm8k
dllm-eval-dags --model mdlm --benchmark gsm8k --dag_dir <dir>
```

---

## 9. Services & UI

### 9.1 FastAPI service (`scripts/serve.py`)

- Launch: `dllm-serve --model mdlm --port 8000`
- Endpoints:
  - `POST /generate` — generate text
  - `POST /switch_strategy` — hot-swap the scheduler
  - `GET /strategies` — list available schedulers
  - `POST /switch_dag` — hot-swap the DAG
  - `GET /health`

### 9.2 Web UI (`scripts/webui.py`)

- Launch: `dllm-webui`
- Single-file HTML dashboard with:
  - Model / strategy switching
  - DAG visualization
  - Episode browser

### 9.3 DAG analysis scripts

- `dllm-visualize-dag` — export PNG / SVG
- `dllm-analyze-dag` — stats on edges, longest path, ready fan-out

---

## 10. CLI Entry Points (`pyproject.toml [project.scripts]`)

v1.4.2 ships **17** entry points:

| Command | Script | Purpose |
|---------|--------|---------|
| `dllm-train` | `train.py` | Training (supports `--name`) |
| `dllm-evaluate` | `evaluate.py` | Single-model evaluation |
| `dllm-eval-dags` | `eval_dags.py` | Batch DAG evaluation |
| `dllm-search-dag` | `search_dag.py` | Structure search |
| `dllm-visualize-dag` | `visualize_dag.py` | Visualization |
| `dllm-serve` | `serve.py` | REST service |
| `dllm-webui` | `webui.py` | Browser dashboard |
| `dllm-run-pipeline` | `run_pipeline.py` | 5-stage end-to-end pipeline |
| `dllm-collect-episodes` | `collect_episodes.py` | Episode collection |
| `dllm-learn-from-episodes` | `learn_from_episodes.py` | Offline learning |
| `dllm-manage-library` | `manage_library.py` | Cleanup / export |
| `dllm-benchmark-schedulers` | `benchmark_schedulers.py` | Scheduler comparison |
| `dllm-analyze-dag` | `analyze_dag.py` | Structural stats |
| `dllm-inspect-episodes` | `inspect_episodes.py` | CLI browse |
| `dllm-generate-templates` | `generate_templates.py` | Emit template candidates |
| `dllm-add-feedback` | `add_feedback.py` | Human-feedback ingestion |
| `dllm-merge-dags` | `merge_dags.py` | DAG merging (union / voting) |

---

## 11. 5-Stage Pipeline (`scripts/run_pipeline.py`)

A one-shot end-to-end pipeline supporting `--resume` and `--stop_on_error`.

1. **download** — cache the HuggingFace dataset
2. **collect** — generate the initial episode library with a baseline scheduler
3. **search** — search a DAG using the episode library as the fitness source
4. **learn** — trigger SFT / GRPO / DiFFPO / UnmaskRL depending on flags
5. **eval** — evaluate the new model + new DAG on the benchmark

Example:

```bash
dllm-run-pipeline \
  --model mdlm \
  --dataset gsm8k \
  --search_method evolutionary \
  --rl_algo diffu_grpo \
  --output_dir runs/exp1
```

---

## 12. Typical Workflows

### 12.1 Reproduce a baseline

```bash
dllm-train --model mdlm --dataset gsm8k --mode pretrain
dllm-evaluate --model mdlm --ckpt checkpoints/mdlm_gsm8k_pretrain_* --benchmark gsm8k
```

### 12.2 DAG-guided inference

```bash
# Evaluate with predefined templates
dllm-eval-dags --model mdlm --benchmark gsm8k --templates cot,skeleton,bidirectional

# Search for the best DAG
dllm-search-dag --model mdlm --dataset gsm8k --method evolutionary --budget 200

# Evaluate with the searched DAG
dllm-evaluate --model mdlm --dag runs/search/best.pt --benchmark gsm8k
```

### 12.3 Episode-based learning

```bash
dllm-collect-episodes --model mdlm --dataset gsm8k --n 10000 --out library.db
dllm-learn-from-episodes --store library.db --mode sft --out ckpt_sft
dllm-learn-from-episodes --store library.db --mode diffu_grpo --out ckpt_rl
```

### 12.4 End-to-end pipeline

```bash
dllm-run-pipeline --model mdlm --dataset gsm8k --output_dir runs/full_exp
```

---

## 13. Configuration System (Hydra)

The `configs/` directory:

```
configs/
├── model/          (mdlm.yaml, sedd.yaml, d3pm.yaml, llada.yaml)
├── graph/          (linear.yaml, cot.yaml, answer_first.yaml, ...)
├── search/         (greedy.yaml, evolutionary.yaml, rl_policy.yaml)
├── task/           (gsm8k.yaml, math.yaml, arc.yaml, ...)
└── experiment/     (combined experiment configs)
```

Override example:

```bash
dllm-train +experiment=mdlm_gsm8k model.lr=5e-4
```

---

## 14. Module Dependency Quick Reference

```
models/base.py            ─┐
models/{mdlm,sedd,d3pm,llada}.py ─┼─→ inference/sampler.py ──→ scripts/evaluate.py
                           │                    │
scheduler/*               ─┘      ┌─────────────┘
graph/dag.py              ──→ scheduler/dag_scheduler.py ──→ inference/dag_sampler.py
graph/templates.py        ──→ search/* ──→ scripts/search_dag.py
library/episode.py        ──→ scripts/collect_episodes.py, learn_from_episodes.py
training/rl_train.py      ──→ scripts/train.py (mode=rl)
eval/reasoning_eval.py    ──→ scripts/evaluate.py, eval_dags.py
```

---

## 15. Version History (selected)

- **v1.0** — initial MDLM + random / confidence / linear schedulers + DAG core
- **v1.2.3** — SEDD + D3PM baselines, additional DAG templates
- **v1.3.0** — LLaDA integration, benchmark suite expanded to 10
- **v1.4.0** — Episode Pipeline + DAG Library + 4 search methods
- **v1.4.1** — Training CLI enhancements (`--name` argument)
- **v1.4.2** — 5-stage `run_pipeline.py` + new CLI scripts
- **v1.4.3** — Bug fixes (`empty()` rename), `publish.yml` repo guard
- **v1.5.0** — Research pipeline (`run_research_pipeline.py`), ablation runner (`run_ablation.py`), batch inference API, model hot-swap

---

## 16. Known Issues

See `docs/BUG_AUDIT_V1.4.2.md` (the accompanying audit report) for the CRITICAL / HIGH / LOW issue lists. Implementation details in this manual may still contain individual bugs; the audit report and the latest commits are authoritative.

---

## 17. Acknowledgements & References

- MDLM: https://github.com/kuleshov-group/mdlm
- SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- LLaDA: https://github.com/ML-GSAI/LLaDA
- d1 (diffu-GRPO): https://github.com/dllm-reasoning/d1
- D3PM: https://arxiv.org/abs/2107.03006
- NOTEARS: https://arxiv.org/abs/1803.01422
