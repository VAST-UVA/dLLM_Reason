# dLLM-Reason API Reference (v1.7.0)

---

## 1. Models (`dllm_reason.models`)

### `DiffusionLM` (Abstract Base)

```python
from dllm_reason.models.base import DiffusionLM, DiffusionOutput

class DiffusionLM(nn.Module):
    def forward(x_t: Tensor, t: Tensor) -> DiffusionOutput
    def compute_loss(x_0: Tensor) -> Tensor
    def noise_input(x_0: Tensor, t: Tensor) -> Tensor
    def sample(scheduler: UnmaskingScheduler, ...) -> Tensor
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `MDLM` | `models.mdlm` | Absorbing-state masked diffusion. Schedules: geometric, linear, cosine. |
| `SEDD` | `models.sedd` | Score-entropy discrete diffusion. Score-based parameterization. |
| `D3PM` | `models.d3pm` | Discrete-time with absorbing/uniform transitions. Hybrid loss. |
| `LLaDAWrapper` | `models.llada` | Wraps HuggingFace LLaDA-8B-Instruct for inference. Supports quantization. |

```python
# Example: LLaDA with 4-bit quantization
from dllm_reason.models.llada import LLaDAWrapper
from transformers import BitsAndBytesConfig

model = LLaDAWrapper(
    model_id="GSAI-ML/LLaDA-8B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)
```

---

## 2. Graph (`dllm_reason.graph`)

### `TokenDAG`

```python
from dllm_reason.graph.dag import TokenDAG

# Constructors
dag = TokenDAG.empty(seq_len)
dag = TokenDAG.linear_chain(seq_len)
dag = TokenDAG.from_edges(seq_len, [(0, 4), (4, 8)])
dag = TokenDAG.from_levels(seq_len, [[0,1,2], [3,4,5], [6,7]])

# Key operations
ready = dag.ready_positions(is_unmasked)     # (B, L) bool tensor
levels = dag.topological_levels()             # list[list[int]]
schedule = dag.to_mask_schedule(num_steps)    # list[list[int]]
valid = dag.is_valid()                        # bool

# Mutations (return new DAG, validate acyclicity)
dag2 = dag.add_edges([(2, 6)])
dag3 = dag.remove_edges([(0, 4)])
dag4 = dag.mutate(num_add=2, num_remove=1)

# Properties
dag.seq_len        # int
dag.adjacency      # (L, L) bool tensor
dag.num_edges()    # int
dag.depth()        # int
```

### `SpanDAG`

Coarse-grained DAG over token spans (reduces search space by `span_size^2`).

```python
from dllm_reason.graph.span_dag import SpanDAG

# Constructors
sdag = SpanDAG.empty(num_spans=8, span_size=32)
sdag = SpanDAG.linear_chain(num_spans=8, span_size=32)
sdag = SpanDAG.cot(num_spans=8, span_size=32, num_reasoning_steps=4)
sdag = SpanDAG.from_levels(num_spans=8, span_size=32, levels=[[0,1],[2,3],[4,5,6,7]])

# Convert to TokenDAG for scheduler
token_dag = sdag.to_token_dag()

# Mutation operators (for search)
sdag.add_edge(src=0, dst=3)
sdag.remove_edge(src=0, dst=3)
sdag.is_valid()
```

### Templates

```python
from dllm_reason.graph.templates import (
    chain_of_thought_dag,      # Sequential reasoning steps, parallel within each
    answer_first_dag,          # Answer tokens first, then reasoning
    skeleton_then_detail_dag,  # Structure -> content
    bidirectional_dag,         # Forward + backward passes
    interleaved_dag,           # Alternating groups
    random_dag,                # Random edges with given density
)

dag = chain_of_thought_dag(seq_len=256, num_steps=4, prompt_len=64)
```

---

## 3. Schedulers (`dllm_reason.scheduler`)

### `UnmaskingScheduler` (Abstract Base)

```python
class UnmaskingScheduler(ABC):
    def select_positions(
        step: int, total_steps: int,
        current_mask: Tensor, is_unmasked: Tensor,
        logits: Tensor, confidences: Tensor,
        block_mask: Tensor | None = None,
        n_to_select: int = 1,
    ) -> Tensor  # bool tensor of positions to unmask
```

### All 13 Implementations

| Class | Module | Strategy | Description |
|-------|--------|----------|-------------|
| `ConfidenceScheduler` | `confidence_scheduler` | `confidence` | Highest model confidence first (LLaDA default) |
| `RandomScheduler` | `random_scheduler` | `random` | Uniform random from masked positions |
| `LinearScheduler` | `linear_scheduler` | `linear` | Strict left-to-right sequential |
| `EntropyScheduler` | `entropy_scheduler` | `entropy` | Lowest entropy (most certain by distribution) first |
| `SemiAutoregressiveScheduler` | `semi_ar_scheduler` | `semi_ar` | Block-by-block L->R, confidence within each block |
| `MaskGITCosineScheduler` | `maskgit_scheduler` | `maskgit_cosine` | Cosine schedule: unmask more tokens early, fewer later |
| `CriticalTokenFirstScheduler` | `critical_token_scheduler` | `critical_token_first` | Highest KL divergence from uniform distribution first |
| `CurriculumScheduler` | `curriculum_scheduler` | `curriculum` | Easy tokens first (high confidence + low entropy) |
| `DAGScheduler` | `dag_scheduler` | `cot`, `skeleton`, `bidirectional`, `answer_first` | DAG-constrained: ready_positions -> eligible -> sub-strategy |
| `AdaptiveDAGScheduler` | `dag_scheduler` | — | DAG + confidence-aware with bypass for stuck situations |
| `AdaptiveDynamicScheduler` | `adaptive_dynamic_scheduler` | `adaptive_dynamic` | **Novel**: Dynamic soft DAG constructed at runtime via pairwise influence |

```python
# Simple scheduler
from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
scheduler = ConfidenceScheduler()

# DAG-constrained scheduler
from dllm_reason.scheduler.dag_scheduler import DAGScheduler
scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
# sub_strategy options: "all_ready", "confidence_topk", "proportional"

# Adaptive dynamic DAG (novel)
from dllm_reason.scheduler.adaptive_dynamic_scheduler import AdaptiveDynamicScheduler
scheduler = AdaptiveDynamicScheduler(
    influence_threshold=0.3,  # higher = fewer constraints
    momentum=0.5,             # EMA smoothing across steps
)
```

---

## 4. Inference (`dllm_reason.inference`)

### `DiffusionSampler`

```python
from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig

config = SamplingConfig(
    num_steps=64,
    block_length=32,
    temperature=0.8,
    cfg_scale=0.0,
    remasking="low_confidence",
    show_progress=True,
    record_trajectory=False,
)
sampler = DiffusionSampler(model, scheduler, config)
result = sampler.sample(prompt_ids=input_ids, prompt_mask=prompt_mask, gen_length=256)
# result.sequences: (B, L) token ids
# result.trajectory: list of (B, L) tensors (if record_trajectory=True)
```

Features:
- **Auto-pad**: `gen_length` and `num_steps` automatically adjusted to be divisible by `block_length`/num_blocks
- **Early-stop**: Skips remaining denoising steps when block is fully unmasked
- **Trim**: Auto-padded tokens removed before returning

### `DAGSampler`

```python
from dllm_reason.inference.dag_sampler import DAGSampler
sampler = DAGSampler(model, dag, num_steps=64)
result = sampler.sample(batch_size=4)
```

---

## 5. Search (`dllm_reason.search`)

### Interface

```python
from dllm_reason.search.base import DAGSearcher, SearchResult

result: SearchResult = searcher.search(model, eval_fn, seq_len, budget)
# result.best_dag: TokenDAG
# result.best_fitness: float
# result.history: list[dict]
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `GreedyEdgeSearch` | `search.greedy` | Iterative add/remove single edges, keep best improvement |
| `EvolutionarySearch` | `search.evolutionary` | Population-based: tournament selection, crossover, topological mutation |
| `RLPolicySearch` | `search.rl_policy` | DAGPolicyNetwork + REINFORCE |
| `DifferentiableSearch` | `search.differentiable` | NOTEARS continuous relaxation with augmented Lagrangian |

```python
# With library integration
from dllm_reason.search.evolutionary import EvolutionarySearch

searcher = EvolutionarySearch(
    population_size=20,
    library=dag_store,
    library_config=lib_config,
    task_description="Solve grade school math problems",
)
result = searcher.search(model, eval_fn, seq_len=256, budget=200)
# Best DAG auto-written back to library
```

### Fitness Functions

```python
from dllm_reason.search.fitness import accuracy_fitness, perplexity_fitness, combined_fitness
score = accuracy_fitness(model, dag, dataset, num_samples=50)
```

---

## 6. Training (`dllm_reason.training`)

| Class | Module | Description |
|-------|--------|-------------|
| `Trainer` | `training.pretrain` | Standard pretraining (ELBO/score-entropy/VLB) |
| `DAGAwareTrainer` | `training.dag_aware_train` | DAG-biased masking during training |
| `Finetuner` | `training.finetune` | Answer-only loss fine-tuning |
| `DiffuGRPO` | `training.rl_train` | Diffu-GRPO reinforcement learning |

---

## 7. Evaluation (`dllm_reason.eval`)

### All 10 Benchmark Evaluators

| Class | Benchmark | Metric | Dataset |
|-------|-----------|--------|---------|
| `MBPPEvaluator` | `mbpp` | pass@1 | Google MBPP (Python) |
| `HumanEvalEvaluator` | `humaneval` | pass@1 | OpenAI HumanEval (Python) |
| `GSM8KEvaluator` | `gsm8k` | exact match | Grade school math |
| `MATHEvaluator` | `math` | exact match | Competition math (extracts `\boxed{}`) |
| `ARCEvaluator` | `arc` | accuracy | ARC-Challenge (science) |
| `MMLUEvaluator` | `mmlu` | accuracy | Massive multitask (57 subjects) |
| `HotpotQAEvaluator` | `hotpotqa` | EM / F1 | Multi-hop QA |
| `ProntoQAEvaluator` | `prontoqa` | accuracy | Formal logic reasoning |
| `GPQAEvaluator` | `gpqa` | accuracy | PhD-level science MCQ (diamond subset) |
| `AIMEEvaluator` | `aime` | accuracy | AMC/AIME competition math (integer 000-999) |

```python
from dllm_reason.eval.benchmarks import BENCHMARK_REGISTRY

evaluator_cls = BENCHMARK_REGISTRY["gsm8k"]
evaluator = evaluator_cls(model=model, scheduler=scheduler, num_samples=100)
metrics = evaluator.evaluate()  # {"accuracy": 0.72, "benchmark": "gsm8k", ...}
```

### DAG Analysis

```python
from dllm_reason.eval.dag_analysis import analyze_dag, compare_dags
stats = analyze_dag(dag)  # DAGStats(num_edges, depth, width, density, ...)
```

### LaTeX Table Generation

```python
# Generate publication-ready comparison table
python scripts/generate_latex_table.py results/summary.json --output paper_table.tex
```

---

## 8. Library (`dllm_reason.library`)

### Quick Start

```python
from dllm_reason.library import (
    LibraryConfig, DAGStore, DAGEntry,
    create_embedder, create_retrieval_channel, create_fusion,
    create_feedback_handler, create_merger, CompositeFitness,
    RetrievalQuery, RetrievalMode, FeedbackSource,
)

# Setup
config = LibraryConfig()
store = DAGStore(config.store)
embedder = create_embedder("sentence_transformer")

# Store a DAG
entry = DAGEntry.from_token_dag(dag, task_description="solve equations", source="search")
store.add(entry)

# Retrieve
channel = create_retrieval_channel(RetrievalMode.SEMANTIC, config.retrieval, embedder)
query = RetrievalQuery(task_description="quadratic formula")
results = channel.retrieve(query, store, top_k=5)  # [(entry, score), ...]

# Fuse multi-channel results
fuser = create_fusion(config.fusion)
merged = fuser.fuse({"semantic": results_a, "structural": results_b}, top_k=3)

# Feedback
handler = create_feedback_handler(FeedbackSource.ELO, config.feedback)
handler.update_pair(entry_a, entry_b, outcome_a=1.0, store=store)

# Merge DAGs
merger = create_merger(config.merge)
adj = merger.merge(entries, scores)  # (seq_len, seq_len) bool tensor

# Composite fitness
fitness = CompositeFitness(config)
result = fitness.evaluate(entry, benchmark="gsm8k")
# result.total, result.components, result.raw_scores, result.weights
```

### Ablation Toggles

```python
config.retrieval.enabled = False              # Disable retrieval
config.retrieval.channels = [RetrievalMode.SEMANTIC]  # Single channel
config.fusion.strategy = FusionStrategy.RRF   # Switch fusion
config.feedback.sources = [FeedbackSource.AUTO]  # Disable Elo
config.constraint.mode = ConstraintMode.SOFT  # Soft constraints
config.enabled = False                        # Kill entire library
```

---

## 9. Model Serving (`scripts/serve.py`)

### REST API

```python
# Start server
dllm-serve --model_id checkpoints/llada-instruct --port 8000 --quantize 4bit

# POST /generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 7*8?", "strategy": "adaptive_dynamic"}'

# GET /strategies  — list all 13 strategies
# GET /health      — model status and device info
```

### Request Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `prompt` | string | required | — |
| `strategy` | string | `"confidence"` | 13 strategies |
| `system_prompt` | string | null | — |
| `max_new_tokens` | int | 128 | 1-2048 |
| `num_steps` | int | 128 | 1-1024 |
| `block_length` | int | 32 | 1-512 |
| `temperature` | float | 0.0 | 0.0-2.0 |
| `cfg_scale` | float | 0.0 | 0.0-10.0 |
| `remasking` | string | `"low_confidence"` | `low_confidence`, `random` |

---

## 10. CLI Entry Points

```bash
dllm-train      # -> scripts/train.py
dllm-eval       # -> scripts/evaluate.py
dllm-eval-dags  # -> scripts/eval_dags.py
dllm-search     # -> scripts/search_dag.py
dllm-viz        # -> scripts/visualize_dag.py
dllm-serve      # -> scripts/serve.py
```

---

## 11. Configuration

All configs in `configs/` directory, compatible with Hydra/OmegaConf.

| Directory | Files | Purpose |
|-----------|-------|---------|
| `configs/model/` | 4 | Model hyperparameters (mdlm, sedd, d3pm, llada) |
| `configs/graph/` | 5 | DAG template parameters |
| `configs/search/` | 4 | Search algorithm settings |
| `configs/task/` | 4 | Dataset paths and preprocessing |
| `configs/eval/` | 4 | Benchmark evaluation settings |
| `configs/experiment/` | 3 | End-to-end experiment combos |
| `configs/library/` | 7 | Library ablation variants |
| `configs/eval_default.yaml` | 1 | Default evaluation config (used by run_eval.sh) |
