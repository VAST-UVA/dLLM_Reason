# dLLM-Reason V1.0 API Reference

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
| `LLaDAWrapper` | `models.llada` | Wraps HuggingFace LLaDA-8B-Instruct for inference. |

```python
# Example
from dllm_reason.models.mdlm import MDLM
model = MDLM(vocab_size=32000, max_seq_len=512, dim=768, num_layers=12, num_heads=12)
loss = model.compute_loss(x_0)
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

### Templates

```python
from dllm_reason.graph.templates import (
    chain_of_thought_dag,      # Sequential reasoning steps, parallel within each
    answer_first_dag,          # Answer tokens first, then reasoning
    skeleton_then_detail_dag,  # Structure → content
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
    ) -> Tensor  # bool tensor of positions to unmask
```

### Implementations

| Class | Strategy |
|-------|----------|
| `RandomScheduler` | Uniform random selection from masked positions |
| `ConfidenceScheduler` | Highest-confidence masked positions first |
| `LinearScheduler` | Left-to-right sequential |
| `DAGScheduler` | DAG-constrained: `ready_positions()` → eligible → sub-strategy |
| `AdaptiveDAGScheduler` | DAG + confidence-aware with bypass for stuck situations |

```python
from dllm_reason.scheduler.dag_scheduler import DAGScheduler
scheduler = DAGScheduler(dag, sub_strategy="confidence_topk")
# sub_strategy options: "all_ready", "confidence_topk", "proportional"
```

---

## 4. Inference (`dllm_reason.inference`)

### `DiffusionSampler`

```python
from dllm_reason.inference.sampler import DiffusionSampler, SamplingConfig

config = SamplingConfig(
    num_steps=64,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    show_progress=True,
    record_trajectory=False,
)
sampler = DiffusionSampler(model, scheduler, config)
result = sampler.sample(batch_size=4, seq_len=256)
# result.sequences: (B, L) token ids
```

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

```python
from dllm_reason.search.evolutionary import EvolutionarySearch
from dllm_reason.search.greedy import GreedyEdgeSearch
from dllm_reason.search.rl_policy import RLPolicySearch
from dllm_reason.search.differentiable import DifferentiableSearch

# With library integration
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

### Benchmark Evaluators

```python
from dllm_reason.eval.benchmarks import MBPPEvaluator, HumanEvalEvaluator
evaluator = MBPPEvaluator(model, scheduler, num_samples=100)
metrics = evaluator.evaluate()  # {"pass@1": 0.31, ...}
```

### DAG Analysis

```python
from dllm_reason.eval.dag_analysis import analyze_dag, compare_dags
stats = analyze_dag(dag)  # DAGStats(num_edges, depth, width, density, ...)
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
# Disable retrieval (ablation)
config.retrieval.enabled = False

# Single channel only
config.retrieval.channels = [RetrievalMode.SEMANTIC]

# Switch fusion strategy
config.fusion.strategy = FusionStrategy.RRF

# Disable Elo
config.feedback.sources = [FeedbackSource.AUTO]

# Soft constraints
config.constraint.mode = ConstraintMode.SOFT

# Kill entire library
config.enabled = False
```

---

## 9. CLI Entry Points

```bash
dllm-train    # → scripts/train.py
dllm-eval     # → scripts/evaluate.py
dllm-eval-dags # → scripts/eval_dags.py
dllm-search   # → scripts/search_dag.py
dllm-viz      # → scripts/visualize_dag.py
```

---

## 10. Configuration

All configs in `configs/` directory, compatible with Hydra/OmegaConf.

| Directory | Files | Purpose |
|-----------|-------|---------|
| `configs/model/` | 4 | Model hyperparameters |
| `configs/graph/` | 5 | DAG template parameters |
| `configs/search/` | 4 | Search algorithm settings |
| `configs/task/` | 4 | Dataset paths and preprocessing |
| `configs/eval/` | 4 | Benchmark evaluation settings |
| `configs/experiment/` | 3 | End-to-end experiment combos |
| `configs/library/` | 7 | Library ablation variants |
