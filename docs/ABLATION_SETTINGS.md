# Ablation Settings

This document lists all ablation dimensions available in the dLLM-Reason research pipeline. Each dimension corresponds to a CLI parameter in `scripts/run_research_pipeline.py`.

Pre-configured experiment sets are defined in `scripts/run_ablation.py`.

---

## Quick Start

```bash
# 1. Start the inference server
python scripts/serve.py --model_id GSAI-ML/LLaDA-8B-Instruct

# 2. Run a quick sanity check (10 samples, 1 scheduler)
python scripts/run_research_pipeline.py \
    --stages 1 --datasets gsm8k --num_samples 10

# 3. Run all ablation experiments
python scripts/run_ablation.py

# 4. Run specific ablations
python scripts/run_ablation.py --experiments scheduler_compare sft_per_template
```

---

## Ablation Dimensions

### Stage 1 — Unmasking Scheduler

| Parameter | `--s1_schedulers` |
|-----------|-------------------|
| Affects | Stage 1 (baseline evaluation) |
| Default | `confidence` |

Available schedulers:

| Scheduler | Type | Description |
|-----------|------|-------------|
| `confidence` | Flat | Unmask highest-confidence tokens first |
| `random` | Flat | Unmask in random order |
| `linear` | Flat | Unmask linearly (left-to-right bias) |
| `entropy` | Flat | Unmask lowest-entropy tokens first |
| `semi_ar` | Flat | Semi-autoregressive (block-by-block L-to-R) |
| `maskgit_cosine` | Flat | MaskGIT-style cosine schedule |
| `critical_token_first` | Flat | Unmask structurally critical tokens first |
| `curriculum` | Flat | Easy-to-hard curriculum ordering |
| `adaptive_dynamic` | Flat | Adaptive step allocation |
| `cot` | DAG | Chain-of-thought: sequential reasoning stages |
| `skeleton` | DAG | Skeleton-first: key tokens before detail |
| `bidirectional` | DAG | Bidirectional: segments filled inward |
| `answer_first` | DAG | Answer-first: conclusion before reasoning |

**Ablation experiment:** `scheduler_compare` — runs all 13 schedulers on gsm8k + math.

---

### Stage 2 — DAG Discovery Method

| Parameter | `--s2_method` |
|-----------|---------------|
| Affects | Stage 2 (DAG discovery) |
| Default | `sweep` |

| Method | Description |
|--------|-------------|
| `sweep` | Try all `--s2_strategies` per prompt, pick best |
| `search` | Per-prompt search with `--s2_search_method` |

**Search algorithms** (`--s2_search_method`, only when `--s2_method=search`):

| Algorithm | Description |
|-----------|-------------|
| `greedy` | Greedy template selection |
| `evolutionary` | Evolutionary DAG structure search |
| `rl_policy` | RL-trained DAG policy |
| `differentiable` | Differentiable DAG relaxation |

**Search budget** (`--s2_search_budget`): number of evaluations per prompt (default: 50).

**Ablation experiments:** `sweep_all_strategies`, `search_greedy`, `search_evolutionary`.

---

### Stage 2 — DAG Template Strategies

| Parameter | `--s2_strategies` |
|-----------|-------------------|
| Affects | Stage 2 sweep mode |
| Default | `confidence cot skeleton bidirectional answer_first linear random` |

These are the strategies tried per prompt during sweep discovery. Any scheduler from Stage 1 can be used here.

---

### Stage 3 — Training Mode

| Parameter | `--s3_mode` |
|-----------|-------------|
| Affects | Stage 3 (training) |
| Default | `sft` |

| Mode | Description | Reference |
|------|-------------|-----------|
| `sft` | Supervised fine-tuning on correct episodes | Standard CE loss |
| `grpo` | Group Relative Policy Optimization | DiffuGRPO |
| `diffppo` | DiFFPO (PPO + joint sampler training) | [Zhao et al. 2024](https://arxiv.org/abs/2510.02212) |
| `unmask_rl` | Frozen LM + lightweight policy net (REINFORCE) | [Jazbec et al. 2025](https://arxiv.org/abs/2512.09106) |

**Ablation experiments:** `sft_per_template`, `grpo_per_template`, `diffppo_default`, `unmask_rl_small`.

---

### Stage 3 — DAG Injection Mode

| Parameter | `--s3_dag_mode` |
|-----------|-----------------|
| Affects | Stage 3 (training) |
| Default | `per_template` |

| Mode | Description |
|------|-------------|
| `per_template` | Group episodes by best template; each group trains with its DAG bias |
| `consensus` | Build a single consensus DAG (majority vote from all correct episodes) |
| `none` | No DAG bias — standard training (ablation control) |

**Ablation experiments:** `sft_per_template` vs `sft_consensus` vs `sft_no_dag`.

---

### Stage 3 — DAG Bias Strength

| Parameter | `--dag_bias_strength` |
|-----------|----------------------|
| Affects | Stage 3 (DAG-aware training) |
| Default | `0.5` |
| Range | `0.0` to `1.0` |

Controls how strongly the DAG biases the masking noise during training:
- `0.0` — No bias (equivalent to `--s3_dag_mode none`)
- `0.5` — Moderate bias (default)
- `1.0` — Full bias (masking strictly follows DAG levels)

**Ablation experiments:** `bias_0.0`, `bias_0.1`, `bias_0.3`, `bias_0.5`, `bias_0.8`, `bias_1.0`.

---

### Stage 3 — Learning Rate

| Parameter | `--lr` |
|-----------|--------|
| Affects | Stage 3 |
| Default | `1e-5` |
| Suggested range | `1e-6` to `1e-4` |

---

### Stage 3 — KL Coefficient

| Parameter | `--kl_coeff` |
|-----------|-------------|
| Affects | Stage 3 (grpo, diffppo) |
| Default | `0.01` |
| Suggested range | `0.001` to `0.1` |

Controls the KL divergence penalty against the reference model.

---

### Stage 3 — PPO Clip Epsilon

| Parameter | `--ppo_clip_eps` |
|-----------|-----------------|
| Affects | Stage 3 (diffppo) |
| Default | `0.2` |
| Suggested values | `0.1`, `0.2`, `0.3` |

---

### Stage 3 — Step-Budget Controller

| Parameter | `--train_sampler` |
|-----------|-------------------|
| Affects | Stage 3 (diffppo) |
| Default | `False` |

When enabled, jointly trains an adaptive step-budget controller alongside the LM.

**Ablation experiment:** `diffppo_default` vs `diffppo_train_sampler`.

---

### Stage 3 — UnmaskRL Policy Size

| Parameter | `--unmask_d_model`, `--unmask_group_size` |
|-----------|------------------------------------------|
| Affects | Stage 3 (unmask_rl) |
| Defaults | `d_model=64`, `group_size=4` |

| Setting | `d_model` | `group_size` | Parameters |
|---------|-----------|-------------|------------|
| Small | 64 | 4 | ~16K |
| Large | 128 | 8 | ~66K |

**Ablation experiments:** `unmask_rl_small` vs `unmask_rl_large`.

---

### Inference Parameters

These affect both Stage 1 and Stage 2 (all inference through the API server).

| Parameter | Default | Suggested values | Description |
|-----------|---------|-----------------|-------------|
| `--gen_length` | 128 | 64, 128, 256, 512 | Generation length in tokens |
| `--num_steps` | 128 | 32, 64, 128, 256 | Diffusion denoising steps |
| `--block_length` | 32 | 8, 16, 32, 64 | Block size for block-wise denoising |
| `--temperature` | 0.0 | 0.0, 0.3, 0.5, 0.7, 1.0 | Sampling temperature (0 = greedy) |

**Ablation experiments:** `steps_32`, `steps_256`, `temp_0.5`, `temp_1.0`.

---

### Dataset

| Parameter | `--datasets` |
|-----------|-------------|
| Default | `gsm8k` |

Available datasets:

| Dataset | Type | Size | Description |
|---------|------|------|-------------|
| `gsm8k` | Math | 1.3K test | Grade school math word problems |
| `math` | Math | 5K test | Competition mathematics |
| `arc` | Reasoning | 1.2K test | AI2 Reasoning Challenge |
| `prontoqa` | Logic | ~1K test | Propositional logic QA |

**Ablation experiment:** `multi_dataset_eval` — evaluates across 4 datasets.

---

### Data Scale

| Parameter | `--num_samples` |
|-----------|----------------|
| Default | `200` |
| Suggested | 10, 50, 200, 500, -1 (full) |

Use small values (10-50) for quick validation, larger (200-500) for publication results, -1 for full evaluation.

---

### Model Precision

| Parameter | `--torch_dtype` |
|-----------|----------------|
| Default | `bfloat16` |
| Options | `bfloat16`, `float16`, `float32` |

The server also supports quantization (`--quantize 4bit/8bit`) for memory-constrained setups.

---

## Recommended Ablation Groups

### Group 1: Scheduler Comparison

**Goal:** Which unmasking scheduler works best for reasoning?

```bash
python scripts/run_ablation.py --experiments scheduler_compare
```

### Group 2: DAG Discovery Methods

**Goal:** Is template sweep sufficient, or does per-prompt search find better DAGs?

```bash
python scripts/run_ablation.py --experiments sweep_all_strategies search_greedy
```

### Group 3: Training Mode Comparison

**Goal:** Which training algorithm benefits most from DAG guidance?

```bash
python scripts/run_ablation.py \
    --experiments sft_per_template grpo_per_template diffppo_default unmask_rl_small \
    --episodes_from runs/ablation/sweep_all_strategies
```

### Group 4: DAG Mode Comparison

**Goal:** Is per-template grouping better than consensus or no-DAG?

```bash
python scripts/run_ablation.py \
    --experiments sft_per_template sft_consensus sft_no_dag \
    --episodes_from runs/ablation/sweep_all_strategies
```

### Group 5: DAG Bias Strength

**Goal:** How sensitive is training to the DAG bias strength?

```bash
python scripts/run_ablation.py \
    --experiments bias_0.0 bias_0.1 bias_0.3 bias_0.5 bias_0.8 bias_1.0 \
    --episodes_from runs/ablation/sweep_all_strategies
```

### Group 6: Cross-Dataset Generalisation

**Goal:** Does the model generalise to unseen datasets after DAG-aware training?

```bash
# Train on gsm8k
python scripts/run_ablation.py --experiments e2e_sft_per_template

# Evaluate on other datasets (after switch_model)
python scripts/run_research_pipeline.py \
    --stages 1 --datasets math arc prontoqa \
    --s1_schedulers confidence cot skeleton \
    --num_samples 200
```

### Group 7: Inference Parameter Sensitivity

**Goal:** How do num_steps, temperature affect accuracy?

```bash
python scripts/run_ablation.py --experiments steps_32 steps_256 temp_0.5 temp_1.0
```

---

## Output Structure

```
runs/ablation/
    scheduler_compare/
        stage1_baseline/
            gsm8k_confidence.json
            gsm8k_random.json
            ...
            summary.json
    sweep_all_strategies/
        stage1_baseline/...
        stage2_discovery/
            episodes.db
            best_dag_per_prompt.json
    sft_per_template/
        stage3_trained/
            pytorch_model.bin
            config.json
            tokenizer/
            template_stats.json
    ...
    ablation_summary.json     # Execution status of all experiments
    accuracy_table.json       # Cross-experiment accuracy comparison
    accuracy_table.csv        # Same data in CSV format
```
