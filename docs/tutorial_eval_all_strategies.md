# Tutorial: Running All-Strategy Evaluation via pip

This tutorial shows how to install **dLLM-Reason** from GitHub and run the full
all-strategies benchmark evaluation (`eval_dags.py --dags all`).

---

## 1. Environment setup

```bash
# Recommended: create a fresh conda env
conda create -n dllm python=3.11 -y
conda activate dllm

# Install PyTorch first (match your CUDA version)
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU only (for testing)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 2. Install dllm-reason

```bash
pip install "git+https://github.com/BDeMo/dLLM_Reason.git"
```

To also get dev tools (pytest, ruff) and retrieval extras (FAISS, sentence-transformers):

```bash
pip install "dllm-reason[dev,library] @ git+https://github.com/BDeMo/dLLM_Reason.git"
```

Verify the install:

```bash
python -c "import dllm_reason; print(dllm_reason.__version__)"
# → 1.6.0

dllm-eval-dags --help
```

---

## 3. Download model & datasets

```bash
# Clone the repo to get the helper scripts (only needed once)
git clone https://github.com/BDeMo/dLLM_Reason.git
cd dLLM_Reason

# Download LLaDA-8B-Instruct → checkpoints/llada-instruct/
python scripts/download_models.py

# Download benchmark datasets → datasets/
python scripts/download_datasets.py

# ── China mirror (HuggingFace is blocked) ──────────────────────────────
python scripts/download_models.py   --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com
```

---

## 4. Quick smoke-test (5 samples, confidence DAG only)

Before the full run, verify everything works end-to-end:

```bash
python scripts/eval_dags.py \
    --dags confidence \
    --benchmarks mbpp \
    --num_samples 5 \
    --output_dir results/smoke_test
```

Expected output:

```
[config] Loaded: configs/eval_default.yaml (auto-detected)
Running DAG strategy: confidence
  mbpp  — 5/5 samples ... pass@1 = 0.800
Results saved → results/smoke_test/
```

---

## 5. Run all strategies

### Option A — convenience shell script

```bash
bash scripts/runs/all_strategies.sh
```

This is equivalent to Option B below.
Results go to `results/all_strategies_<timestamp>/`.

### Option B — direct Python call

```bash
python scripts/eval_dags.py \
    --dags confidence random entropy semi_ar maskgit_cosine critical_token_first \
          curriculum linear cot skeleton bidirectional answer_first adaptive_dynamic \
    --output_dir results/all_strategies_$(date +%Y%m%d_%H%M%S)
```

### Option C — CLI entry point (after pip install)

```bash
dllm-eval-dags \
    --dags confidence random entropy semi_ar maskgit_cosine critical_token_first \
          curriculum linear cot skeleton bidirectional answer_first adaptive_dynamic \
    --output_dir results/all_strategies
```

---

## 6. Common CLI flags

All flags below override the corresponding key in `configs/eval_default.yaml`.

### Benchmarks

| Flag | Default | Options |
|------|---------|---------|
| `--benchmarks` | `mbpp humaneval` | `mbpp humaneval hotpotqa mmlu` |
| `--num_samples` | `null` (full dataset) | any integer, e.g. `100` |
| `--no_run_tests` | off | skip code execution, generate only |
| `--verbose_errors` | off | print per-sample stderr |

### Model & inference

| Flag | Default | Notes |
|------|---------|-------|
| `--model_id` | `checkpoints/llada-instruct` | HF repo ID or local path |
| `--torch_dtype` | `bfloat16` | `bfloat16 \| float16 \| float32` |
| `--num_steps` | `128` | diffusion denoising steps |
| `--block_length` | `32` | tokens per block (`max_new_tokens` must be divisible) |
| `--temperature` | `0.0` | `0` = greedy argmax |
| `--cfg_scale` | `0.0` | classifier-free guidance; `0` = disabled |
| `--remasking` | `low_confidence` | `low_confidence \| random` |
| `--max_new_tokens` | `128` | max generation length |

### DAG strategies

| Flag | Default | Notes |
|------|---------|-------|
| `--dags` | `confidence` | space-separated list from the 13 strategies below |
| `--cot_steps` | `4` | reasoning segments for the `cot` DAG |
| `--mmlu_subjects` | `null` | restrict MMLU to specific subjects |

**Available DAG strategies:**

| Strategy | Description |
|----------|-------------|
| `confidence` | Unmask tokens in order of model confidence (highest first) |
| `random` | Uniformly random unmasking order (no DAG constraint, LLaDA baseline) |
| `linear` | Left-to-right sequential unmasking |
| `cot` | Chain-of-thought segments: reasoning blocks before answer |
| `skeleton` | Sketch key tokens first, then fill in details |
| `bidirectional` | Alternates from both ends toward the center |
| `answer_first` | Generate the answer span first, then supporting tokens |
| `entropy` | Unmask lowest-entropy (most certain by distribution) positions first |
| `semi_ar` | Semi-autoregressive: block-by-block L→R, confidence within block |
| `maskgit_cosine` | MaskGIT cosine schedule: more tokens early, fewer later |
| `critical_token_first` | Unmask most influential (highest KL from uniform) positions first |
| `curriculum` | Easy tokens first (high confidence + low entropy), hard tokens last |
| `adaptive_dynamic` | Dynamic soft DAG: runtime pairwise influence graph (**novel**) |

### Output saving

| Flag | Default | Notes |
|------|---------|-------|
| `--save_outputs` | off | **master switch** — enables all file saving |
| `--no_save_qa` | off | exclude prompt + generated answer |
| `--no_save_ground_truth` | off | exclude reference answers |
| `--record_trajectory` | off | save per-step unmasking states (**large/slow**) |
| `--output_formats` | `json xlsx` | `json` and/or `xlsx` |

### Misc

| Flag | Default | Notes |
|------|---------|-------|
| `--output_dir` | `results` | directory for all output files |
| `--resume` | off | skip strategies whose result JSON already exists |
| `--config` | auto-detected | path to a custom YAML config file |

---

## 7. Save per-sample outputs

```bash
python scripts/eval_dags.py \
    --dags confidence random entropy semi_ar maskgit_cosine critical_token_first \
          curriculum linear cot skeleton bidirectional answer_first adaptive_dynamic \
    --save_outputs \
    --output_dir results/all_with_outputs
```

Or use the dedicated script (with commented options):

```bash
# Edit scripts/runs/save_outputs.sh to toggle what to save, then:
bash scripts/runs/save_outputs.sh
```

Output files produced (one set per strategy):

```
results/all_with_outputs/
├── confidence_samples.json       # all samples: prompt, generated, ground truth, score
├── confidence_samples.xlsx       # same content, Excel format
├── random_samples.json
├── random_samples.xlsx
├── ...
└── summary.json                  # aggregated scores across all strategies
```

If `--record_trajectory` is also set:

```
├── confidence_trajectory.json    # per-step token states for every sample (large)
```

---

## 8. Use a custom config file

Copy and edit the default config:

```bash
cp configs/eval_default.yaml configs/my_run.yaml
# Edit configs/my_run.yaml as needed
python scripts/eval_dags.py --config configs/my_run.yaml
```

CLI flags always override the config file:

```bash
# Config says num_samples: null, but CLI forces 50
python scripts/eval_dags.py --config configs/my_run.yaml --num_samples 50
```

---

## 9. Expected output structure

```
results/all_strategies_20260407_120000/
├── confidence_results.json
├── random_results.json
├── linear_results.json
├── entropy_results.json
├── semi_ar_results.json
├── maskgit_cosine_results.json
├── critical_token_first_results.json
├── curriculum_results.json
├── adaptive_dynamic_results.json
├── cot_results.json
├── skeleton_results.json
├── bidirectional_results.json
├── answer_first_results.json
└── summary.json          ← comparison table across all strategies
```

`summary.json` example:

```json
{
  "mbpp": {
    "confidence":   {"pass@1": 0.812},
    "random":       {"pass@1": 0.734},
    "linear":       {"pass@1": 0.756},
    "entropy":      {"pass@1": 0.788},
    "semi_ar":      {"pass@1": 0.779},
    "cot":          {"pass@1": 0.789},
    "skeleton":     {"pass@1": 0.798},
    "bidirectional":{"pass@1": 0.743},
    "answer_first": {"pass@1": 0.771}
  },
  "humaneval": { ... }
}
```

---

## 10. Reproduce paper results

```bash
python scripts/eval_dags.py \
    --dags confidence random entropy semi_ar maskgit_cosine critical_token_first \
          curriculum linear cot skeleton bidirectional answer_first adaptive_dynamic \
    --benchmarks mbpp humaneval hotpotqa mmlu gsm8k math arc prontoqa gpqa aime \
    --num_steps 128 \
    --max_new_tokens 128 \
    --temperature 0.0 \
    --cfg_scale 0.0 \
    --remasking low_confidence \
    --save_outputs \
    --output_dir results/paper_$(date +%Y%m%d)
```

Estimated runtime on a single A100 (80 GB): ~6–8 hours for the full dataset.

Use `--num_samples 200` for a faster representative subset (~45 min).
