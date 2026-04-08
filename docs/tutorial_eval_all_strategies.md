# Tutorial: dLLM-Reason v1.4.0 — Complete Usage Guide

This guide covers every feature of dLLM-Reason: installation, evaluation,
serving, Web UI, DAG search, and paper reproduction.

---

## 1. Environment Setup

```bash
conda create -n dllm python=3.11 -y
conda activate dllm

# Install PyTorch (match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only
```

---

## 2. Install dLLM-Reason

```bash
# Option A: from PyPI (stable release)
pip install dllm-reason

# Option B: from GitHub (latest dev)
pip install "git+https://github.com/BDeMo/dLLM_Reason.git"

# With all optional extras
pip install "dllm-reason[dev,library,serve]"
```

Verify:

```bash
python -c "import dllm_reason; print(dllm_reason.__version__)"
# → 1.4.0

dllm-eval-dags --help
```

### Available extras

| Extra | Installs | For |
|-------|----------|-----|
| `dev` | pytest, ruff | testing / linting |
| `library` | faiss-cpu, sentence-transformers, scikit-learn | DAG Library retrieval |
| `serve` | fastapi, uvicorn, bitsandbytes | REST API / Web UI / quantization |

---

## 3. Download Model & Datasets

```bash
git clone https://github.com/BDeMo/dLLM_Reason.git
cd dLLM_Reason

python scripts/download_models.py                              # → checkpoints/llada-instruct/
python scripts/download_datasets.py                            # → datasets/

# China HuggingFace mirror
python scripts/download_models.py   --mirror https://hf-mirror.com
python scripts/download_datasets.py --mirror https://hf-mirror.com
```

---

## 4. Smoke Test (< 2 min)

```bash
dllm-eval-dags \
    --dags confidence \
    --benchmarks mbpp \
    --num_samples 5 \
    --output_dir results/smoke_test
```

Expected:

```
[config] Loaded: configs/eval_default.yaml (auto-detected)
Running DAG strategy: confidence
  mbpp — 5/5 samples ... pass@1 = 0.800
Saved: results/smoke_test/mbpp_confidence.json
```

---

## 5. Evaluate All 13 Strategies

### One-liner

```bash
bash scripts/runs/full_comparison.sh
```

This runs **13 strategies x 10 benchmarks** and saves everything to
`results/full_comparison_<timestamp>/`.

### Or pick what you need

```bash
dllm-eval-dags \
    --dags confidence random entropy semi_ar maskgit_cosine \
          critical_token_first curriculum linear \
          cot skeleton bidirectional answer_first adaptive_dynamic \
    --benchmarks gsm8k math mbpp humaneval arc mmlu hotpotqa prontoqa gpqa aime \
    --num_steps 128 \
    --num_samples 100 \
    --save_outputs \
    --output_dir results/my_run
```

### Per-strategy convenience scripts

```bash
bash scripts/runs/confidence.sh          # highest-confidence first (LLaDA default)
bash scripts/runs/random.sh              # uniform random
bash scripts/runs/entropy.sh             # lowest-entropy first
bash scripts/runs/semi_ar.sh             # block-by-block L→R
bash scripts/runs/linear.sh              # strict left-to-right
bash scripts/runs/cot.sh                 # Chain-of-Thought DAG
bash scripts/runs/skeleton.sh            # skeleton-then-detail DAG
bash scripts/runs/bidirectional.sh       # both ends toward center
bash scripts/runs/answer_first.sh        # answer region first
bash scripts/runs/all_strategies.sh      # all 13 in one run
bash scripts/runs/full_comparison.sh     # 13 strategies x 10 benchmarks
```

All scripts forward extra arguments:

```bash
bash scripts/runs/cot.sh --benchmarks gsm8k math --num_samples 50 --cot_steps 6
```

### 13 strategies

| Strategy | Type | Description |
|----------|------|-------------|
| `confidence` | Flat | Highest model confidence first (LLaDA default) |
| `random` | Flat | Uniform random |
| `entropy` | Flat | Lowest-entropy (most certain) first |
| `semi_ar` | Flat | Block-by-block L→R, confidence within block |
| `maskgit_cosine` | Flat | MaskGIT cosine schedule |
| `critical_token_first` | Flat | Highest KL from uniform first |
| `curriculum` | Flat | Easy tokens first, hard last |
| `linear` | Flat | Strict left-to-right |
| `cot` | DAG | Chain-of-Thought segments |
| `skeleton` | DAG | Structure first, detail later |
| `bidirectional` | DAG | Both ends toward center |
| `answer_first` | DAG | Answer region first |
| `adaptive_dynamic` | Dynamic | Pairwise influence graph at runtime (**novel**) |

### 10 benchmarks

| Benchmark | Type | Metric |
|-----------|------|--------|
| `mbpp` | Code | pass@1 |
| `humaneval` | Code | pass@1 |
| `gsm8k` | Math | exact match |
| `math` | Math (competition) | exact match |
| `arc` | Science | accuracy |
| `mmlu` | Knowledge | accuracy |
| `hotpotqa` | Multi-hop QA | EM / F1 |
| `prontoqa` | Logic | accuracy |
| `gpqa` | PhD-level science | accuracy |
| `aime` | Competition math | accuracy |

---

## 6. CLI Reference

### Benchmarks & samples

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmarks` | `mbpp humaneval` | Space-separated benchmark list |
| `--num_samples` | `null` (all) | Number of samples per benchmark |
| `--no_run_tests` | off | Skip code execution (generate only) |
| `--verbose_errors` | off | Print per-sample stderr |

### Model & inference

| Flag | Default | Description |
|------|---------|-------------|
| `--model_id` | `checkpoints/llada-instruct` | HF repo ID or local path |
| `--torch_dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--num_steps` | `128` | Diffusion denoising steps |
| `--block_length` | `32` | Tokens per block |
| `--max_new_tokens` | `128` | Max generation length |
| `--temperature` | `0.0` | 0 = greedy argmax |
| `--cfg_scale` | `0.0` | Classifier-free guidance (0 = off) |
| `--remasking` | `low_confidence` | `low_confidence` / `random` |

### Strategies & DAG

| Flag | Default | Description |
|------|---------|-------------|
| `--dags` | `confidence` | Space-separated strategy list (13 choices) |
| `--cot_steps` | `4` | Reasoning segments for `cot` |
| `--mmlu_subjects` | `null` | Restrict MMLU subjects |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `results` | Output directory |
| `--save_outputs` | off | Write per-sample JSON + Excel |
| `--record_trajectory` | off | Write per-step unmasking states |
| `--output_formats` | `json xlsx` | `json` and/or `xlsx` |
| `--resume` | off | Skip already-completed runs |
| `--config` | auto-detected | Custom YAML config path |

---

## 7. Save Detailed Outputs

```bash
dllm-eval-dags \
    --dags confidence cot adaptive_dynamic \
    --benchmarks gsm8k mbpp \
    --save_outputs \
    --record_trajectory \
    --num_samples 20 \
    --output_dir results/detailed
```

Output:

```
results/detailed/
├── gsm8k_confidence.json          # summary metrics
├── gsm8k_confidence_samples.json  # per-sample: prompt, generated, ground truth, score
├── gsm8k_confidence_samples.xlsx  # same as spreadsheet
├── gsm8k_confidence_trajectory.json  # per-step unmasking states
├── gsm8k_cot.json
├── gsm8k_cot_samples.json
├── ...
└── summary.json                   # all strategies x all benchmarks
```

---

## 8. REST API Server

```bash
pip install "dllm-reason[serve]"

# Start (bfloat16)
dllm-serve --model_id checkpoints/llada-instruct --port 8000

# With 4-bit quantization (~5 GB VRAM)
dllm-serve --model_id checkpoints/llada-instruct --quantize 4bit

# With 8-bit quantization (~9 GB VRAM)
dllm-serve --model_id checkpoints/llada-instruct --quantize 8bit
```

### Endpoints

```bash
# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is 7*8?","strategy":"confidence","max_new_tokens":128}'

# List strategies
curl http://localhost:8000/strategies

# Health check
curl http://localhost:8000/health
```

### Python client

```python
import requests

r = requests.post("http://localhost:8000/generate", json={
    "prompt": "Solve: 15 * 23",
    "strategy": "adaptive_dynamic",
    "max_new_tokens": 256,
})
print(r.json()["text"])
```

---

## 9. Web UI (Interactive Dashboard)

```bash
pip install "dllm-reason[serve]"

# Full mode (model + UI)
dllm-webui --model_id checkpoints/llada-instruct --port 7860

# With quantization
dllm-webui --model_id checkpoints/llada-instruct --quantize 4bit --port 7860

# Results viewer only (no GPU needed)
dllm-webui --no_model --port 7860
```

Open `http://localhost:7860` in your browser.

| Tab | Description |
|-----|-------------|
| **Generate** | Type a prompt, pick a strategy, adjust parameters, generate |
| **Compare** | Select multiple strategies, compare outputs side-by-side |
| **Trajectory** | Watch tokens unmask step-by-step |
| **Results** | Browse `results/` directory, view benchmark comparison tables |

---

## 10. DAG Structure Search

Search for optimal DAG structures that maximize task performance.

### Convenience scripts

```bash
bash scripts/runs/search_greedy.sh             # greedy edge add/remove
bash scripts/runs/search_evolutionary.sh       # population-based evolution
bash scripts/runs/search_rl.sh                 # RL policy (REINFORCE)
bash scripts/runs/search_differentiable.sh     # NOTEARS continuous relaxation
bash scripts/runs/search_e2e.sh                # end-to-end DAG learning (v1.4.0)
bash scripts/runs/search_nas.sh                # NAS-style search (v1.4.0)
```

### Python API

```python
# End-to-end DAG learning
from dllm_reason.search.e2e_dag_learner import E2EDAGLearner, E2EConfig
learner = E2EDAGLearner(config=E2EConfig(lr_dag=3e-3, tau_start=1.0, tau_end=0.1))
result = learner.search(model, eval_fn, seq_len=256, budget=200)
print(f"Best fitness: {result.best_fitness}, edges: {result.best_dag.num_edges()}")

# NAS SuperNet (DARTS-like)
from dllm_reason.search.nas_search import NASDAGSearch, NASConfig
searcher = NASDAGSearch(config=NASConfig(mode="supernet", span_size=16))
result = searcher.search(model, eval_fn, seq_len=256, budget=200)

# NAS Controller (ENAS-like)
searcher = NASDAGSearch(config=NASConfig(mode="controller", span_size=16))
result = searcher.search(model, eval_fn, seq_len=256, budget=200)

# Evolutionary (with DAG Library seeding)
from dllm_reason.search.evolutionary import EvolutionarySearch
searcher = EvolutionarySearch(population_size=20, library=dag_store)
result = searcher.search(model, eval_fn, seq_len=256, budget=200)
```

| Method | Type | Description |
|--------|------|-------------|
| `greedy` | Black-box | Add/remove edges iteratively |
| `evolutionary` | Black-box | Tournament selection + crossover + mutation |
| `rl_policy` | Black-box | GRU controller + REINFORCE |
| `differentiable` | Gradient | NOTEARS augmented Lagrangian |
| `e2e` | Gradient | Joint DAG + task loss (Gumbel-Sigmoid) |
| `nas` | Gradient/RL | DARTS supernet or ENAS controller |

---

## 11. LaTeX Table

```bash
python scripts/generate_latex_table.py results/full_comparison_*/summary.json \
    --output paper_table.tex
```

Generates a booktabs table with bold for best-per-column, ready for submission.

---

## 12. Config File

All defaults live in `configs/eval_default.yaml`. CLI flags always override.

```bash
# Use custom config
cp configs/eval_default.yaml configs/my_run.yaml
# Edit as needed ...
dllm-eval-dags --config configs/my_run.yaml

# CLI override
dllm-eval-dags --config configs/my_run.yaml --num_samples 50
```

---

## 13. Reproduce Paper Results (Full)

**One-command full reproduction** (A100 80GB, ~6-8h):

```bash
bash scripts/runs/full_comparison.sh
```

Or manually:

```bash
dllm-eval-dags \
    --dags confidence random entropy semi_ar maskgit_cosine \
          critical_token_first curriculum linear \
          cot skeleton bidirectional answer_first adaptive_dynamic \
    --benchmarks gsm8k math mbpp humaneval arc mmlu hotpotqa prontoqa gpqa aime \
    --num_steps 128 --max_new_tokens 128 \
    --temperature 0.0 --cfg_scale 0.0 --remasking low_confidence \
    --save_outputs \
    --output_dir results/paper_$(date +%Y%m%d)
```

**Faster subset** (~45 min):

```bash
dllm-eval-dags \
    --dags confidence entropy cot adaptive_dynamic \
    --benchmarks gsm8k mbpp humaneval \
    --num_samples 200 \
    --save_outputs \
    --output_dir results/quick_$(date +%Y%m%d)
```

Then generate the LaTeX table:

```bash
python scripts/generate_latex_table.py results/paper_*/summary.json --output paper_table.tex
```

---

## 14. Quick-Copy Commands

```bash
# === Install ===
pip install dllm-reason                                       # PyPI
pip install "dllm-reason[serve]"                              # + API/WebUI
pip install "dllm-reason[dev,library,serve]"                  # everything

# === Evaluate ===
dllm-eval-dags --dags confidence --benchmarks mbpp --num_samples 5  # smoke test
bash scripts/runs/all_strategies.sh                           # all 13 strategies
bash scripts/runs/full_comparison.sh                          # 13 x 10 full

# === Serve ===
dllm-serve --model_id checkpoints/llada-instruct --quantize 4bit
dllm-webui --model_id checkpoints/llada-instruct --port 7860

# === Search ===
bash scripts/runs/search_evolutionary.sh
bash scripts/runs/search_e2e.sh
bash scripts/runs/search_nas.sh

# === Generate tables ===
python scripts/generate_latex_table.py results/*/summary.json --output table.tex
```
