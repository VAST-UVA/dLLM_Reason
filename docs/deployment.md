# Deployment Guide

This guide covers deploying dLLM-Reason as a REST API service for inference.

---

## 1. Installation

```bash
# Core + serving extras (FastAPI, uvicorn, bitsandbytes)
pip install "dllm-reason[serve]"

# Or from GitHub
pip install "dllm-reason[serve] @ git+https://github.com/BDeMo/dLLM_Reason.git"
```

---

## 2. Quick Start

```bash
# Start with default settings (bfloat16, port 8000)
dllm-serve --model_id checkpoints/llada-instruct

# Or equivalently:
python scripts/serve.py --model_id checkpoints/llada-instruct
```

The server exposes three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate text with a given unmasking strategy |
| `/strategies` | GET | List all 13 available strategies |
| `/health` | GET | Health check (model status, device info) |

---

## 3. Quantization

Reduce memory usage with 4-bit or 8-bit quantization via bitsandbytes:

```bash
# 4-bit NF4 quantization (~5 GB VRAM for 8B model)
dllm-serve --model_id GSAI-ML/LLaDA-8B-Instruct --quantize 4bit

# 8-bit quantization (~9 GB VRAM for 8B model)
dllm-serve --model_id GSAI-ML/LLaDA-8B-Instruct --quantize 8bit

# Full precision bfloat16 (~16 GB VRAM for 8B model)
dllm-serve --model_id GSAI-ML/LLaDA-8B-Instruct
```

Requirements for quantization:
- CUDA GPU (bitsandbytes does not support CPU or MPS)
- `bitsandbytes>=0.41.0` (installed automatically with `[serve]` extra)

---

## 4. API Usage

### Generate text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Solve step by step: What is 15 * 23?",
    "strategy": "confidence",
    "max_new_tokens": 256,
    "num_steps": 128,
    "temperature": 0.0
  }'
```

Response:

```json
{
  "text": "15 * 23 = 345",
  "strategy": "confidence",
  "elapsed_seconds": 2.341,
  "num_tokens": 42
}
```

### Hot-switch strategies

No model reload needed — just change the `strategy` field:

```bash
# Try different strategies on the same prompt
for strategy in confidence entropy adaptive_dynamic cot; do
  curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"What is 7*8?\", \"strategy\": \"$strategy\"}" \
    | python -m json.tool
done
```

### Request parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | User message |
| `strategy` | string | `"confidence"` | Unmasking strategy (see `/strategies`) |
| `system_prompt` | string | null | Optional system message |
| `max_new_tokens` | int | 128 | Max tokens to generate (1-2048) |
| `num_steps` | int | 128 | Denoising steps (1-1024) |
| `block_length` | int | 32 | Tokens per denoising block (1-512) |
| `temperature` | float | 0.0 | Sampling temperature (0=greedy) |
| `cfg_scale` | float | 0.0 | Classifier-free guidance (0=disabled) |
| `remasking` | string | `"low_confidence"` | Remasking strategy |

---

## 5. Production Deployment

### Behind nginx

```nginx
upstream dllm {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://dllm;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;  # LLM inference can be slow
    }
}
```

### With Docker

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .
RUN pip install -e ".[serve]"

# Download model at build time (or mount as volume)
# RUN python scripts/download_models.py

EXPOSE 8000
CMD ["dllm-serve", "--model_id", "checkpoints/llada-instruct", "--quantize", "4bit"]
```

```bash
docker build -t dllm-reason .
docker run --gpus all -p 8000:8000 -v ./checkpoints:/app/checkpoints dllm-reason
```

### Multi-GPU

The server uses `device_map="auto"` by default, which distributes layers across all available GPUs via HuggingFace Accelerate. No extra configuration needed.

```bash
# Uses all visible GPUs
CUDA_VISIBLE_DEVICES=0,1 dllm-serve --model_id GSAI-ML/LLaDA-8B-Instruct
```

---

## 6. Python Client

```python
import requests

def generate(prompt, strategy="confidence", **kwargs):
    resp = requests.post("http://localhost:8000/generate", json={
        "prompt": prompt,
        "strategy": strategy,
        **kwargs,
    })
    resp.raise_for_status()
    return resp.json()

result = generate("What is 7 * 8?", strategy="adaptive_dynamic", max_new_tokens=256)
print(result["text"])
print(f"Strategy: {result['strategy']}, Time: {result['elapsed_seconds']}s")
```

---

## 7. Available Strategies

All 13 strategies can be hot-switched via the `strategy` parameter:

| Strategy | Description |
|----------|-------------|
| `confidence` | Highest model confidence first (LLaDA default) |
| `random` | Uniform random unmasking |
| `entropy` | Lowest entropy (most certain) first |
| `semi_ar` | Block-by-block left-to-right |
| `maskgit_cosine` | MaskGIT cosine schedule |
| `critical_token_first` | Highest KL divergence from uniform |
| `curriculum` | Easy tokens first, hard tokens last |
| `linear` | Strict left-to-right |
| `cot` | Chain-of-thought DAG |
| `skeleton` | Skeleton tokens first, then detail |
| `bidirectional` | Both ends toward center |
| `answer_first` | Answer region first |
| `adaptive_dynamic` | Dynamic soft DAG (novel) |
