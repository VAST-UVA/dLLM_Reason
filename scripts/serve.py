"""FastAPI inference server for dLLM-Reason.

Serves LLaDA model with configurable DAG strategies via REST API.
Supports hot-switching strategies without reloading the model.

Usage:
    python scripts/serve.py --model_id checkpoints/llada-instruct --port 8000
    python scripts/serve.py --model_id checkpoints/llada-instruct --quantize 4bit

API endpoints:
    POST /generate    — generate text with a given strategy
    GET  /strategies  — list available strategies
    GET  /health      — health check
"""

import argparse
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="dLLM-Reason", version="1.6.0")

# ── Version note ─────────────────────────────────────────────────────────────
# Install serving extras: pip install "dllm-reason[serve]"
# CLI entry point:        dllm-serve --model_id checkpoints/llada-instruct
# ─────────────────────────────────────────────────────────────────────────────

# Global model reference (loaded at startup)
_model = None
_model_id = ""


class GenerateRequest(BaseModel):
    prompt: str
    strategy: str = Field(default="confidence", description="Unmasking strategy")
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    num_steps: int = Field(default=128, ge=1, le=1024)
    block_length: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    cfg_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    remasking: str = Field(default="low_confidence")


class GenerateResponse(BaseModel):
    text: str
    strategy: str
    elapsed_seconds: float
    num_tokens: int


AVAILABLE_STRATEGIES = [
    "confidence", "random", "entropy", "semi_ar",
    "maskgit_cosine", "critical_token_first", "curriculum",
    "linear", "cot", "skeleton", "bidirectional", "answer_first",
    "adaptive_dynamic",
]


def build_scheduler(strategy: str, gen_len: int, block_length: int, device):
    """Build a scheduler from strategy name."""
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
    from dllm_reason.scheduler.random_scheduler import RandomScheduler
    from dllm_reason.scheduler.linear_scheduler import LinearScheduler
    from dllm_reason.scheduler.entropy_scheduler import EntropyScheduler
    from dllm_reason.scheduler.semi_ar_scheduler import SemiAutoregressiveScheduler
    from dllm_reason.scheduler.maskgit_scheduler import MaskGITCosineScheduler
    from dllm_reason.scheduler.critical_token_scheduler import CriticalTokenFirstScheduler
    from dllm_reason.scheduler.curriculum_scheduler import CurriculumScheduler
    from dllm_reason.scheduler.adaptive_dynamic_scheduler import AdaptiveDynamicScheduler

    schedulers = {
        "confidence": lambda: ConfidenceScheduler(),
        "random": lambda: RandomScheduler(),
        "linear": lambda: LinearScheduler(),
        "entropy": lambda: EntropyScheduler(),
        "semi_ar": lambda: SemiAutoregressiveScheduler(block_size=block_length),
        "maskgit_cosine": lambda: MaskGITCosineScheduler(),
        "critical_token_first": lambda: CriticalTokenFirstScheduler(),
        "curriculum": lambda: CurriculumScheduler(),
        "adaptive_dynamic": lambda: AdaptiveDynamicScheduler(),
    }

    if strategy in schedulers:
        return schedulers[strategy]()

    # DAG-based strategies
    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag, answer_first_dag,
    )

    if strategy == "cot":
        dag = chain_of_thought_dag(gen_len, num_steps=4, device=device)
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "skeleton":
        dag = skeleton_then_detail_dag(
            gen_len, list(range(0, gen_len, 3)), list(range(1, gen_len, 3)), device=device,
        )
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "bidirectional":
        dag = bidirectional_dag(gen_len, num_segments=4, device=device)
        return DAGScheduler(dag, sub_strategy="confidence_topk")
    elif strategy == "answer_first":
        dag = answer_first_dag(
            gen_len, list(range(int(gen_len * 0.8), gen_len)), device=device,
        )
        return DAGScheduler(dag, sub_strategy="confidence_topk")

    raise ValueError(f"Unknown strategy: {strategy}")


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_id, "device": str(_model.device if _model else "none")}


@app.get("/strategies")
def strategies():
    return {"strategies": AVAILABLE_STRATEGIES}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    if req.strategy not in AVAILABLE_STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy}. Available: {AVAILABLE_STRATEGIES}")

    scheduler = build_scheduler(req.strategy, req.max_new_tokens, req.block_length, _model.device)

    t0 = time.time()
    text = _model.generate(
        prompt=req.prompt,
        generation_len=req.max_new_tokens,
        block_length=req.block_length,
        scheduler=scheduler,
        num_steps=req.num_steps,
        temperature=req.temperature,
        cfg_scale=req.cfg_scale,
        remasking=req.remasking,
        system_prompt=req.system_prompt,
    )
    elapsed = time.time() - t0

    return GenerateResponse(
        text=text,
        strategy=req.strategy,
        elapsed_seconds=round(elapsed, 3),
        num_tokens=len(text.split()),
    )


def main():
    global _model, _model_id

    parser = argparse.ArgumentParser(description="dLLM-Reason inference server")
    parser.add_argument("--model_id", type=str, default="checkpoints/llada-instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["4bit", "8bit"],
                        help="Load model with quantization (requires bitsandbytes)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    _model_id = args.model_id

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    print(f"Loading model: {args.model_id}")

    # Build quantization config if requested
    quant_config = None
    if args.quantize == "4bit":
        print("Loading with 4-bit quantization (bitsandbytes)")
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map[args.torch_dtype],
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantize == "8bit":
        print("Loading with 8-bit quantization (bitsandbytes)")
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    from dllm_reason.models.llada import LLaDAWrapper
    _model = LLaDAWrapper(
        model_id=args.model_id,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        quantization_config=quant_config,
    )

    print(f"Model loaded on {_model.device}, serving at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
