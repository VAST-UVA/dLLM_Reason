"""Training entry point for dLLM models.

Supports 4 training modes: pretrain, finetune, dag_aware, rl.

Usage:
    # Pretrain MDLM on GSM8K:
    dllm-train --model mdlm --dataset gsm8k --mode pretrain \
        --output_dir checkpoints/mdlm_gsm8k

    # Fine-tune LLaDA on GSM8K (answer-only loss):
    dllm-train --model llada --dataset gsm8k --mode finetune \
        --checkpoint GSAI-ML/LLaDA-8B-Instruct \
        --loss_on_answer_only \
        --output_dir checkpoints/llada_gsm8k_finetuned

    # DAG-aware training (biased masking to match inference):
    dllm-train --model mdlm --dataset gsm8k --mode dag_aware \
        --dag_type cot --dag_bias_strength 0.5 \
        --output_dir checkpoints/mdlm_gsm8k_dag

    # RL training (DiffuGRPO):
    dllm-train --model mdlm --dataset gsm8k --mode rl \
        --output_dir checkpoints/mdlm_gsm8k_rl
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dLLM model")

    # Model
    parser.add_argument("--model", type=str, default="mdlm",
                        choices=["mdlm", "sedd", "d3pm", "llada"],
                        help="Model to train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path or HF ID to load pretrained weights from")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Data
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "arc", "prontoqa"])
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_train_samples", type=int, default=None)

    # Training mode
    parser.add_argument("--mode", type=str, default="pretrain",
                        choices=["pretrain", "finetune", "dag_aware", "rl"])
    parser.add_argument("--loss_on_answer_only", action="store_true",
                        help="Compute loss only on answer positions (fine-tuning)")

    # DAG-aware training
    parser.add_argument("--dag_aware", action="store_true",
                        help="Use DAG-biased masking during training")
    parser.add_argument("--dag_type", type=str, default="cot",
                        choices=["cot", "skeleton", "linear", "bidirectional"])
    parser.add_argument("--dag_cot_steps", type=int, default=4)
    parser.add_argument("--dag_bias_strength", type=float, default=0.5)

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/run")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")

    return parser.parse_args()


def build_model(args, vocab_size: int):
    from dllm_reason.utils.registry import MODEL_REGISTRY
    # Trigger imports to populate registry
    import dllm_reason.models.mdlm
    import dllm_reason.models.sedd
    import dllm_reason.models.d3pm

    model_cls = MODEL_REGISTRY.get(args.model)
    model = model_cls(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {args.checkpoint}")

    return model


def build_dag(args, seq_len: int, device):
    from dllm_reason.graph.templates import (
        chain_of_thought_dag, skeleton_then_detail_dag,
        bidirectional_dag,
    )
    from dllm_reason.graph.dag import TokenDAG

    if args.dag_type == "cot":
        return chain_of_thought_dag(seq_len, args.dag_cot_steps, device=device)
    elif args.dag_type == "skeleton":
        structural = list(range(0, seq_len, 3))
        detail = list(range(1, seq_len, 3))
        return skeleton_then_detail_dag(seq_len, structural, detail, device=device)
    elif args.dag_type == "linear":
        return TokenDAG.linear_chain(seq_len, device=device)
    elif args.dag_type == "bidirectional":
        return bidirectional_dag(seq_len, device=device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer + Data ─────────────────────────────────────────────────────
    from dllm_reason.data.tokenizer import get_tokenizer
    from dllm_reason.data.reasoning_datasets import load_reasoning_dataset, ReasoningDataset
    from dllm_reason.data.collator import DiffusionCollator

    tokenizer = get_tokenizer(
        args.checkpoint if args.model == "llada" else args.tokenizer,
        add_mask_token=True,
    )
    vocab_size = len(tokenizer)

    print(f"Loading dataset: {args.dataset}")
    train_data = load_reasoning_dataset(args.dataset, split="train")
    val_data = load_reasoning_dataset(args.dataset, split="test")

    if args.max_train_samples:
        train_data = train_data[:args.max_train_samples]

    mask_token_id = tokenizer.mask_token_id or vocab_size

    train_ds = ReasoningDataset(train_data, tokenizer, max_seq_len=args.max_seq_len)
    val_ds = ReasoningDataset(val_data[:500], tokenizer, max_seq_len=args.max_seq_len)

    collator = DiffusionCollator(mask_token_id=mask_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator, num_workers=2)

    # ── Model ────────────────────────────────────────────────────────────────
    if args.model == "llada":
        from dllm_reason.models.llada import LLaDAWrapper
        model = LLaDAWrapper(
            model_id=args.checkpoint or "GSAI-ML/LLaDA-8B-Instruct",
            max_seq_len=args.max_seq_len,
        )
    else:
        model = build_model(args, vocab_size)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    from dllm_reason.training.pretrain import TrainConfig
    from dllm_reason.training.finetune import FinetuneConfig, Finetuner

    train_cfg_cls = FinetuneConfig if args.mode == "finetune" else TrainConfig
    train_cfg = train_cfg_cls(
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        save_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )
    if args.mode == "finetune":
        train_cfg.loss_on_answer_only = args.loss_on_answer_only

    if args.dag_aware or args.mode == "dag_aware":
        from dllm_reason.training.dag_aware_train import DAGAwareTrainer
        dag = build_dag(args, args.max_seq_len, device)
        trainer = DAGAwareTrainer(
            model, train_loader, dag,
            dag_bias_strength=args.dag_bias_strength,
            val_loader=val_loader,
            config=train_cfg,
        )
        print(f"DAG-aware training with {args.dag_type} DAG (bias={args.dag_bias_strength})")
    elif args.mode == "finetune":
        trainer = Finetuner(model, train_loader, val_loader, train_cfg)
    elif args.mode == "rl":
        from dllm_reason.training.rl_train import DiffuGRPO, RLTrainConfig
        from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler
        # RL requires a reference model (frozen copy)
        import copy
        ref_model = copy.deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad = False
        rl_cfg = RLTrainConfig(lr=args.lr, num_iterations=args.max_steps)
        # Default reward: placeholder (user should override via config)
        def _placeholder_reward(seq, batch):
            return 0.0
        trainer = DiffuGRPO(
            model, ref_model, ConfidenceScheduler(),
            reward_fn=_placeholder_reward,
            train_loader=train_loader,
            config=rl_cfg,
        )
        print("RL training (DiffuGRPO) — provide reward_fn via Python API for real use")
    else:
        from dllm_reason.training.pretrain import Trainer
        trainer = Trainer(model, train_loader, val_loader, train_cfg)

    print(f"Starting {args.mode} training...")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
