"""Learn from stored episodes: SFT, GRPO, DiFFPO, or UnmaskRL.

Four learning modes
-------------------
sft       — Supervised fine-tuning on *correct* episodes.
            (prompt, correct_output) pairs via standard cross-entropy loss.

grpo      — Group Relative Policy Optimization (DiffuGRPO).
            Reward = +1 for correct, -1 for wrong. Groups are built by
            clustering all episodes with the same prompt.

diffppo   — Diffusion Fast and Furious Policy Optimization.
            PPO-style RL with importance-ratio clipping + joint sampler
            training (adaptive step budget per prompt).
            Reference: Zhao et al. 2024 https://arxiv.org/abs/2510.02212

unmask_rl — RL-trained unmasking policy (process-level RL).
            Trains a lightweight single-layer transformer policy that
            decides which tokens to unmask at each diffusion step.
            The LM weights are kept **frozen**; only the policy net trains.
            Reference: Jazbec et al. 2025 https://arxiv.org/abs/2512.09106

All modes support DAG-aware training when episodes contain a DAG.

Usage
-----
# SFT on all correct math episodes
python scripts/learn_from_episodes.py \\
    --db_path episodes/gsm8k.db \\
    --model_id checkpoints/llada-instruct \\
    --mode sft \\
    --task_type math \\
    --output_dir checkpoints/llada-sft-math

# GRPO with DAG-aware masking
python scripts/learn_from_episodes.py \\
    --db_path episodes/gsm8k.db \\
    --model_id checkpoints/llada-instruct \\
    --mode grpo \\
    --dag_aware \\
    --output_dir checkpoints/llada-grpo-math

# DiFFPO with joint sampler training
python scripts/learn_from_episodes.py \\
    --db_path episodes/gsm8k.db \\
    --model_id checkpoints/llada-instruct \\
    --mode diffppo \\
    --ppo_clip_eps 0.2 \\
    --train_sampler \\
    --min_steps 8 --max_steps 128 \\
    --output_dir checkpoints/llada-diffppo-math

# UnmaskRL: train unmasking policy, keep LM frozen
python scripts/learn_from_episodes.py \\
    --db_path episodes/gsm8k.db \\
    --model_id checkpoints/llada-instruct \\
    --mode unmask_rl \\
    --unmask_group_size 4 \\
    --unmask_num_steps 32 \\
    --output_dir checkpoints/llada-unmask-policy

# Dry-run: print dataset stats then exit
python scripts/learn_from_episodes.py \\
    --db_path episodes/gsm8k.db \\
    --model_id none \\
    --mode stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dllm_reason.library.episode import DAGEpisode, EpisodeStore
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


# ── Dataset wrappers ──────────────────────────────────────────────────────────

class SFTEpisodeDataset(Dataset):
    """Tokenises (prompt + correct_output) pairs for SFT."""

    def __init__(
        self,
        episodes: list[DAGEpisode],
        tokenizer,
        max_length: int = 512,
    ):
        self.episodes = [e for e in episodes if e.correct is True and e.output]
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not self.episodes:
            raise ValueError("No correct episodes found for SFT.")
        logger.info(f"SFT dataset: {len(self.episodes)} correct episodes")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep = self.episodes[idx]

        # Encode full sequence: chat prompt + output
        chat = [
            {"role": "user",      "content": ep.prompt},
            {"role": "assistant", "content": ep.output},
        ]
        enc = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False,
            return_tensors="pt", max_length=self.max_length, truncation=True,
        )  # (1, L)
        input_ids = enc[0]

        # Build prompt mask: True for all tokens in the user-turn prefix
        prompt_chat = [{"role": "user", "content": ep.prompt}]
        prompt_enc = self.tokenizer.apply_chat_template(
            prompt_chat, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", max_length=self.max_length, truncation=True,
        )
        prompt_len = prompt_enc.shape[1]

        prompt_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        prompt_mask[:prompt_len] = True

        return {"input_ids": input_ids, "prompt_mask": prompt_mask}


class GRPOEpisodeDataset(Dataset):
    """Groups episodes by prompt for GRPO training.

    Each item is a dict with:
        prompt_ids   (prompt_len,) long
        episodes     list[DAGEpisode] — all episodes for this prompt
        rewards      (len(episodes),) float  +1 / -1
    """

    def __init__(
        self,
        episodes: list[DAGEpisode],
        tokenizer,
        max_prompt_length: int = 256,
        min_group_size: int = 2,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

        # Group by prompt
        from collections import defaultdict
        groups: dict[str, list[DAGEpisode]] = defaultdict(list)
        for ep in episodes:
            if ep.is_evaluated:
                groups[ep.prompt].append(ep)

        self.groups = [
            eps for eps in groups.values()
            if len(eps) >= min_group_size
        ]
        if not self.groups:
            raise ValueError(
                f"No groups with >= {min_group_size} evaluated episodes found."
            )
        logger.info(
            f"GRPO dataset: {len(self.groups)} prompt groups, "
            f"{sum(len(g) for g in self.groups)} total episodes"
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict:
        group = self.groups[idx]
        prompt = group[0].prompt

        chat = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", max_length=self.max_prompt_length, truncation=True,
        )[0]  # (prompt_len,)

        rewards = torch.tensor(
            [ep.reward for ep in group], dtype=torch.float32
        )
        return {
            "prompt_ids": prompt_ids,
            "episodes":   group,
            "rewards":    rewards,
        }


# ── SFT training loop ─────────────────────────────────────────────────────────

def run_sft(
    model,
    tokenizer,
    episodes: list[DAGEpisode],
    args: argparse.Namespace,
    dag_aware: bool = False,
) -> None:
    """Standard supervised fine-tuning on correct episodes."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torch.nn.functional as F

    dataset = SFTEpisodeDataset(episodes, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_sft_collate,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    device = model.device
    model.train()

    # Optionally inject DAG-biased noise using the best-known DAG
    # (average adjacency over all correct episodes that have a DAG)
    if dag_aware:
        dag = _build_aggregate_dag(episodes, device)
        if dag is not None:
            logger.info(f"DAG-aware training enabled: {dag}")
            # monkey-patch noise_input
            from dllm_reason.training.dag_aware_train import DAGAwareTrainer
            _patch_dag_noise(model, dag, bias_strength=0.5)

    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids  = batch["input_ids"].to(device)   # (B, L)
            prompt_mask= batch["prompt_mask"].to(device)  # (B, L) bool
            B, L = input_ids.shape

            # Sample uniform noise timestep
            t = torch.rand(B, device=device)
            x_t = model.noise_input(input_ids, t)         # masked version

            output = model.forward(x_t, t)
            logits = output.logits  # (B, L, V)

            # CE loss only on generation positions (not prompt)
            gen_mask = ~prompt_mask  # (B, L)
            logits_flat  = logits[gen_mask]        # (N, V)
            targets_flat = input_ids[gen_mask]     # (N,)
            loss = F.cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs}  "
                    f"step {global_step}  loss={loss.item():.4f}"
                )

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

    _save_model(model, tokenizer, args.output_dir)


def _sft_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a list of SFT items to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id = 0
    input_ids   = torch.stack([
        torch.nn.functional.pad(b["input_ids"],   (0, max_len - b["input_ids"].shape[0]),   value=pad_id)
        for b in batch
    ])
    prompt_mask = torch.stack([
        torch.nn.functional.pad(b["prompt_mask"], (0, max_len - b["prompt_mask"].shape[0]), value=False)
        for b in batch
    ])
    return {"input_ids": input_ids, "prompt_mask": prompt_mask}


# ── GRPO training loop ────────────────────────────────────────────────────────

def run_grpo(
    model,
    ref_model,
    tokenizer,
    episodes: list[DAGEpisode],
    args: argparse.Namespace,
) -> None:
    """GRPO fine-tuning using stored episode rewards."""
    import torch.nn.functional as F
    from torch.optim import AdamW

    dataset = GRPOEpisodeDataset(
        episodes, tokenizer, max_prompt_length=args.max_length
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    device = model.device
    model.train()

    kl_coeff   = args.kl_coeff
    clip_ratio = args.clip_ratio
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            prompt_ids: torch.Tensor = batch["prompt_ids"][0].to(device)  # (prompt_len,)
            episodes_in_group: list[DAGEpisode] = batch["episodes"][0]    # list
            rewards: torch.Tensor = batch["rewards"][0].to(device)        # (G,)

            G = len(episodes_in_group)
            if G < 2:
                continue

            # Group-relative advantages
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)   # (G,)

            all_policy_loss = []
            all_kl = []

            for g_idx, ep in enumerate(episodes_in_group):
                # Re-encode full sequence (prompt + stored output)
                chat_full = [
                    {"role": "user",      "content": ep.prompt},
                    {"role": "assistant", "content": ep.output},
                ]
                full_ids = tokenizer.apply_chat_template(
                    chat_full, tokenize=True, add_generation_prompt=False,
                    return_tensors="pt", max_length=args.max_length, truncation=True,
                ).to(device)  # (1, L)

                B, L = full_ids.shape
                prompt_len = prompt_ids.shape[0]
                prompt_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
                prompt_mask[:, :prompt_len] = True

                # Compute log-prob under policy and reference
                t = torch.full((B,), 0.01, device=device)
                x_t = model.noise_input(full_ids, t)

                with torch.no_grad():
                    ref_out = ref_model.forward(x_t, t)
                    ref_logp = F.log_softmax(ref_out.logits, dim=-1)

                pol_out  = model.forward(x_t, t)
                pol_logp = F.log_softmax(pol_out.logits, dim=-1)

                gen_mask = ~prompt_mask  # (1, L)
                gen_ids  = full_ids[gen_mask]    # (N,)

                pol_token_lp  = pol_logp [gen_mask].gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
                ref_token_lp  = ref_logp [gen_mask].gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)

                seq_pol_lp = pol_token_lp.sum()
                seq_ref_lp = ref_token_lp.sum()

                kl = seq_pol_lp - seq_ref_lp
                policy_loss = -adv[g_idx] * seq_pol_lp

                all_policy_loss.append(policy_loss)
                all_kl.append(kl)

            if not all_policy_loss:
                continue

            loss = torch.stack(all_policy_loss).mean() + kl_coeff * torch.stack(all_kl).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                mean_r = rewards.mean().item()
                logger.info(
                    f"Epoch {epoch+1}  step {global_step}  "
                    f"loss={loss.item():.4f}  mean_reward={mean_r:.3f}"
                )

        avg = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f}")

    _save_model(model, tokenizer, args.output_dir)


# ── DiFFPO training loop ──────────────────────────────────────────────────────

def run_diffppo(
    model,
    ref_model,
    tokenizer,
    episodes: list[DAGEpisode],
    args: argparse.Namespace,
) -> None:
    """DiFFPO fine-tuning: PPO with importance-ratio clipping + joint sampler.

    Reference: Zhao et al. 2024  https://arxiv.org/abs/2510.02212
    """
    from torch.utils.data import DataLoader
    from dllm_reason.training.rl_train import DiFFPO, DiFFPOConfig
    from dllm_reason.scheduler.confidence_scheduler import ConfidenceScheduler

    dataset = GRPOEpisodeDataset(episodes, tokenizer, max_prompt_length=args.max_length)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Build a reward function that looks up pre-stored episode correctness
    # (We use the stored reward rather than re-running inference for speed)
    def reward_fn(seq_ids: torch.Tensor, batch: dict) -> float:
        # batch["episodes"] contains the episode group; return mean stored reward
        group: list[DAGEpisode] = batch["episodes"][0]
        return sum(ep.reward for ep in group) / max(len(group), 1)

    cfg = DiFFPOConfig(
        lr              = args.lr,
        num_iterations  = args.epochs * len(loader),
        group_size      = 1,          # episodes already stored; single pass per group
        ppo_clip_eps    = args.ppo_clip_eps,
        kl_coeff        = args.kl_coeff,
        log_every       = args.log_every,
        train_sampler   = args.train_sampler,
        step_budget_lambda = args.step_budget_lambda,
        min_steps       = args.min_steps,
        max_steps       = args.max_steps,
    )

    scheduler = ConfidenceScheduler()
    trainer = DiFFPO(
        model       = model,
        ref_model   = ref_model,
        scheduler   = scheduler,
        reward_fn   = reward_fn,
        train_loader= loader,
        config      = cfg,
    )

    print(
        f"\nRunning DiFFPO for {cfg.num_iterations} iterations "
        f"(train_sampler={cfg.train_sampler}) ..."
    )
    trainer.train()
    _save_model(model, tokenizer, args.output_dir)


# ── UnmaskRL training loop ────────────────────────────────────────────────────

def run_unmask_rl(
    model,
    tokenizer,
    episodes: list[DAGEpisode],
    args: argparse.Namespace,
) -> None:
    """Train an unmasking policy via process-level REINFORCE.

    The LM weights are frozen; only the lightweight transformer policy is
    updated.  Best for improving inference quality without modifying the LM.

    Reference: Jazbec et al. 2025  https://arxiv.org/abs/2512.09106
    """
    from dllm_reason.training.rl_train import (
        UnmaskingPolicyRL,
        UnmaskingPolicyConfig,
    )

    dataset = GRPOEpisodeDataset(episodes, tokenizer, max_prompt_length=args.max_length)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    def reward_fn(seq_ids: torch.Tensor, batch: dict) -> float:
        group: list[DAGEpisode] = batch["episodes"][0]
        return sum(ep.reward for ep in group) / max(len(group), 1)

    cfg = UnmaskingPolicyConfig(
        lr             = args.lr,
        num_iterations = args.epochs * len(loader),
        group_size     = args.unmask_group_size,
        num_steps      = args.unmask_num_steps,
        max_grad_norm  = 1.0,
        log_every      = args.log_every,
        policy_d_model = args.unmask_d_model,
        policy_n_heads = args.unmask_n_heads,
    )

    trainer = UnmaskingPolicyRL(
        model        = model,
        reward_fn    = reward_fn,
        train_loader = loader,
        config       = cfg,
    )

    print(
        f"\nRunning UnmaskRL for {cfg.num_iterations} iterations "
        f"(group_size={cfg.group_size}, steps={cfg.num_steps}, "
        f"d_model={cfg.policy_d_model}) ..."
    )
    trainer.train()

    # Save policy weights alongside the (unchanged) LM
    import os
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    policy_path = os.path.join(out, "unmask_policy.pt")
    trainer.save_policy(policy_path)
    # Also save the original LM so the output dir is self-contained
    _save_model(model, tokenizer, out)


# ── DAG helpers for training ──────────────────────────────────────────────────

def _build_aggregate_dag(
    episodes: list[DAGEpisode],
    device: str | torch.device,
    correct_only: bool = True,
):
    """Build a 'consensus' TokenDAG by averaging adjacency matrices of episodes.

    Returns the DAG with the most common edges (majority vote), or None if
    no episodes carry DAG data.
    """
    from dllm_reason.graph.dag import TokenDAG

    candidates = [
        e for e in episodes
        if e.dag_adjacency is not None
        and e.dag_seq_len > 0
        and (not correct_only or e.correct is True)
    ]
    if not candidates:
        return None

    seq_len = candidates[0].dag_seq_len
    acc = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    used = 0
    for ep in candidates:
        if ep.dag_seq_len != seq_len:
            continue
        flat = [cell for row in ep.dag_adjacency for cell in row]
        adj = torch.tensor(flat, dtype=torch.float32).reshape(seq_len, seq_len)
        acc += adj
        used += 1

    if used == 0:
        return None

    # Majority vote
    majority = (acc / used) >= 0.5
    return TokenDAG(majority.bool().to(device))


def _patch_dag_noise(model, dag, bias_strength: float = 0.5) -> None:
    """Monkey-patch model.noise_input to use DAG-biased masking."""
    levels = dag.topological_levels()
    level_bias = torch.zeros(dag.seq_len)
    for level_idx, positions in enumerate(levels):
        bias_val = level_idx / max(len(levels) - 1, 1)
        for pos in positions:
            level_bias[pos] = bias_val

    original_noise = model.noise_input

    def biased_noise(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x_0.shape
        dev = x_0.device
        lb = level_bias[:L].to(dev)
        base = t[:, None].expand(B, L)
        biased = (1 - bias_strength) * base + bias_strength * (base * (0.5 + lb))
        biased = biased.clamp(0, 1)
        mask = torch.rand(B, L, device=dev) < biased
        return torch.where(mask, model.mask_token_id, x_0)

    model.noise_input = biased_noise
    model._original_noise_input = original_noise  # stash for restore


def _save_model(model, tokenizer, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nModel saved to {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a dLLM using stored DAG episodes."
    )

    # Data
    parser.add_argument("--db_path", default="episodes/episodes.db",
                        help="EpisodeStore SQLite path")
    parser.add_argument("--task_type", default=None,
                        help="Filter episodes by task_type (default: all)")
    parser.add_argument("--strategy", default=None,
                        help="Filter episodes by strategy_name (default: all)")
    parser.add_argument("--min_score", type=float, default=None,
                        help="Minimum episode score to include")

    # Model
    parser.add_argument("--model_id", required=True,
                        help="HuggingFace model ID or local path  ('none' for stats-only)")
    parser.add_argument("--model_type", default="llada",
                        choices=["llada", "mdlm", "sedd", "d3pm"])
    parser.add_argument("--torch_dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])

    # Training mode
    parser.add_argument("--mode", default="sft",
                        choices=["sft", "grpo", "diffppo", "unmask_rl", "stats"],
                        help=(
                            "sft: supervised fine-tuning, "
                            "grpo: DiffuGRPO, "
                            "diffppo: DiFFPO (PPO + joint sampler), "
                            "unmask_rl: learned unmasking policy (LM frozen), "
                            "stats: info only"
                        ))
    parser.add_argument("--dag_aware", action="store_true",
                        help="Use DAG-biased masking during training")

    # Hyperparameters (shared)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--max_length", type=int,   default=512)
    parser.add_argument("--log_every",  type=int,   default=20)
    parser.add_argument("--kl_coeff",   type=float, default=0.01,
                        help="KL penalty (GRPO / DiFFPO)")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip ratio (GRPO)")

    # DiFFPO-specific
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2,
                        help="ε for DiFFPO importance-ratio clipping")
    parser.add_argument("--train_sampler", action="store_true",
                        help="DiFFPO: jointly train the step-budget controller")
    parser.add_argument("--min_steps", type=int, default=8,
                        help="DiFFPO: minimum denoising steps for controller")
    parser.add_argument("--max_steps", type=int, default=128,
                        help="DiFFPO: maximum denoising steps for controller")
    parser.add_argument("--step_budget_lambda", type=float, default=0.1,
                        help="DiFFPO: weight of step-efficiency loss")

    # UnmaskRL-specific
    parser.add_argument("--unmask_group_size", type=int, default=4,
                        help="UnmaskRL: REINFORCE rollouts per prompt (group size)")
    parser.add_argument("--unmask_num_steps", type=int, default=32,
                        help="UnmaskRL: diffusion steps per rollout")
    parser.add_argument("--unmask_d_model", type=int, default=64,
                        help="UnmaskRL: policy transformer d_model")
    parser.add_argument("--unmask_n_heads", type=int, default=4,
                        help="UnmaskRL: policy transformer attention heads")

    # Output
    parser.add_argument("--output_dir", default="checkpoints/finetuned",
                        help="Where to save the fine-tuned model")

    return parser.parse_args()


def load_model(args):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.torch_dtype]

    print(f"Loading {args.model_type} from {args.model_id!r} ...")
    if args.model_type == "llada":
        from dllm_reason.models.llada import LLaDAWrapper
        model = LLaDAWrapper(
            model_id=args.model_id, torch_dtype=dtype, device_map="auto",
        )
    elif args.model_type == "mdlm":
        from dllm_reason.models.mdlm import MDLMWrapper
        model = MDLMWrapper(
            model_id=args.model_id, torch_dtype=dtype, device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported model_type={args.model_type!r}")

    tokenizer = model.tokenizer
    print(f"Model ready on {model.device}")
    return model, tokenizer


def main():
    args = parse_args()

    store = EpisodeStore(args.db_path)
    store.print_stats()

    if args.mode == "stats":
        return

    # Load episodes
    episodes = store.query(
        task_type=args.task_type,
        strategy_name=args.strategy,
        min_score=args.min_score,
        limit=100_000,
    )
    print(f"Loaded {len(episodes)} episodes for training.")
    if not episodes:
        print("No episodes match the filter. Exiting.")
        return

    if args.model_id.lower() == "none":
        print("model_id='none': skipping model load.")
        return

    model, tokenizer = load_model(args)

    if args.mode == "sft":
        print(f"\nRunning SFT for {args.epochs} epochs ...")
        run_sft(
            model, tokenizer, episodes, args,
            dag_aware=args.dag_aware,
        )

    elif args.mode == "grpo":
        print(f"\nRunning GRPO for {args.epochs} epochs ...")
        # Load a frozen reference copy of the same model
        print("Loading frozen reference model ...")
        ref_model, _ = load_model(args)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()

        run_grpo(model, ref_model, tokenizer, episodes, args)

    elif args.mode == "diffppo":
        print(f"\nRunning DiFFPO (Zhao et al. 2024) ...")
        print("Loading frozen reference model ...")
        ref_model, _ = load_model(args)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()

        run_diffppo(model, ref_model, tokenizer, episodes, args)

    elif args.mode == "unmask_rl":
        print(f"\nRunning UnmaskRL (Jazbec et al. 2025) — LM frozen, training policy ...")
        run_unmask_rl(model, tokenizer, episodes, args)


if __name__ == "__main__":
    main()
