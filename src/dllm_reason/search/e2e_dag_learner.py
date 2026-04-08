"""End-to-End DAG Learning — jointly learn DAG structure and evaluate task performance.

Unlike the separate search methods (greedy, evolutionary, etc.) that treat
the fitness function as a black box, the E2E learner propagates gradients
from the task loss through a differentiable DAG parameterization.

Architecture:
    DAGParameterization (soft edges)
        → DAGScheduler (differentiable relaxation)
        → DiffusionSampler (model forward pass)
        → Task loss (e.g. cross-entropy on answer tokens)
        ← Backprop through soft edge probabilities

The key challenge: the discrete unmasking order is non-differentiable.
We address this via:
1. Soft scheduling: use edge probabilities as soft masks on logits
2. Gumbel-Sigmoid for edge sampling with straight-through gradients
3. NOTEARS acyclicity regularization

This enables learning task-specific DAG structures end-to-end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.search.base import DAGSearcher, SearchResult
from dllm_reason.search.differentiable import DifferentiableDAG
from dllm_reason.utils.registry import SEARCH_REGISTRY
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


class DAGReadinessModule(nn.Module):
    """Differentiable readiness computation for soft DAGs.

    Given soft edge probabilities P and an unmasked set U, compute a
    differentiable approximation of the readiness score for each position:

        readiness(j) = prod_{i: P(i,j) > threshold} sigmoid(alpha * (u_i - 0.5))

    where u_i indicates whether position i is unmasked (1.0) or masked (0.0),
    and alpha controls the sharpness of the sigmoid approximation.
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        edge_probs: torch.Tensor,   # (L, L) soft adjacency
        is_unmasked: torch.Tensor,   # (B, L) float in [0, 1]
    ) -> torch.Tensor:
        """Compute differentiable readiness scores.

        Returns: (B, L) readiness in [0, 1]
        """
        # edge_probs[i, j] = probability of edge i->j
        # For each j, readiness = product of (1 - P(i,j) + P(i,j) * u_i)
        # = "for each parent, either the edge doesn't exist or parent is unmasked"

        # (L, L) -> broadcast with (B, L)
        # satisfied[b, i, j] = 1 - P(i,j) + P(i,j) * u_i[b]
        # = probability that constraint (i->j) is satisfied
        u = is_unmasked.unsqueeze(-1)  # (B, L, 1)
        P = edge_probs.unsqueeze(0)     # (1, L, L)

        satisfied = 1.0 - P + P * u  # (B, L, L)

        # readiness(j) = prod_i satisfied(i, j)
        # Use log-sum for numerical stability
        log_satisfied = torch.log(satisfied.clamp(min=1e-8))  # (B, L, L)
        log_readiness = log_satisfied.sum(dim=1)  # (B, L) — sum over parent dim
        readiness = torch.exp(log_readiness)  # (B, L)

        return readiness


@dataclass
class E2EConfig:
    """Configuration for E2E DAG learning."""
    lr_dag: float = 3e-3
    lr_schedule: str = "cosine"       # "cosine" or "constant"
    warmup_steps: int = 10
    acyclicity_weight: float = 1.0
    sparsity_weight: float = 0.01     # L1 on edge probabilities
    entropy_weight: float = 0.01      # encourage decisive edges (0 or 1)
    tau_start: float = 1.0            # Gumbel temperature start
    tau_end: float = 0.1              # Gumbel temperature end (annealed)
    tau_anneal: str = "linear"        # "linear" or "cosine"
    rho_init: float = 1.0
    rho_max: float = 1e8
    rho_multiply: float = 10.0
    h_tol: float = 1e-6
    checkpoint_every: int = 50
    max_edges_ratio: float = 0.3      # max edges as fraction of L^2


@SEARCH_REGISTRY.register("e2e")
class E2EDAGLearner(DAGSearcher):
    """End-to-end DAG structure learning with task-driven optimization.

    Jointly optimizes DAG structure (edge probabilities) to maximize
    task performance via differentiable relaxation.

    Compared to DifferentiableDAGSearch:
    - Uses a differentiable readiness module (not just NOTEARS)
    - Supports curriculum-style temperature annealing
    - Has sparsity and entropy regularization
    - Supports warm-starting from an existing DAG
    """

    def __init__(
        self,
        config: E2EConfig | None = None,
        init_dag: TokenDAG | None = None,
    ):
        self.config = config or E2EConfig()
        self.init_dag = init_dag

    def search(
        self,
        model,
        eval_fn: Callable,
        seq_len: int,
        budget: int = 200,
        task_loss_fn: Callable | None = None,
        save_dir: str | None = None,
        **kwargs,
    ) -> SearchResult:
        """Run end-to-end DAG learning.

        Args:
            model: DiffusionLM
            eval_fn: callable(model, dag) -> float (non-differentiable fitness)
            seq_len: generation sequence length
            budget: number of optimization steps
            task_loss_fn: optional differentiable loss for gradient-based
                optimization. If None, falls back to eval_fn + NOTEARS only.
            save_dir: optional directory to save checkpoints
        """
        cfg = self.config
        device = model.device

        # Initialize differentiable DAG
        diff_dag = DifferentiableDAG(seq_len, tau=cfg.tau_start).to(device)

        # Warm-start from existing DAG
        if self.init_dag is not None:
            with torch.no_grad():
                adj = self.init_dag.adjacency.float().to(device)
                # Initialize theta so sigmoid(theta/tau) ≈ adj
                diff_dag.theta.data = torch.logit(adj.clamp(0.01, 0.99)) * cfg.tau_start

        readiness_module = DAGReadinessModule().to(device)

        optimizer = torch.optim.AdamW(
            diff_dag.parameters(), lr=cfg.lr_dag, weight_decay=0.0
        )

        # Augmented Lagrangian state
        lmbda = 0.0
        rho = cfg.rho_init
        prev_h = float("inf")

        # Track best
        best_dag = diff_dag.to_dag()
        best_fitness = eval_fn(model, best_dag)
        history = [{"fitness": best_fitness, "step": 0, "h": 0, "tau": cfg.tau_start}]

        logger.info(f"E2E DAG Learning: seq_len={seq_len}, budget={budget}")
        logger.info(f"Initial fitness: {best_fitness:.4f}")

        for step in range(1, budget + 1):
            diff_dag.train()

            # Temperature annealing
            progress = step / budget
            if cfg.tau_anneal == "cosine":
                import math
                tau = cfg.tau_end + 0.5 * (cfg.tau_start - cfg.tau_end) * (1 + math.cos(math.pi * progress))
            else:
                tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * progress
            diff_dag.tau = tau

            # Get soft edge probabilities
            edge_probs = diff_dag.get_edge_probs(hard=False)

            # Acyclicity penalty
            h = diff_dag.acyclicity_penalty()

            # Sparsity: L1 on edge probs
            sparsity = edge_probs.sum()

            # Edge entropy: encourage binary edges (0 or 1)
            eps = 1e-8
            edge_entropy = -(edge_probs * (edge_probs + eps).log() +
                             (1 - edge_probs) * (1 - edge_probs + eps).log()).sum()

            # Edge count constraint
            max_edges = int(cfg.max_edges_ratio * seq_len * seq_len)
            edge_count_penalty = F.relu(edge_probs.sum() - max_edges)

            # Composite loss
            loss = (
                cfg.acyclicity_weight * (lmbda * h + (rho / 2) * h * h) +
                cfg.sparsity_weight * sparsity +
                cfg.entropy_weight * edge_entropy +
                0.01 * edge_count_penalty
            )

            # If task_loss_fn provided, add differentiable task loss
            if task_loss_fn is not None:
                task_loss = task_loss_fn(model, edge_probs, readiness_module)
                loss = loss + task_loss

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(diff_dag.parameters(), 1.0)
            optimizer.step()

            # Update Lagrangian multipliers
            h_val = h.item()
            if h_val > 0.25 * prev_h:
                rho = min(rho * cfg.rho_multiply, cfg.rho_max)
            lmbda = lmbda + rho * h_val
            prev_h = h_val

            # Periodic evaluation with hard DAG
            if step % 5 == 0 or step == budget:
                current_dag = diff_dag.to_dag()
                fitness = eval_fn(model, current_dag)

                if fitness > best_fitness:
                    best_dag = current_dag
                    best_fitness = fitness
                    logger.info(
                        f"Step {step}/{budget}: new best {best_fitness:.4f} "
                        f"(edges={current_dag.num_edges()}, h={h_val:.6f}, tau={tau:.3f})"
                    )

                history.append({
                    "fitness": best_fitness,
                    "current_fitness": fitness,
                    "step": step,
                    "h": h_val,
                    "tau": tau,
                    "lambda": lmbda,
                    "rho": rho,
                    "num_edges": current_dag.num_edges(),
                    "sparsity": sparsity.item(),
                })

            # Checkpoint
            if save_dir and step % cfg.checkpoint_every == 0:
                self._save_checkpoint(diff_dag, history, save_dir, step)

        # Final save
        if save_dir:
            self._save_checkpoint(diff_dag, history, save_dir, budget)

        return SearchResult(
            best_dag=best_dag,
            best_fitness=best_fitness,
            history=history,
            metadata={
                "method": "e2e",
                "final_h": prev_h,
                "final_tau": tau,
                "total_steps": budget,
            },
        )

    @staticmethod
    def _save_checkpoint(diff_dag, history, save_dir, step):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(diff_dag.state_dict(), save_path / f"dag_params_step{step}.pt")
        with open(save_path / f"history_step{step}.json", "w") as f:
            json.dump(history, f, indent=2)
