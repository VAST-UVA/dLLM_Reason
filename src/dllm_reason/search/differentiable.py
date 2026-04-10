"""Differentiable DAG learning using the NOTEARS framework.

Parameterizes edge probabilities with continuous values and enforces
acyclicity via the NOTEARS constraint: tr(exp(A * A)) - d = 0.

Reference: Zheng et al., "DAGs with NO TEARS", NeurIPS 2018
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.search.base import DAGSearcher, SearchResult
from dllm_reason.utils.registry import SEARCH_REGISTRY
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


class DifferentiableDAG(nn.Module):
    """Learnable DAG parameterization with NOTEARS acyclicity constraint.

    Edge weights theta are continuous parameters. The DAG is sampled
    via Gumbel-Sigmoid: edge_prob = sigmoid((theta + gumbel_noise) / tau).
    Acyclicity is enforced via augmented Lagrangian: h(A) = tr(exp(A*A)) - d = 0.
    """

    def __init__(self, seq_len: int, tau: float = 0.5):
        super().__init__()
        self.seq_len = seq_len
        self.tau = tau

        # Edge weight parameters (continuous, unconstrained)
        self.theta = nn.Parameter(torch.zeros(seq_len, seq_len))
        # Mask out diagonal (no self-loops)
        self.register_buffer(
            "diag_mask",
            ~torch.eye(seq_len, dtype=torch.bool),
        )

    def get_edge_probs(self, hard: bool = False) -> torch.Tensor:
        """Get edge probabilities via Gumbel-Sigmoid.

        Args:
            hard: if True, use straight-through estimator for hard edges

        Returns:
            (seq_len, seq_len) edge probabilities in [0, 1]
        """
        # Mask diagonal
        masked_theta = self.theta * self.diag_mask.float()

        if self.training:
            # Gumbel-Sigmoid sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(masked_theta).clamp(1e-8)).clamp(1e-8))
            probs = torch.sigmoid((masked_theta + gumbel) / self.tau)
        else:
            probs = torch.sigmoid(masked_theta / self.tau)

        if hard:
            hard_probs = (probs > 0.5).float()
            probs = hard_probs - probs.detach() + probs  # Straight-through

        return probs

    def acyclicity_penalty(self) -> torch.Tensor:
        """NOTEARS acyclicity constraint: h(A) = tr(exp(A * A)) - d.

        h(A) = 0 iff A is a DAG.
        """
        probs = self.get_edge_probs()
        # Element-wise square
        M = probs * probs
        # Matrix exponential via Taylor series (more stable for autograd)
        d = self.seq_len
        power = torch.eye(d, device=M.device)
        result = torch.eye(d, device=M.device)
        for k in range(1, d + 1):
            power = torch.mm(power, M) / k
            result = result + power
        h = torch.trace(result) - d
        return h

    def to_dag(self) -> TokenDAG:
        """Convert current parameters to a hard TokenDAG."""
        with torch.no_grad():
            probs = self.get_edge_probs(hard=False)
            adj = (probs > 0.5).bool()
            # Ensure acyclicity by removing low-weight back edges
            dag = TokenDAG(adj)
            if not dag.is_valid():
                # Greedily remove edges by weight until acyclic
                adj = self._enforce_acyclicity(adj, probs)
                dag = TokenDAG(adj)
            return dag

    def _enforce_acyclicity(
        self, adj: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Remove minimum-weight edges to break cycles."""
        adj = adj.clone()
        edges = adj.nonzero(as_tuple=False)
        # Sort by weight ascending (remove lightest first)
        edge_weights = weights[edges[:, 0], edges[:, 1]]
        sorted_idx = edge_weights.argsort()

        for idx in sorted_idx:
            src, dst = edges[idx].tolist()
            adj[src, dst] = False
            if TokenDAG(adj).is_valid():
                return adj
            adj[src, dst] = True  # Restore if didn't help

        return adj


@SEARCH_REGISTRY.register("differentiable")
class DifferentiableDAGSearch(DAGSearcher):
    """Differentiable DAG search using NOTEARS + augmented Lagrangian.

    Jointly optimizes:
    - Task loss (via the dLLM with soft DAG scheduling)
    - Acyclicity constraint h(A) = 0

    Uses augmented Lagrangian method:
    L = task_loss + lambda * h(A) + (rho/2) * h(A)^2
    where lambda and rho are increased when h(A) doesn't decrease.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        lambda_init: float = 0.0,
        rho_init: float = 1.0,
        rho_max: float = 1e16,
        h_tol: float = 1e-8,
        rho_multiply: float = 10.0,
    ):
        self.lr = lr
        self.lambda_init = lambda_init
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.h_tol = h_tol
        self.rho_multiply = rho_multiply

    def search(
        self,
        model,
        eval_fn,
        seq_len: int,
        budget: int = 100,
        **kwargs,
    ) -> SearchResult:
        device = model.device

        diff_dag = DifferentiableDAG(seq_len).to(device)
        optimizer = torch.optim.Adam(diff_dag.parameters(), lr=self.lr)

        lmbda = self.lambda_init
        rho = self.rho_init
        prev_h = float("inf")
        baseline = 0.0          # running mean of fitness for REINFORCE

        best_dag = diff_dag.to_dag()
        best_fitness = eval_fn(model, best_dag)
        history = [{"fitness": best_fitness, "step": 0}]

        for step in range(budget):
            diff_dag.train()

            # Sample a DAG via Gumbel-Sigmoid and compute its edge probs.
            # `probs` is the sigmoid of (theta + gumbel_noise)/tau and IS
            # differentiable w.r.t. theta.
            probs = diff_dag.get_edge_probs(hard=False)
            # Hard sample (straight-through)
            hard = (probs > 0.5).float()
            current_adj = (hard - probs).detach() + probs  # (L, L), grad to theta
            current_dag = TokenDAG(current_adj.detach().bool())

            # Evaluate fitness of the sampled DAG (non-differentiable)
            fitness = eval_fn(model, current_dag)
            fitness_t = torch.as_tensor(
                float(fitness), device=device, dtype=probs.dtype,
            )

            # REINFORCE surrogate — log-prob of the sampled Bernoulli edges
            # Higher-than-baseline fitness → push probs toward the sample
            eps = 1e-8
            log_probs = (
                hard * torch.log(probs + eps)
                + (1.0 - hard) * torch.log(1.0 - probs + eps)
            )
            log_prob_sum = log_probs.sum()

            advantage = fitness_t - baseline
            policy_loss = -(advantage * log_prob_sum)

            # Acyclicity constraint (fully differentiable)
            h = diff_dag.acyclicity_penalty()
            h_loss = lmbda * h + (rho / 2) * h * h

            loss = policy_loss + h_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update baseline (EMA)
            baseline = 0.9 * baseline + 0.1 * float(fitness)

            # Update Lagrangian multipliers
            h_val = h.item()
            if h_val > 0.25 * prev_h:
                rho = min(rho * self.rho_multiply, self.rho_max)
            lmbda = lmbda + rho * h_val
            prev_h = h_val

            if fitness > best_fitness:
                best_dag = current_dag
                best_fitness = fitness
                logger.info(f"Step {step}: new best {best_fitness:.4f}, h={h_val:.6f}")

            history.append({
                "fitness": best_fitness,
                "step": step + 1,
                "h": h_val,
                "lambda": lmbda,
                "rho": rho,
            })

            if h_val < self.h_tol:
                logger.info(f"Acyclicity satisfied at step {step}")

        return SearchResult(
            best_dag=best_dag,
            best_fitness=best_fitness,
            history=history,
            metadata={"method": "differentiable", "final_h": prev_h},
        )
