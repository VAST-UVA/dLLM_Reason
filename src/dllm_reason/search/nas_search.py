"""NAS-style DAG Architecture Search.

Treats the DAG structure as a neural architecture search problem:
- Search space: DAG edge configurations (which edges exist)
- Performance estimator: lightweight proxy evaluation (few samples)
- Search controller: RNN/Transformer policy that generates DAG configurations

Inspired by ENAS (Efficient NAS) and DARTS, but adapted for DAG scheduling:
- ENAS-style: parameter sharing across DAG configurations
- DARTS-style: continuous relaxation of discrete edge choices
- Supernet: a single "supernet DAG" with all possible edges, each weighted

Key difference from standard DARTS:
- We optimize DAG topology, not layer operations
- The acyclicity constraint adds a unique challenge (NAS has no such constraint)
- Our search space is (L choose 2) binary decisions for L positions

Search Strategies:
1. SuperDAG + continuous relaxation (DARTS-like)
2. Controller-based sequential construction (ENAS-like)
3. Progressive growing: start small, gradually add edges
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.graph.span_dag import SpanDAG
from dllm_reason.search.base import DAGSearcher, SearchResult
from dllm_reason.utils.registry import SEARCH_REGISTRY
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SuperDAG: DARTS-like continuous relaxation over span-level DAGs
# ──────────────────────────────────────────────────────────────────────────────

class SuperDAG(nn.Module):
    """SuperDAG: a weighted superset of all possible DAG configurations.

    Each potential edge (i -> j) for i < j has a learnable architecture
    parameter alpha[i, j]. The soft adjacency is sigmoid(alpha).

    For efficiency, operates at span level (SpanDAG) rather than token level,
    reducing the search space from L^2 to (L/span_size)^2.
    """

    def __init__(self, num_spans: int, span_size: int):
        super().__init__()
        self.num_spans = num_spans
        self.span_size = span_size

        # Architecture parameters: only upper-triangular (i < j) to avoid
        # trivial cycles. Initialize near zero (undecided).
        self.alpha = nn.Parameter(torch.zeros(num_spans, num_spans))

        # Mask: only allow edges where i != j
        mask = ~torch.eye(num_spans, dtype=torch.bool)
        self.register_buffer("edge_mask", mask)

    def get_soft_adjacency(self, tau: float = 1.0) -> torch.Tensor:
        """Get soft adjacency matrix with temperature-scaled sigmoid."""
        return torch.sigmoid(self.alpha / tau) * self.edge_mask.float()

    def get_hard_adjacency(self, threshold: float = 0.5) -> torch.Tensor:
        """Get discrete adjacency by thresholding."""
        with torch.no_grad():
            return (torch.sigmoid(self.alpha) > threshold) & self.edge_mask

    def acyclicity_penalty(self, tau: float = 1.0) -> torch.Tensor:
        """NOTEARS penalty on soft adjacency."""
        A = self.get_soft_adjacency(tau)
        M = A * A
        d = self.num_spans
        power = torch.eye(d, device=M.device)
        result = torch.eye(d, device=M.device)
        for k in range(1, d + 1):
            power = torch.mm(power, M) / k
            result = result + power
        return torch.trace(result) - d

    def to_span_dag(self, threshold: float = 0.5) -> SpanDAG:
        """Convert to a discrete SpanDAG."""
        adj = self.get_hard_adjacency(threshold)
        sdag = SpanDAG(num_spans=self.num_spans, span_size=self.span_size)
        sdag.adjacency = adj
        if not sdag.is_valid():
            # Remove lowest-weight back edges
            soft = torch.sigmoid(self.alpha)
            edges = adj.nonzero(as_tuple=False)
            if edges.numel() > 0:
                weights = soft[edges[:, 0], edges[:, 1]]
                for idx in weights.argsort():
                    src, dst = edges[idx].tolist()
                    adj[src, dst] = False
                    sdag.adjacency = adj
                    if sdag.is_valid():
                        break
                    adj[src, dst] = True
        sdag.adjacency = adj
        return sdag

    def to_token_dag(self, threshold: float = 0.5) -> TokenDAG:
        """Convert to a TokenDAG."""
        return self.to_span_dag(threshold).to_token_dag()


# ──────────────────────────────────────────────────────────────────────────────
# DAG Controller: ENAS-like sequential edge construction
# ──────────────────────────────────────────────────────────────────────────────

class DAGController(nn.Module):
    """ENAS-style controller that generates DAG configurations sequentially.

    For each span pair (i, j), the controller decides whether to add an edge.
    The controller is a small GRU that takes the previously decided edges
    as context and outputs edge probabilities.

    The controller is trained with REINFORCE:
    reward = fitness(generated_dag)
    """

    def __init__(self, num_spans: int, hidden_dim: int = 64):
        super().__init__()
        self.num_spans = num_spans
        self.hidden_dim = hidden_dim

        # Number of potential edges (upper triangle)
        self.num_decisions = num_spans * (num_spans - 1) // 2

        # GRU controller
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=True)
        self.edge_head = nn.Linear(hidden_dim, 1)

        # Embedding for span indices
        self.span_embed = nn.Embedding(num_spans, 2)

    def forward(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate DAG configurations.

        Returns:
            edge_decisions: (batch_size, num_decisions) binary
            log_probs: (batch_size, num_decisions) log probabilities
        """
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        decisions = []
        log_probs = []

        # Enumerate all possible edges (i, j) where i < j
        edge_pairs = []
        for i in range(self.num_spans):
            for j in range(i + 1, self.num_spans):
                edge_pairs.append((i, j))

        for step, (src, dst) in enumerate(edge_pairs):
            # Input: [src_embed, dst_embed, prev_decision]
            src_emb = self.span_embed(torch.tensor([src], device=device))  # (1, 2)
            dst_emb = self.span_embed(torch.tensor([dst], device=device))  # (1, 2)

            prev_dec = torch.zeros(batch_size, 1, device=device)
            if decisions:
                prev_dec = decisions[-1].float().unsqueeze(-1)

            # Combine inputs: (batch, 1, 3)
            # Previously the second term was `src_emb_exp[:, 1:]` which threw
            # away the destination embedding entirely (bug C13). We now feed
            # one scalar from src and one scalar from dst so the GRU can
            # distinguish edges with the same source.
            src_emb_exp = src_emb.expand(batch_size, -1)  # (batch, 2)
            dst_emb_exp = dst_emb.expand(batch_size, -1)  # (batch, 2)
            inp = torch.cat(
                [src_emb_exp[:, :1], dst_emb_exp[:, :1], prev_dec], dim=-1,
            )
            inp = inp.unsqueeze(1)  # (batch, 1, 3)

            out, h = self.gru(inp, h)
            logit = self.edge_head(out.squeeze(1)).squeeze(-1)  # (batch,)

            # Sample or greedy
            prob = torch.sigmoid(logit / temperature)
            if greedy:
                decision = (prob > 0.5).long()
            else:
                decision = torch.bernoulli(prob).long()

            log_prob = torch.where(
                decision == 1,
                torch.log(prob + 1e-8),
                torch.log(1 - prob + 1e-8),
            )

            decisions.append(decision)
            log_probs.append(log_prob)

        return (
            torch.stack(decisions, dim=1),   # (batch, num_decisions)
            torch.stack(log_probs, dim=1),   # (batch, num_decisions)
        )

    def decisions_to_dag(
        self,
        decisions: torch.Tensor,
        span_size: int,
    ) -> list[TokenDAG]:
        """Convert decisions to TokenDAGs."""
        batch_size = decisions.shape[0]
        dags = []

        for b in range(batch_size):
            adj = torch.zeros(self.num_spans, self.num_spans, dtype=torch.bool)
            idx = 0
            for i in range(self.num_spans):
                for j in range(i + 1, self.num_spans):
                    if decisions[b, idx].item() == 1:
                        adj[i, j] = True
                    idx += 1

            sdag = SpanDAG(num_spans=self.num_spans, span_size=span_size)
            sdag.adjacency = adj
            if sdag.is_valid():
                dags.append(sdag.to_token_dag())
            else:
                # Fallback: no-edges DAG
                dags.append(TokenDAG.no_edges(self.num_spans * span_size))

        return dags


# ──────────────────────────────────────────────────────────────────────────────
# NAS Search: main search class combining SuperDAG + Controller
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NASConfig:
    """Configuration for NAS-style DAG search."""
    mode: str = "supernet"            # "supernet" (DARTS) or "controller" (ENAS)
    span_size: int = 16               # tokens per span (reduces search space)
    # SuperNet config
    lr_alpha: float = 3e-3            # architecture parameter learning rate
    tau_start: float = 2.0
    tau_end: float = 0.1
    acyclicity_weight: float = 1.0
    sparsity_weight: float = 0.01
    # Controller config
    controller_hidden: int = 64
    lr_controller: float = 1e-3
    controller_batch: int = 8         # samples per REINFORCE step
    baseline_ema: float = 0.95        # exponential moving average for baseline
    # Common
    proxy_samples: int = 20           # samples for lightweight proxy eval
    full_eval_every: int = 20         # full evaluation interval


@SEARCH_REGISTRY.register("nas")
class NASDAGSearch(DAGSearcher):
    """NAS-style DAG architecture search.

    Two modes:
    1. **supernet** (DARTS-like): Continuous relaxation over a SuperDAG.
       Uses gradient descent on architecture parameters with NOTEARS
       acyclicity constraint. Efficient but may get stuck in local optima.

    2. **controller** (ENAS-like): A small GRU generates DAG configurations
       sequentially. Trained with REINFORCE using proxy fitness as reward.
       More exploration but slower convergence.

    Both modes operate at span level for efficiency.
    """

    def __init__(self, config: NASConfig | None = None):
        self.config = config or NASConfig()

    def search(
        self,
        model,
        eval_fn: Callable,
        seq_len: int,
        budget: int = 200,
        **kwargs,
    ) -> SearchResult:
        cfg = self.config
        num_spans = seq_len // cfg.span_size
        if num_spans * cfg.span_size != seq_len:
            # Round up
            num_spans = (seq_len + cfg.span_size - 1) // cfg.span_size
            actual_len = num_spans * cfg.span_size
            logger.warning(
                f"seq_len={seq_len} not divisible by span_size={cfg.span_size}; "
                f"using {actual_len} (num_spans={num_spans})"
            )

        if cfg.mode == "supernet":
            return self._search_supernet(model, eval_fn, num_spans, cfg.span_size, budget)
        elif cfg.mode == "controller":
            return self._search_controller(model, eval_fn, num_spans, cfg.span_size, budget)
        else:
            raise ValueError(f"Unknown NAS mode: {cfg.mode}")

    def _search_supernet(
        self, model, eval_fn, num_spans, span_size, budget
    ) -> SearchResult:
        """DARTS-like supernet search."""
        cfg = self.config
        device = model.device

        supernet = SuperDAG(num_spans, span_size).to(device)
        optimizer = torch.optim.Adam(supernet.parameters(), lr=cfg.lr_alpha)

        lmbda = 0.0
        rho = 1.0
        prev_h = float("inf")

        best_dag = supernet.to_token_dag()
        best_fitness = eval_fn(model, best_dag)
        history = [{"fitness": best_fitness, "step": 0}]

        logger.info(f"NAS SuperNet: {num_spans} spans x {span_size} tokens, budget={budget}")

        for step in range(1, budget + 1):
            supernet.train()

            progress = step / budget
            tau = cfg.tau_end + (cfg.tau_start - cfg.tau_end) * (1 - progress)

            # Architecture loss
            h = supernet.acyclicity_penalty(tau)
            soft_adj = supernet.get_soft_adjacency(tau)
            sparsity = soft_adj.sum()

            loss = (
                cfg.acyclicity_weight * (lmbda * h + (rho / 2) * h * h) +
                cfg.sparsity_weight * sparsity
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(supernet.parameters(), 1.0)
            optimizer.step()

            # Update Lagrangian
            h_val = h.item()
            if h_val > 0.25 * prev_h:
                rho = min(rho * 10, 1e8)
            lmbda = lmbda + rho * h_val
            prev_h = h_val

            # Evaluate periodically
            if step % cfg.full_eval_every == 0 or step == budget:
                current_dag = supernet.to_token_dag()
                fitness = eval_fn(model, current_dag)

                if fitness > best_fitness:
                    best_dag = current_dag
                    best_fitness = fitness
                    logger.info(
                        f"Step {step}: new best {best_fitness:.4f} "
                        f"(edges={current_dag.num_edges()}, h={h_val:.4f})"
                    )

                history.append({
                    "fitness": best_fitness,
                    "current_fitness": fitness,
                    "step": step,
                    "h": h_val,
                    "tau": tau,
                    "num_edges": current_dag.num_edges(),
                })

        return SearchResult(
            best_dag=best_dag,
            best_fitness=best_fitness,
            history=history,
            metadata={"method": "nas_supernet", "num_spans": num_spans, "span_size": span_size},
        )

    def _search_controller(
        self, model, eval_fn, num_spans, span_size, budget
    ) -> SearchResult:
        """ENAS-like controller search."""
        cfg = self.config
        device = model.device

        controller = DAGController(num_spans, cfg.controller_hidden).to(device)
        optimizer = torch.optim.Adam(controller.parameters(), lr=cfg.lr_controller)

        baseline = 0.0  # EMA baseline for REINFORCE
        best_dag = TokenDAG.no_edges(num_spans * span_size)
        best_fitness = eval_fn(model, best_dag)
        history = [{"fitness": best_fitness, "step": 0}]

        logger.info(f"NAS Controller: {num_spans} spans x {span_size} tokens, budget={budget}")

        step = 0
        while step < budget:
            # Sample batch of DAG configurations
            decisions, log_probs = controller(
                batch_size=cfg.controller_batch,
                temperature=max(0.5, 1.0 - step / budget),
            )
            dags = controller.decisions_to_dag(decisions, span_size)

            # Evaluate each
            rewards = []
            for dag in dags:
                fitness = eval_fn(model, dag)
                rewards.append(fitness)
                step += 1

                if fitness > best_fitness:
                    best_dag = dag
                    best_fitness = fitness
                    logger.info(f"Step {step}: new best {best_fitness:.4f}")

                if step >= budget:
                    break

            # REINFORCE update
            rewards_t = torch.tensor(rewards[:len(dags)], device=device)
            advantage = rewards_t - baseline

            # Update baseline (EMA)
            baseline = cfg.baseline_ema * baseline + (1 - cfg.baseline_ema) * rewards_t.mean().item()

            # Policy gradient loss
            batch_log_probs = log_probs[:len(dags)]  # (batch, num_decisions)
            pg_loss = -(batch_log_probs.sum(dim=1) * advantage).mean()

            optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()

            history.append({
                "fitness": best_fitness,
                "step": step,
                "mean_reward": rewards_t.mean().item(),
                "baseline": baseline,
                "pg_loss": pg_loss.item(),
            })

        return SearchResult(
            best_dag=best_dag,
            best_fitness=best_fitness,
            history=history,
            metadata={"method": "nas_controller", "num_spans": num_spans, "span_size": span_size},
        )
