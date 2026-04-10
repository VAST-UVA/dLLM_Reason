"""RL-based DAG construction policy.

Learns a policy network that constructs DAGs edge-by-edge,
conditioned on the task. This enables task-conditional DAG
construction, allowing the search to generalize across tasks.
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


class DAGPolicyNetwork(nn.Module):
    """Policy network that outputs edge probabilities for DAG construction.

    Input: current partial DAG state + task embedding
    Output: probability of adding each possible edge

    Architecture: small transformer that attends to current DAG adjacency
    and produces edge logits.
    """

    def __init__(self, max_seq_len: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Embed each position with its current in/out degree
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.degree_proj = nn.Linear(2, hidden_dim)  # in_degree, out_degree

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Output: edge logits for each pair (i, j)
        self.edge_head = nn.Bilinear(hidden_dim, hidden_dim, 1)

        # Stop action logit
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        adjacency: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            adjacency: (batch, max_seq_len, max_seq_len) current DAG adjacency
            seq_len: actual sequence length

        Returns:
            edge_logits: (batch, seq_len, seq_len) logits for adding each edge
            stop_logit: (batch, 1) logit for stopping construction
        """
        B = adjacency.shape[0]
        device = adjacency.device

        # Position embeddings
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)

        # Degree features
        adj = adjacency[:, :seq_len, :seq_len].float()
        in_degree = adj.sum(dim=1)   # (B, seq_len)
        out_degree = adj.sum(dim=2)  # (B, seq_len)
        degree_feat = torch.stack([in_degree, out_degree], dim=-1)  # (B, seq_len, 2)
        degree_emb = self.degree_proj(degree_feat)

        h = pos_emb + degree_emb  # (B, seq_len, hidden_dim)

        for layer in self.layers:
            h = layer(h)

        # Edge logits: bilinear(h[i], h[j]) for all pairs
        hi = h.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (B, L, L, D)
        hj = h.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (B, L, L, D)
        edge_logits = self.edge_head(
            hi.reshape(-1, h.shape[-1]),
            hj.reshape(-1, h.shape[-1]),
        ).reshape(B, seq_len, seq_len)

        # Mask self-loops and existing edges
        mask = torch.eye(seq_len, device=device).bool().unsqueeze(0) | adjacency[:, :seq_len, :seq_len].bool()
        edge_logits = edge_logits.masked_fill(mask, -float("inf"))

        # Stop logit
        stop_logit = self.stop_head(h.mean(dim=1))  # (B, 1)

        return edge_logits, stop_logit


@SEARCH_REGISTRY.register("rl_policy")
class RLPolicySearch(DAGSearcher):
    """RL-based DAG search using REINFORCE.

    Trains a policy network to construct DAGs edge-by-edge.
    The reward is the downstream task accuracy when using the
    constructed DAG for unmasking.
    """

    def __init__(
        self,
        max_seq_len: int = 512,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        max_edges_per_dag: int = 50,
        baseline_ema: float = 0.9,
    ):
        self.max_seq_len = max_seq_len
        self.policy = DAGPolicyNetwork(max_seq_len, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.max_edges_per_dag = max_edges_per_dag
        self.baseline_ema = baseline_ema
        self.baseline = 0.0

    def search(
        self,
        model,
        eval_fn,
        seq_len: int,
        budget: int = 100,
        **kwargs,
    ) -> SearchResult:
        device = model.device
        self.policy = self.policy.to(device)

        best_dag = TokenDAG.no_edges(seq_len, device=device)
        best_fitness = eval_fn(model, best_dag)
        history = [{"fitness": best_fitness, "step": 0}]

        for step in range(budget):
            # Construct DAG using policy
            dag, log_probs = self._construct_dag(seq_len, device)

            # Evaluate
            fitness = eval_fn(model, dag)

            # Update baseline
            self.baseline = self.baseline_ema * self.baseline + (1 - self.baseline_ema) * fitness

            # REINFORCE update
            advantage = fitness - self.baseline
            policy_loss = -advantage * sum(log_probs)

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            if fitness > best_fitness:
                best_dag = dag
                best_fitness = fitness
                logger.info(f"Step {step}: new best {best_fitness:.4f}")

            history.append({"fitness": best_fitness, "step": step + 1, "reward": fitness})

        return SearchResult(
            best_dag=best_dag,
            best_fitness=best_fitness,
            history=history,
            metadata={"method": "rl_policy"},
        )

    def _construct_dag(
        self,
        seq_len: int,
        device: torch.device,
    ) -> tuple[TokenDAG, list[torch.Tensor]]:
        """Construct a DAG by sequentially adding edges using the policy."""
        adjacency = torch.zeros(1, self.max_seq_len, self.max_seq_len, device=device)
        log_probs = []

        for _ in range(self.max_edges_per_dag):
            edge_logits, stop_logit = self.policy(adjacency, seq_len)

            # Decide: add edge or stop
            all_logits = torch.cat([
                edge_logits.view(1, -1),
                stop_logit,
            ], dim=-1)
            all_probs = F.softmax(all_logits, dim=-1)
            action = torch.multinomial(all_probs, 1).item()

            if action == seq_len * seq_len:
                # Stop action
                log_probs.append(torch.log(all_probs[0, action] + 1e-8))
                break

            # Decode edge
            src = action // seq_len
            dst = action % seq_len

            # Check acyclicity
            test_adj = adjacency.clone()
            test_adj[0, src, dst] = 1
            test_dag = TokenDAG(test_adj[0, :seq_len, :seq_len])
            if test_dag.is_valid():
                adjacency[0, src, dst] = 1
                log_probs.append(torch.log(all_probs[0, action] + 1e-8))
            else:
                # Invalid edge — still record for gradient but don't add
                log_probs.append(torch.log(all_probs[0, action] + 1e-8) * 0.1)

        dag = TokenDAG(adjacency[0, :seq_len, :seq_len])
        return dag, log_probs
