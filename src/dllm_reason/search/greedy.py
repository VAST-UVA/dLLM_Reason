"""Greedy edge search for DAG structure optimization.

Starts from an initial DAG (empty or template) and greedily adds/removes
edges that improve the fitness function.

Supports optional DAG Library integration for initialization and writeback.
"""

from __future__ import annotations

import random
from typing import Optional

import torch

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.search.base import DAGSearcher, SearchResult
from dllm_reason.utils.registry import SEARCH_REGISTRY
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@SEARCH_REGISTRY.register("greedy")
class GreedyEdgeSearch(DAGSearcher):
    """Greedy DAG search by adding/removing single edges.

    At each step:
    1. Generate candidate edges to add or remove
    2. Evaluate each candidate by running inference
    3. Keep the best improving modification
    4. Repeat until budget exhausted or no improvement found
    """

    def __init__(
        self,
        initial_dag: TokenDAG | None = None,
        init_templates: list[str] | None = None,
        num_candidates: int = 10,
        patience: int = 5,
        library: Optional["DAGStore"] = None,
        task_description: str = "",
    ):
        """
        Args:
            initial_dag:    Explicit starting DAG.  Takes priority over templates.
            init_templates: Names from ``TEMPLATE_NAMES``.  When provided and
                            ``initial_dag`` is None, the searcher evaluates all
                            listed templates and starts from the best one (costs
                            len(init_templates) evaluations from the budget).
                            Pass ``None`` to skip template warm-start entirely.
        """
        self.initial_dag = initial_dag
        self.init_templates = init_templates
        self.num_candidates = num_candidates
        self.patience = patience
        self.library = library
        self.task_description = task_description

    def _pick_best_template(
        self,
        model,
        eval_fn,
        seq_len: int,
    ) -> tuple[TokenDAG, float, int]:
        """Evaluate all init_templates and return the best one.

        Returns:
            (best_dag, best_fitness, evals_used)
        """
        from dllm_reason.graph.templates import build_all_templates
        templates = build_all_templates(seq_len, device=model.device,
                                        names=self.init_templates)
        best_dag, best_fit = None, -float("inf")
        for name, dag in templates.items():
            fit = eval_fn(model, dag)
            logger.info(f"Template warm-start: {name} → fitness={fit:.4f}")
            if fit > best_fit:
                best_dag, best_fit = dag, fit
        return best_dag, best_fit, len(templates)

    def search(
        self,
        model,
        eval_fn,
        seq_len: int,
        budget: int = 100,
        **kwargs,
    ) -> SearchResult:
        # --- Initialize starting point ---
        evals_done = 0
        if self.initial_dag is not None:
            # Explicit seed
            current_dag = self.initial_dag
            current_fitness = eval_fn(model, current_dag)
            evals_done += 1
        elif self.init_templates:
            # Warm-start: evaluate all templates, pick best
            current_dag, current_fitness, n = self._pick_best_template(
                model, eval_fn, seq_len
            )
            evals_done += n
            logger.info(
                f"Template warm-start complete: best fitness={current_fitness:.4f} "
                f"({current_dag.num_edges()} edges), used {n}/{budget} budget"
            )
        else:
            current_dag = TokenDAG.empty(seq_len, device=model.device)
            current_fitness = eval_fn(model, current_dag)
            evals_done += 1

        best_dag = current_dag
        best_fitness = current_fitness
        history = [{"fitness": current_fitness, "edges": current_dag.num_edges(),
                    "step": evals_done}]

        no_improve = 0

        while evals_done < budget and no_improve < self.patience:
            candidates = self._generate_candidates(current_dag)
            improved = False

            for candidate_dag in candidates:
                if evals_done >= budget:
                    break

                fitness = eval_fn(model, candidate_dag)
                evals_done += 1
                history.append({"fitness": fitness, "edges": candidate_dag.num_edges(),
                                 "step": evals_done})

                if fitness > best_fitness:
                    best_dag = candidate_dag
                    best_fitness = fitness
                    current_dag = candidate_dag
                    current_fitness = fitness
                    improved = True
                    no_improve = 0
                    logger.info(
                        f"Step {evals_done}: improved to {fitness:.4f} "
                        f"({candidate_dag.num_edges()} edges)"
                    )
                    break

            if not improved:
                no_improve += 1

        result = SearchResult(
            best_dag=best_dag,
            best_fitness=best_fitness,
            history=history,
            metadata={"method": "greedy", "total_steps": evals_done},
        )

        # Write back to library
        if self.library is not None:
            from dllm_reason.library.entry import DAGEntry
            entry = DAGEntry.from_token_dag(
                best_dag,
                task_description=self.task_description,
                source="search",
                search_method="greedy",
            )
            entry.add_benchmark_score("search_fitness", {"fitness": best_fitness})
            self.library.add(entry)
            logger.info(f"Wrote best DAG to library (fitness={best_fitness:.4f})")

        return result

    def _generate_candidates(self, dag: TokenDAG) -> list[TokenDAG]:
        """Generate candidate DAGs by adding/removing edges."""
        candidates = []
        seq_len = dag.seq_len
        adj = dag.adjacency

        for _ in range(self.num_candidates):
            # Randomly choose to add or remove
            if random.random() < 0.6 or dag.num_edges() == 0:
                # Add an edge
                src = random.randint(0, seq_len - 1)
                dst = random.randint(0, seq_len - 1)
                if src != dst and not adj[src, dst]:
                    try:
                        new_dag = dag.add_edges([(src, dst)])
                        candidates.append(new_dag)
                    except ValueError:
                        pass  # Would create cycle
            else:
                # Remove an edge
                edges = adj.nonzero(as_tuple=False)
                if len(edges) > 0:
                    idx = random.randint(0, len(edges) - 1)
                    edge = edges[idx].tolist()
                    new_dag = dag.remove_edges([tuple(edge)])
                    candidates.append(new_dag)

        return candidates
