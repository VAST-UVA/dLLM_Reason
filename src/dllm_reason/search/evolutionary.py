"""Evolutionary search for DAG structure optimization.

Maintains a population of DAGs and evolves them through selection,
crossover, and mutation. Expected to be the most practical search method.

Supports optional DAG Library integration:
- Seed population from library (retrieved by task similarity)
- Write back best results to library after search
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import torch

from dllm_reason.graph.dag import TokenDAG
from dllm_reason.graph.constraints import topological_mutation
from dllm_reason.library.config import RetrievalMode
from dllm_reason.search.base import DAGSearcher, SearchResult
from dllm_reason.utils.registry import SEARCH_REGISTRY
from dllm_reason.utils.logging import get_logger

logger = get_logger(__name__)


@SEARCH_REGISTRY.register("evolutionary")
class EvolutionarySearch(DAGSearcher):
    """Evolutionary DAG search with population-based optimization.

    Algorithm:
    1. Initialize population from templates + random DAGs
    2. Each generation: evaluate fitness, select parents, crossover, mutate
    3. Replace worst with offspring
    4. Repeat for `budget` total evaluations

    If a DAG Library is provided, seeds population from retrieved DAGs
    and writes back the best result after search completes.
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        initial_dags: list[TokenDAG] | None = None,
        init_templates: list[str] | None = None,
        library: Optional["DAGStore"] = None,
        library_config: Optional["LibraryConfig"] = None,
        task_description: str = "",
    ):
        """
        Args:
            initial_dags:    Explicit list of seed DAGs (added before templates).
            init_templates:  Names from ``TEMPLATE_NAMES`` to include in the
                             initial population.  Pass ``None`` to use a compact
                             default set; pass ``[]`` to disable templates entirely
                             (population filled with random DAGs only).
                             Default set: ["cot", "skeleton", "bidirectional",
                             "answer_first", "empty"].
        """
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.initial_dags = initial_dags or []
        # None → use default set; [] → no templates
        self.init_templates: list[str] | None = init_templates
        self.library = library
        self.library_config = library_config
        self.task_description = task_description

    def search(
        self,
        model,
        eval_fn: Callable,
        seq_len: int,
        budget: int = 100,
        **kwargs,
    ) -> SearchResult:
        device = model.device

        # Initialize population
        population = self._init_population(seq_len, device)
        fitnesses = []

        # Evaluate initial population
        for dag in population:
            fitnesses.append(eval_fn(model, dag))

        evals_done = len(population)
        history = []

        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_dag = population[best_idx]
        best_fitness = fitnesses[best_idx]

        logger.info(f"Initial best fitness: {best_fitness:.4f}")
        history.append({"fitness": best_fitness, "step": 0, "generation": 0})

        generation = 0
        while evals_done < budget:
            generation += 1

            # Selection: tournament
            parents = self._tournament_select(population, fitnesses, k=2)

            # Crossover
            if random.random() < self.crossover_rate:
                offspring = self._crossover(parents[0], parents[1], seq_len, device)
            else:
                offspring = random.choice(parents)

            # Mutation
            if random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)

            # Evaluate
            offspring_fitness = eval_fn(model, offspring)
            evals_done += 1

            # Replace worst
            worst_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if offspring_fitness > fitnesses[worst_idx]:
                population[worst_idx] = offspring
                fitnesses[worst_idx] = offspring_fitness

            # Track best
            if offspring_fitness > best_fitness:
                best_dag = offspring
                best_fitness = offspring_fitness
                logger.info(f"Gen {generation}: new best {best_fitness:.4f} ({best_dag.num_edges()} edges)")

            history.append({
                "fitness": best_fitness,
                "step": evals_done,
                "generation": generation,
                "pop_mean_fitness": sum(fitnesses) / len(fitnesses),
            })

        result = SearchResult(
            best_dag=best_dag,
            best_fitness=best_fitness,
            history=history,
            metadata={"method": "evolutionary", "generations": generation, "evals": evals_done},
        )

        # Write back best DAG to library
        self._writeback_to_library(best_dag, best_fitness)

        return result

    # Default templates used when init_templates=None
    _DEFAULT_TEMPLATES = ["cot", "skeleton", "bidirectional", "answer_first", "empty"]

    def _init_population(self, seq_len: int, device) -> list[TokenDAG]:
        """Initialize population: explicit seeds → library → templates → random.

        Priority order (earlier sources take slots first):
        1. ``initial_dags``  passed at construction
        2. Library-retrieved DAGs (if library configured)
        3. Named templates (``init_templates`` list or default set)
        4. Random DAGs to fill remaining slots
        """
        pop: list[TokenDAG] = list(self.initial_dags)

        # 1. Library seeds
        if self.library is not None and self.library_config is not None:
            library_dags = self._seed_from_library(seq_len, device)
            pop.extend(library_dags)
            if library_dags:
                logger.info(f"Seeded {len(library_dags)} DAGs from library")

        # 2. Template seeds
        template_names = (
            self._DEFAULT_TEMPLATES
            if self.init_templates is None
            else self.init_templates
        )
        if template_names and len(pop) < self.population_size:
            from dllm_reason.graph.templates import build_all_templates
            templates = build_all_templates(seq_len, device=device, names=template_names)
            slots_left = self.population_size - len(pop)
            selected = list(templates.values())[:slots_left]
            pop.extend(selected)
            logger.info(
                f"Seeded {len(selected)} template DAGs "
                f"({list(templates.keys())[:len(selected)]})"
            )

        # 3. Random DAGs to fill remaining slots
        while len(pop) < self.population_size:
            from dllm_reason.graph.templates import random_dag
            density = random.uniform(0.01, 0.2)
            pop.append(random_dag(seq_len, density=density, device=device))

        return pop[:self.population_size]

    def _seed_from_library(self, seq_len: int, device) -> list[TokenDAG]:
        """Retrieve candidate DAGs from library to seed the population."""
        if self.library is None or self.library_config is None:
            return []
        if not self.library_config.retrieval.enabled:
            return []

        from dllm_reason.library.retrieval import RetrievalQuery, create_retrieval_channel
        from dllm_reason.library.embedder import create_embedder

        top_k = min(
            self.library_config.retrieval.top_k,
            self.population_size // 2,  # at most half the population from library
        )

        dags = []
        try:
            # Try semantic retrieval first
            if RetrievalMode.SEMANTIC in self.library_config.active_retrieval_channels():
                embedder = create_embedder("random")  # lightweight fallback
                channel = create_retrieval_channel(
                    RetrievalMode.SEMANTIC, self.library_config.retrieval, embedder
                )
                query = RetrievalQuery(task_description=self.task_description)
                results = channel.retrieve(query, self.library, top_k=top_k)
                for entry, _score in results:
                    if entry.seq_len == seq_len:
                        dags.append(entry.to_token_dag(device=str(device)))

            # Also pull top-performing DAGs
            if RetrievalMode.PERFORMANCE in self.library_config.active_retrieval_channels():
                channel = create_retrieval_channel(
                    RetrievalMode.PERFORMANCE, self.library_config.retrieval
                )
                query = RetrievalQuery(target_metric="accuracy")
                results = channel.retrieve(query, self.library, top_k=top_k)
                for entry, _score in results:
                    if entry.seq_len == seq_len:
                        dags.append(entry.to_token_dag(device=str(device)))
        except Exception as e:
            logger.warning(f"Library seeding failed, falling back to random: {e}")

        return dags

    def _writeback_to_library(self, best_dag: TokenDAG, best_fitness: float) -> None:
        """Store the best DAG found back into the library."""
        if self.library is None:
            return

        from dllm_reason.library.entry import DAGEntry

        entry = DAGEntry.from_token_dag(
            best_dag,
            task_description=self.task_description,
            source="search",
            search_method="evolutionary",
        )
        entry.add_benchmark_score("search_fitness", {"fitness": best_fitness})
        self.library.add(entry)
        logger.info(f"Wrote best DAG to library (fitness={best_fitness:.4f})")

    def _tournament_select(
        self,
        population: list[TokenDAG],
        fitnesses: list[float],
        k: int = 2,
        tournament_size: int = 3,
    ) -> list[TokenDAG]:
        """Tournament selection: pick k parents."""
        parents = []
        for _ in range(k):
            contestants = random.sample(range(len(population)), min(tournament_size, len(population)))
            winner = max(contestants, key=lambda i: fitnesses[i])
            parents.append(population[winner])
        return parents

    def _crossover(self, dag1: TokenDAG, dag2: TokenDAG, seq_len: int, device) -> TokenDAG:
        """Crossover: combine subgraphs from two parents.

        Strategy: for each position, take its incoming edges from either
        parent with equal probability.
        """
        adj1 = dag1.adjacency
        adj2 = dag2.adjacency

        # For each column (child position), pick parent's edges
        mask = torch.rand(seq_len, device=device) > 0.5
        new_adj = torch.where(
            mask.unsqueeze(0).expand(seq_len, seq_len),
            adj1.to(device), adj2.to(device),
        )

        new_dag = TokenDAG(new_adj)
        if not new_dag.is_valid():
            # If cycle created, fall back to one parent
            return dag1
        return new_dag

    def _mutate(self, dag: TokenDAG) -> TokenDAG:
        """Mutate by adding/removing random edges."""
        try:
            num_add = random.randint(0, 3)
            num_remove = random.randint(0, 3)
            return topological_mutation(dag, num_add=num_add, num_remove=num_remove)
        except (ValueError, RuntimeError):
            return dag
