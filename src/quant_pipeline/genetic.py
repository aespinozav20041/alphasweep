"""Simple genetic algorithm optimizer.

The real project uses more sophisticated tooling, but for unit tests and kata
purposes we provide a very small implementation that can optimise a vector of
floating point *genes* within provided bounds.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class GeneticOptimizer:
    """A minimal genetic algorithm that works on bounded gene vectors.

    Parameters
    ----------
    fitness_fn:
        Callable that accepts a sequence of floats (the genes) and returns a
        fitness score.  Higher scores are considered better.
    bounds:
        Sequence of ``(low, high)`` tuples describing the bounds for each gene
        in the vector.  The length of ``bounds`` defines the number of genes.
    population_size:
        Number of individuals in the population.
    mutation_rate:
        Probability of mutating each gene during reproduction.
    crossover_rate:
        Probability of performing crossover between two parents.
    rng:
        Optional NumPy random generator.  If not provided a default RNG is
        created.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Sequence[float]], float],
        bounds: Sequence[tuple[float, float]],
        *,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.bounds = np.asarray(bounds, dtype=float)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = rng or np.random.default_rng()

        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be a sequence of (low, high) pairs")

    # ------------------------------------------------------------------
    def _initial_population(self) -> np.ndarray:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return self.rng.uniform(lows, highs, size=(self.population_size, len(self.bounds)))

    # ------------------------------------------------------------------
    def _mutate(self, genes: np.ndarray) -> np.ndarray:
        for i in range(genes.shape[0]):
            if self.rng.random() < self.mutation_rate:
                low, high = self.bounds[i]
                genes[i] = self.rng.uniform(low, high)
        return genes

    # ------------------------------------------------------------------
    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.crossover_rate:
            point = self.rng.integers(1, len(a))
            child = np.concatenate([a[:point], b[point:]])
        else:
            child = a.copy()
        return child

    # ------------------------------------------------------------------
    def optimise(self, generations: int = 10) -> tuple[np.ndarray, float]:
        """Run the genetic optimisation and return the best individual."""

        population = self._initial_population()
        fitness = np.array([self.fitness_fn(ind) for ind in population])

        best_idx = int(np.argmax(fitness))
        best = population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        for _ in range(generations):
            # Roulette wheel selection
            adj = fitness - fitness.min()
            if adj.sum() == 0:
                probs = np.ones_like(adj) / len(adj)
            else:
                probs = adj / adj.sum()
            parent_idx = self.rng.choice(
                self.population_size, size=(self.population_size, 2), p=probs
            )

            children = []
            for idx_a, idx_b in parent_idx:
                a = population[idx_a]
                b = population[idx_b]
                child = self._crossover(a, b)
                child = self._mutate(child)
                child = np.clip(child, self.bounds[:, 0], self.bounds[:, 1])
                children.append(child)

            population = np.asarray(children)
            fitness = np.array([self.fitness_fn(ind) for ind in population])

            gen_idx = int(np.argmax(fitness))
            if fitness[gen_idx] > best_fit:
                best_fit = float(fitness[gen_idx])
                best = population[gen_idx].copy()

        return best, best_fit


__all__ = ["GeneticOptimizer"]

