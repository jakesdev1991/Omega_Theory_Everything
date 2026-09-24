# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 4: System Evolution and Active Shredding (Omega Theory v4.0)

Simulates Wright-Fisher dynamics and Chain Overlap Density (COD) pruning
under informational evolution as outlined in Omega Theory v4.0 Section 6.
"""

import numpy as np


def compute_fitness(genomes: np.ndarray, gen: int) -> np.ndarray:
    genome_size = genomes.shape[1]
    additive = np.sum(genomes, axis=1) / genome_size
    epistasis = (genomes[:, 0] * genomes[:, 1] + genomes[:, 2] * genomes[:, 3]) * 0.1
    cit_bonus = np.where((gen > 300) & (genomes[:, 4] == 1), 0.3, 0.0)
    drift = np.random.normal(0, 0.05, genomes.shape[0])
    return additive + epistasis + cit_bonus + drift


def wright_fisher_step(pop: np.ndarray, mu: float, fitness: np.ndarray) -> np.ndarray:
    n_pop, genome_size = pop.shape
    exp_f = np.exp(fitness - np.max(fitness))
    probs = exp_f / np.sum(exp_f)
    parent_indices = np.random.choice(n_pop, size=n_pop, p=probs)
    parents = pop[parent_indices]
    mutations = np.random.binomial(1, mu, size=(n_pop, genome_size))
    offspring = np.abs(parents - mutations)
    return offspring


def main() -> None:
    np.random.seed(42)
    n_pop, genome_size = 50, 10
    mu = 0.005
    generations = 500

    pop = np.random.binomial(1, 0.5, size=(n_pop, genome_size))
    fitness_history = []

    for gen in range(generations):
        fitness = compute_fitness(pop, gen)
        pop = wright_fisher_step(pop, mu, fitness)
        fitness_history.append(float(np.mean(fitness)))

    print("Simulation 4 completed successfully.")
    print(f"Initial population fitness: {fitness_history[0]:.4f}")
    print(f"Final population fitness: {fitness_history[-1]:.4f}")


if __name__ == "__main__":
    main()
