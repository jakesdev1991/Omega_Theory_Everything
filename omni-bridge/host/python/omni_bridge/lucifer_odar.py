# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Any, List, Tuple


class ODAREngine:
    """
    Open-Domain Adaptive Reasoner (ODAR).
    Computes the Variational Free Energy Objective with a Varentropy Risk Penalty.
    Minimizes: Surprisal + lambda * Var(Surprisal)
    """

    def __init__(self, lambda_penalty: float = 1.5):
        self.lambda_penalty = lambda_penalty

    def compute_varentropy_penalty(self, log_probs: List[float]) -> float:
        """
        Calculates the variance of the negative log probabilities (surprisal).
        High variance indicates token volatility (hallucinations/uncertainty).
        """
        if not log_probs or len(log_probs) < 2:
            return 0.0

        # Surprisal is negative log probability
        surprisals = [-lp for lp in log_probs]

        mean_surprisal = sum(surprisals) / len(surprisals)
        variance = sum((s - mean_surprisal) ** 2 for s in surprisals) / len(surprisals)

        return variance

    def evaluate_candidate_chain(
        self, log_probs: List[float]
    ) -> Tuple[float, float, float]:
        """
        Evaluates a reasoning chain candidate.
        Returns: (free_energy, mean_surprisal, varentropy)
        Lower free energy is better.
        """
        if not log_probs:
            return float("inf"), float("inf"), 0.0

        surprisals = [-lp for lp in log_probs]
        mean_surprisal = sum(surprisals) / len(surprisals)
        varentropy = self.compute_varentropy_penalty(log_probs)

        # Variational Free Energy Objective: F(y|x) = Mean(Surprisal) + lambda * Var(Surprisal)
        free_energy = mean_surprisal + (self.lambda_penalty * varentropy)

        return free_energy, mean_surprisal, varentropy

    def select_best_candidate(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """
        Takes a list of candidate dictionaries containing 'log_probs' and 'text'.
        Selects the candidate with the lowest Variational Free Energy.
        """
        best_candidate = None
        best_score = float("inf")

        for idx, candidate in enumerate(candidates):
            log_probs = candidate.get("log_probs", [])
            free_energy, _, var = self.evaluate_candidate_chain(log_probs)

            candidate["free_energy"] = free_energy
            candidate["varentropy"] = var

            if free_energy < best_score:
                best_score = free_energy
                best_candidate = candidate

        return best_candidate
