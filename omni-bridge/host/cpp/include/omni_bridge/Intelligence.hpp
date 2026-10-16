// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <optional>

namespace omni::bridge {

// ODAR: Variational Free Energy Risk Penalty
class OdarEngine {
private:
    float lambda_penalty_;

public:
    explicit OdarEngine(float lambda_penalty = 1.5f) : lambda_penalty_(lambda_penalty) {}

    struct EvaluationResult {
        float free_energy;
        float mean_surprisal;
        float varentropy;
    };

    EvaluationResult evaluate_candidate_chain(const std::vector<float>& log_probs) const noexcept {
        if (log_probs.empty() || log_probs.size() < 2) {
            return {__builtin_inff(), __builtin_inff(), 0.0f};
        }

        float sum_surprisal = 0.0f;
        for (float lp : log_probs) {
            sum_surprisal += (-lp);
        }
        
        float mean_surprisal = sum_surprisal / log_probs.size();
        
        float variance_sum = 0.0f;
        for (float lp : log_probs) {
            float diff = (-lp) - mean_surprisal;
            variance_sum += (diff * diff);
        }
        float varentropy = variance_sum / log_probs.size();
        
        float free_energy = mean_surprisal + (lambda_penalty_ * varentropy);
        
        return {free_energy, mean_surprisal, varentropy};
    }
};

// ATOMS: Psychological Profile Generator
struct AtomsProfile {
    std::string beliefs;
    std::string intentions;
    std::string desires;
    std::string emotions;
    std::string knowledge;
    std::string percepts;
    std::string non_literal;

    bool requires_empathic_routing() const noexcept {
        auto contains = [](const std::string& str, const std::string& sub) {
            return str.find(sub) != std::string::npos;
        };
        // Simplified check
        return contains(emotions, "frustration") || contains(emotions, "urgency") ||
               contains(beliefs, "error") || contains(beliefs, "false");
    }
};

} // namespace omni::bridge
