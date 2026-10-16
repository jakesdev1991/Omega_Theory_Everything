// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>

namespace omni::bridge::quantum {

class QStarSelectiveRouter {
private:
    float base_confidence_threshold_; // tau
    float shadow_price_quantum_;      // lambda_quantum
    float budget_depletion_rate_;

public:
    QStarSelectiveRouter(float base_tau = 0.70f, float initial_shadow_price = 0.0f)
        : base_confidence_threshold_(base_tau), shadow_price_quantum_(initial_shadow_price), budget_depletion_rate_(0.01f) {}

    // Dynamic Shadow Price Calibration
    void update_shadow_price(float current_budget, float max_budget) {
        float depletion = 1.0f - (current_budget / max_budget);
        shadow_price_quantum_ = std::max(0.0f, std::exp(5.0f * (depletion - 0.8f)) - 1.0f);
    }

    // Adaptive Threshold Adjustment: tau(lambda)
    float current_threshold() const {
        // As shadow price increases (budget drops), we raise the threshold 
        // to restrict specialized fallback strictly to the most uncertain tail-risk payloads.
        return std::min(0.99f, base_confidence_threshold_ + (0.10f * shadow_price_quantum_));
    }

    enum class RouteDecision {
        FAST_CLASSICAL_BACKBONE, // Hermes / Local NPU
        SPECIALIZED_FALLBACK     // Lucifer / Quantum / Massive Frontier
    };

    RouteDecision evaluate_route(float confidence_score) const {
        if (confidence_score >= current_threshold()) {
            return RouteDecision::FAST_CLASSICAL_BACKBONE;
        }
        return RouteDecision::SPECIALIZED_FALLBACK;
    }
};

class TransferQStarDecoder {
private:
    float alpha_; // Temperature/scaling factor
    float max_reward_bound_;

public:
    TransferQStarDecoder(float alpha = 1.0f, float max_reward_bound = 10.0f)
        : alpha_(alpha), max_reward_bound_(max_reward_bound) {}

    // Token-Level Adaptive Routing (TARo) Modulation
    // Formula: pi*_Alg(z|s_t) = pi_BL(z|s_t) * exp( (1/alpha) * Q*(s_t, z) )
    void apply_transfer_qstar(std::vector<float>& logits, const std::vector<float>& target_q_values) const {
        if (logits.size() != target_q_values.size()) {
            throw std::invalid_argument("Logits and Q-values dimension mismatch.");
        }

        float max_logit = -1e9f;
        for (size_t i = 0; i < logits.size(); ++i) {
            // Apply the mathematically optimal shift (log space addition is equivalent to exp multiplication in probability space)
            float shift = target_q_values[i] / alpha_;
            logits[i] += shift;
            if (logits[i] > max_logit) max_logit = logits[i];
        }

        // Softmax stabilization
        float sum_exp = 0.0f;
        for (float& logit : logits) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }
        
        // Normalize
        for (float& prob : logits) {
            prob /= sum_exp;
        }
    }

    // Indirect Transfer via Importance Sampling
    // Adjusts trajectory rewards without retraining using ratio of policy probabilities
    float compute_importance_weight(float prob_target_reward, float prob_baseline_reward) const {
        if (prob_baseline_reward <= 1e-9f) return 1.0f; // Clamp to avoid infinity
        return prob_target_reward / prob_baseline_reward;
    }
};

} // namespace omni::bridge::quantum
