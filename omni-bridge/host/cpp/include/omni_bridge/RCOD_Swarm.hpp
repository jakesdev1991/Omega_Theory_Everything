// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>

namespace omni::bridge::quantum {

enum class SwarmPhase {
    VACUUM,
    PRE_SHOCK,
    PHASE_TRANSITION_COLLAPSE
};

struct SwarmMetrics {
    float median_mu;
    float macro_mu;
    float swarm_dissonance;
    SwarmPhase phase;
};

class RCODStochasticSwarmEngine {
private:
    size_t macro_window_size_;
    size_t num_stochastic_leaves_;
    size_t min_leaf_size_;
    size_t max_leaf_size_;

    // Calculate Chain Overlap Density (Phi) = cos(theta)
    static float calculate_phi(const std::vector<float>& s_prev, const std::vector<float>& s_curr) {
        float dot = std::inner_product(s_prev.begin(), s_prev.end(), s_curr.begin(), 0.0f);
        float norm_prev = std::sqrt(std::inner_product(s_prev.begin(), s_prev.end(), s_prev.begin(), 0.0f));
        float norm_curr = std::sqrt(std::inner_product(s_curr.begin(), s_curr.end(), s_curr.begin(), 0.0f));
        if (norm_prev == 0.0f || norm_curr == 0.0f) return 1.0f;
        float cos_theta = dot / (norm_prev * norm_curr);
        return std::clamp(cos_theta, -1.0f, 1.0f);
    }

    // Calculate Reverse Chain Overlap Density (mu) = sin(theta) = sqrt(1 - Phi^2)
    static float calculate_mu(float phi) {
        return std::sqrt(std::max(0.0f, 1.0f - (phi * phi)));
    }

    float compute_window_mu(const std::vector<std::vector<float>>& trajectory, size_t start, size_t end) const {
        if (end <= start + 1) return 0.0f;
        std::vector<float> mu_vals;
        for (size_t i = start + 1; i < end; ++i) {
            float phi = calculate_phi(trajectory[i-1], trajectory[i]);
            mu_vals.push_back(calculate_mu(phi));
        }
        return std::accumulate(mu_vals.begin(), mu_vals.end(), 0.0f) / mu_vals.size();
    }

public:
    RCODStochasticSwarmEngine(size_t macro_window = 100, size_t leaves = 10, size_t min_leaf = 5, size_t max_leaf = 20)
        : macro_window_size_(macro_window), num_stochastic_leaves_(leaves), min_leaf_size_(min_leaf), max_leaf_size_(max_leaf) {}

    SwarmMetrics evaluate_trajectory(const std::vector<std::vector<float>>& continuous_trajectory) {
        if (continuous_trajectory.size() < min_leaf_size_) {
            return {0.0f, 0.0f, 0.0f, SwarmPhase::VACUUM};
        }

        size_t n = continuous_trajectory.size();
        
        // 1. Macro Anchor Evaluation
        size_t macro_start = (n > macro_window_size_) ? n - macro_window_size_ : 0;
        float macro_mu = compute_window_mu(continuous_trajectory, macro_start, n);

        // 2. Stochastic Leaves Evaluation
        std::mt19937 rng(42); // Deterministic for reproducibility in pipeline
        std::vector<float> leaf_mus;
        for (size_t i = 0; i < num_stochastic_leaves_; ++i) {
            std::uniform_int_distribution<size_t> size_dist(min_leaf_size_, std::min(max_leaf_size_, n));
            size_t leaf_size = size_dist(rng);
            
            std::uniform_int_distribution<size_t> start_dist(macro_start, n - leaf_size);
            size_t start_idx = start_dist(rng);
            
            leaf_mus.push_back(compute_window_mu(continuous_trajectory, start_idx, start_idx + leaf_size));
        }

        // 3. Compute Medians and Dissonance
        std::vector<float> sorted_mus = leaf_mus;
        std::sort(sorted_mus.begin(), sorted_mus.end());
        float median_mu = sorted_mus[sorted_mus.size() / 2];

        float sum_mu = std::accumulate(leaf_mus.begin(), leaf_mus.end(), 0.0f);
        float mean_mu = sum_mu / leaf_mus.size();
        float variance = 0.0f;
        for (float mu : leaf_mus) {
            variance += (mu - mean_mu) * (mu - mean_mu);
        }
        float swarm_dissonance = std::sqrt(variance / leaf_mus.size());

        // 4. Phase Transition Logic
        SwarmPhase phase = SwarmPhase::VACUUM;
        float threshold_dissonance = 0.15f; 
        float threshold_macro = 0.6f;       

        if (median_mu > threshold_macro && macro_mu > threshold_macro) {
            phase = SwarmPhase::PHASE_TRANSITION_COLLAPSE;
        } else if (swarm_dissonance > threshold_dissonance) {
            phase = SwarmPhase::PRE_SHOCK;
        }

        return {median_mu, macro_mu, swarm_dissonance, phase};
    }
};

} // namespace omni::bridge::quantum
