// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include "omni_bridge/Appa.hpp"
#include "omni_bridge/Intelligence.hpp"
#include <string>
#include <algorithm>

namespace omni::bridge {

class OmniRouter {
private:
    float hermes_cost_weight_;
    float lucifer_cost_weight_;
    float cumulative_spend_{0.0f};
    float budget_limit_{1000.0f};

    float calculate_primal_dual_cost(float base_cost, int estimated_tokens, float risk_multiplier) const noexcept {
        float dual_variable = std::max(0.0f, (cumulative_spend_ / budget_limit_) - 0.8f);
        float immediate_cost = base_cost * estimated_tokens * risk_multiplier;
        return immediate_cost + (dual_variable * immediate_cost);
    }

public:
    OmniRouter(float hermes_weight = 1.0f, float lucifer_weight = 0.5f)
        : hermes_cost_weight_(hermes_weight), lucifer_cost_weight_(lucifer_weight) {}

    std::string route_task(const AtomsProfile& profile, const SecurityLabel& label, int estimated_tokens = 100) const noexcept {
        // Hard Constraints
        if (label.integrity == SecurityIntegrity::TRUSTED && 
            profile.intentions.find("Action execution") != std::string::npos) {
            return "hermes";
        }

        if (profile.requires_empathic_routing()) {
            return "lucifer";
        }

        // Soft Constraints
        float hermes_cost = calculate_primal_dual_cost(hermes_cost_weight_, estimated_tokens, 1.2f);
        float lucifer_cost = calculate_primal_dual_cost(lucifer_cost_weight_, estimated_tokens, 1.8f);

        return (hermes_cost <= lucifer_cost) ? "hermes" : "lucifer";
    }
};

} // namespace omni::bridge
