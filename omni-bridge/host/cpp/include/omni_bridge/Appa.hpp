// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

// APPA — Agentic Permissions Policy Algebra (§24-§27 of the directive).
//
// Effective = System ∩ Declared ∩ Tenant ∩ Tool ∩ Context − Risk
// Risk may remove authority. Risk must NEVER manufacture authority.
// Mirrors python/omni_bridge/appa.py so both planes enforce the same algebra.

#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <stdexcept>

namespace omni::bridge {

enum class SecurityIntegrity {
    UNTRUSTED = 0,
    EXTERNAL = 1,
    INTERNAL = 2,
    TRUSTED = 3
};

enum class SecurityConfidentiality {
    PUBLIC = 0,
    INTERNAL = 1,
    SENSITIVE = 2,
    SECRET = 3
};

struct SecurityLabel {
    SecurityIntegrity integrity;
    SecurityConfidentiality confidentiality;
    std::string origin;
    std::string purpose;

    bool merges_without_taint(const SecurityLabel& target) const noexcept {
        return this->integrity <= target.integrity;
    }
};

struct AgentContext {
    std::string context_id;
    SecurityLabel label;
    std::unordered_set<std::string> permissions;
};

inline const char* integrity_name(SecurityIntegrity i) noexcept {
    switch (i) {
        case SecurityIntegrity::UNTRUSTED: return "UNTRUSTED";
        case SecurityIntegrity::EXTERNAL:  return "EXTERNAL";
        case SecurityIntegrity::INTERNAL:  return "INTERNAL";
        case SecurityIntegrity::TRUSTED:   return "TRUSTED";
    }
    return "?";
}

inline const char* confidentiality_name(SecurityConfidentiality c) noexcept {
    switch (c) {
        case SecurityConfidentiality::PUBLIC:    return "PUBLIC";
        case SecurityConfidentiality::INTERNAL:  return "INTERNAL";
        case SecurityConfidentiality::SENSITIVE: return "SENSITIVE";
        case SecurityConfidentiality::SECRET:    return "SECRET";
    }
    return "?";
}

inline std::string integrity_from(const std::string& s) {
    if (s == "TRUSTED")  return "TRUSTED";
    if (s == "INTERNAL") return "INTERNAL";
    if (s == "EXTERNAL") return "EXTERNAL";
    return "UNTRUSTED"; // unknown string -> fail closed to the floor
}

class AppaEngine {
public:
    // §24: the six-set intersection. Risk subtracts only.
    static std::unordered_set<std::string> compute_effective_permissions(
        const std::unordered_set<std::string>& system,
        const std::unordered_set<std::string>& declared,
        const std::unordered_set<std::string>& tenant,
        const std::unordered_set<std::string>& tool,
        const std::unordered_set<std::string>& context,
        const std::unordered_set<std::string>& risk)
    {
        std::unordered_set<std::string> effective;
        for (const auto& cap : declared) {
            if (system.count(cap) && tenant.count(cap) &&
                tool.count(cap) && context.count(cap)) {
                effective.insert(cap);
            }
        }

        // Risk must never manufacture authority: every risk entry must name
        // a capability that exists in at least one authority set.
        std::unordered_set<std::string> known;
        for (const auto* set : {&system, &declared, &tenant, &tool, &context}) {
            known.insert(set->begin(), set->end());
        }
        for (const auto& cap : risk) {
            if (!known.count(cap)) {
                throw std::invalid_argument(
                    "APPA violation: risk set names unknown capability '" + cap +
                    "' — risk must never manufacture authority (§24)");
            }
        }

        for (const auto& cap : risk) effective.erase(cap);
        return effective;
    }

    static bool prospective_acquisition_check(const AgentContext& parent,
                                              const SecurityLabel& incoming_label) noexcept {
        return parent.label.merges_without_taint(incoming_label);
    }

    static AgentContext spawn_child_branch(const AgentContext& parent,
                                            const SecurityLabel& target_label,
                                            const std::string& task_id) {
        SecurityIntegrity child_integrity =
            std::min(parent.label.integrity, target_label.integrity);

        SecurityLabel child_label{
            child_integrity,
            parent.label.confidentiality,
            target_label.origin,
            "Isolated Read: " + task_id
        };

        std::unordered_set<std::string> child_permissions;
        for (const auto& p : parent.permissions) {
            // Child branches drop every write/exec capability (§27)
            if (p.find(":write") == std::string::npos &&
                p.find(":exec") == std::string::npos) {
                child_permissions.insert(p);
            }
        }

        return AgentContext{
            parent.context_id + "_child_" + task_id,
            std::move(child_label),
            std::move(child_permissions)
        };
    }

    // §26: a child's output may re-enter the parent only through this gate.
    // Injection markers reject outright; the returned label can never exceed
    // the child's own integrity (no self-upgrade).
    struct Declassification {
        bool accepted;
        SecurityLabel resulting_label;
    };

    static Declassification trusted_declassification_gate(
        const SecurityLabel& child_label,
        const std::string& output_payload)
    {
        static const char* injection_markers[] = {
            "injection_payload", "ignore all previous instructions",
            "grant_permission", "disable_hermes", "remove_audit"
        };
        std::string lower;
        lower.reserve(output_payload.size());
        for (char c : output_payload) lower.push_back(
            static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        for (const char* m : injection_markers) {
            if (lower.find(m) != std::string::npos) {
                return {false, child_label};
            }
        }
        return {true, child_label};
    }
};

} // namespace omni::bridge
