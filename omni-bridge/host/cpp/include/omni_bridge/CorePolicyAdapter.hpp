// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// CorePolicyAdapter — unifies the deployed daemon with the portable core.
//
// The host daemon previously implemented APPA/routing inline (documented
// gap in host/README.md). This adapter closes that gap: every request the
// daemon accepts is converted into a core Proposal + ExecutionContext
// and authorized by OmniBridge::evaluate() — the same 8-step pipeline
// with the type-level guarantees the core's acceptance suite proves:
//
//   * governance-only capabilities (PolicyWrite, SkillPromote,
//     MintOrMoveValue, SecretRead, CredentialRead) can NEVER be granted
//     by an automated path — only NeedsGovernance, resolved solely by
//     human_governance() (§22);
//   * risk ladder with monotonic masks (Critical = zero capabilities);
//   * audit gate: ledger unavailable => High+ risk fails closed;
//   * tier = max(tool floor, policy floor), T5 prohibited rejected;
//   * skill eligibility checked in-pipeline (§57.3);
//   * kill switches parsed once, fail closed.
//
// The vendored core (repo copy) uses its native namespace `omni` (kept native in the repo copy;
// time so it cannot collide with the daemon's own omni_core::bridge types).
// Every core reference below is fully qualified — no using-directives.
//
// The daemon retains: socket transport, its persistent BLAKE3 JSONL
// ledger (mirroring the core's in-memory chain for durability), Wasmtime
// T1 enforcement, adapter dispatch, and ToE grounding — all enforcement,
// no policy. Policy authority lives in the core from here on.

#pragma once

#include "../../vendor/omni_bridge_core.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace omni::bridge {

// Capability mapping between the daemon's string world and the core's
// typed CapSet. Bridge dispatch exercises read + IPC + ledger append;
// process execution belongs to the sandbox tier, not to the routing
// request, so it is NOT part of the dispatch tool's requirements.
inline constexpr omni_core::CapSet kDispatchCaps =
    omni_core::kCapNetRead | omni_core::kCapNetDial |
    omni_core::kCapFsRead | omni_core::kCapLedgerAppend;
inline constexpr omni_core::CapSet kDaemonSystemCaps =
    kDispatchCaps | omni_core::kCapFsWrite | omni_core::kCapProcExec;

class CorePolicyAdapter {
public:
    explicit CorePolicyAdapter(omni_core::CapSet system_caps,
                               omni_core::CapSet tenant_caps)
        : config_(make_config()),
          bridge_(config_, system_caps, tenant_caps) {
        bridge_lucifer_enabled_ = config_.lucifer_enabled;
        register_default_tools();
    }

    // Map the daemon's integrity string to the core's label lattice.
    // The core's lattice is coarser by design (binary integrity): only
    // TRUSTED maps to Trusted; INTERNAL/EXTERNAL are honest data
    // classifications that still start at the Untrusted floor — the
    // daemon's attestation upgrade path (future ed25519 envelopes)
    // is what will mint Trusted, not the label string.
    static omni_core::SecurityLabel to_label(const std::string& integrity,
                                             const std::string& origin) {
        omni_core::Integrity i = omni_core::Integrity::Untrusted;
        if (integrity == "TRUSTED") i = omni_core::Integrity::Trusted;
        omni_core::Origin o = omni_core::Origin::System;
        if (origin == "raw_socket") o = omni_core::Origin::Retrieved;
        else if (origin == "smoke_test" || origin == "it_test" ||
                 origin == "bench") o = omni_core::Origin::Human;
        else if (origin == "web") o = omni_core::Origin::Retrieved;
        return {i, omni_core::Confidentiality::Internal, o,
                omni_core::Purpose::Task};
    }

    // Authorize one daemon request through the core pipeline.
    // Returns the core Decision; the daemon enforces it.
    [[nodiscard]] omni_core::Decision authorize(
        const std::string& trace_id,
        const std::string& tool_id,          // "" => default dispatch tool
        const std::string& integrity,
        const std::string& origin,
        bool injected,                        // classifier verdict
        bool child_branch,                    // §27 restricted context?
        omni_core::Risk risk = omni_core::Risk::Low) {

        const omni_core::SecurityLabel label = to_label(integrity, origin);
        // Injection evidence escalates risk and clamps the label floor.
        const omni_core::SecurityLabel effective =
            injected ? omni_core::untrusted_external() : label;

        omni_core::ExecutionContext ctx;
        ctx.id = child_branch ? "child:" + trace_id : "root";
        ctx.declared_caps = kDispatchCaps;
        ctx.context_caps = child_branch
            ? static_cast<omni_core::CapSet>(kDispatchCaps & ~omni_core::kCapFsWrite)
            : static_cast<omni_core::CapSet>(~omni_core::kCapNone);
        ctx.risk_ceiling = omni_core::Risk::Medium;
        ctx.content_label = effective;
        ctx.is_child = child_branch;

        omni_core::Proposal p;
        p.trace_id = trace_id;
        p.tool_id = tool_id.empty() ? default_tool_ : tool_id;
        p.route_id = "bridge";
        p.requested_caps = kDispatchCaps;
        p.assessed_risk = injected ? omni_core::Risk::High : risk;
        p.content_label = effective;
        // Kill-switch degradation: when Lucifer is disabled, the request
        // is processed by the governance plane instead — an exploration
        // proposal with lucifer_enabled=false would be rejected by the
        // core's own §72 gate, but the daemon's contract is degraded
        // service (hermes), not refusal. Only label it a Lucifer
        // proposal when the exploration plane is actually available.
        p.from_lucifer = child_branch && bridge_lucifer_enabled_;

        return bridge_.evaluate(p, ctx);
    }

    // The single human gate (§22). In production this is a separate
    // authenticated channel; the daemon never calls this on an
    // agent-provided name. The evaluation stalls at NeedsGovernance
    // (T4_HumanApproved) and authorize_governance converts it with the
    // minted approval — the human approval IS the authorization.
    [[nodiscard]] omni_core::Decision resolve_governance(
        const std::string& trace_id,
        const omni_core::HumanGovernor& gov,
        const std::string& action) {
        auto approval = bridge_.human_governance(gov, action);
        omni_core::Proposal p;
        p.trace_id = trace_id;
        p.tool_id = kGovernanceTool;
        p.route_id = "governance";
        p.requested_caps = omni_core::kGovernanceOnlyCaps;
        p.assessed_risk = omni_core::Risk::Medium;
        p.content_label = {omni_core::Integrity::Trusted,
                           omni_core::Confidentiality::Internal,
                           omni_core::Origin::Human,
                           omni_core::Purpose::Governance};
        p.from_lucifer = false;
        omni_core::ExecutionContext ctx;
        ctx.id = "governance";
        ctx.declared_caps = static_cast<omni_core::CapSet>(~omni_core::kCapNone);
        ctx.context_caps = ctx.declared_caps;
        ctx.risk_ceiling = omni_core::Risk::Critical;
        ctx.content_label = p.content_label;
        auto d = bridge_.evaluate(p, ctx);
        if (d.status ==
            omni_core::Decision::Status::NeedsGovernance) {
            d = bridge_.authorize_governance(d, p, approval);
        }
        return d;
    }

    // Route instability feedback (§57.6/7).
    void record_route_failure(const std::string& id) {
        routes_.record_failure(id);
    }
    void record_route_success(const std::string& id) {
        routes_.record_success(id);
    }
    [[nodiscard]] bool route_unstable() const {
        return routes_.unstable();
    }
    [[nodiscard]] std::string select_route(const std::string& preferred) {
        const auto* r = routes_.select(preferred);
        return r ? r->id : std::string("hermes");
    }

    [[nodiscard]] bool ledger_healthy() const {
        return bridge_.ledger().healthy() &&
               !bridge_.ledger().read_only();
    }

    // Whether the exploration plane is enabled per the core config
    // (kill switch state, parsed once at construction).
    [[nodiscard]] bool lucifer_enabled() const {
        return config_.lucifer_enabled;
    }

    // Skill lifecycle passthrough: registration, audited transitions,
    // and governed promotion (the only path to Promoted).
    [[nodiscard]] std::uint32_t register_skill(
        const omni_core::Sha256::Digest& digest) {
        return bridge_.register_skill(digest);
    }
    [[nodiscard]] bool transition_skill(std::uint32_t id,
                                        omni_core::SkillState to,
                                        std::string* why = nullptr) {
        return bridge_.transition_skill(id, to, nullptr, why);
    }
    [[nodiscard]] bool promote_skill(std::uint32_t id,
                                     const omni_core::HumanGovernor& gov,
                                     std::string* why = nullptr) {
        auto approval = bridge_.human_governance(gov, "skill-promotion");
        return bridge_.transition_skill(id, omni_core::SkillState::Promoted,
                                        &approval, why);
    }
    [[nodiscard]] bool skill_eligible(std::uint32_t id) const noexcept {
        return bridge_.skill_eligible(id);
    }

    [[nodiscard]] const omni_core::OmniBridge& core() const {
        return bridge_;
    }

private:
    static omni_core::BridgeConfig make_config() {
        omni_core::BridgeConfig c;
        // §72 kill switches — parsed once, fail closed
        auto off = [](const char* name) {
            const char* v = std::getenv(name);
            return v && (*v == '1' || *v == 't' || *v == 'T');
        };
        if (off("OMNI_LUCIFER_OFF")) c.lucifer_enabled = false;
        if (off("OMNI_GENERATION_OFF")) c.generation = false;
        if (off("OMNI_HERMES_RETRIEVAL_OFF")) c.hermes_retrieval = false;
        if (off("OMNI_LEARNING_OFF")) c.learning = false;
        if (off("OMNI_AUDIT_READONLY")) c.audit_readonly = true;
        if (off("OMNI_DYNAMIC_EXPERTISE_OFF")) c.dynamic_expertise = false;
        return c;
    }

    void register_default_tools() {
        // The dispatch tool: what a normal bridge request exercises.
        omni_core::ToolManifest dispatch;
        dispatch.id = default_tool_;
        dispatch.version = 1;
        dispatch.required_caps = kDispatchCaps;
        dispatch.min_tier = omni_core::SandboxTier::T0_InProcess;
        dispatch.deterministic = true;
        dispatch.side_effects = true;   // ledger append + IPC dispatch
        dispatch.requires_generation = false;
        dispatch.timeout_ms = 5000;
        std::string why;
        if (!bridge_.register_tool(dispatch, &why)) {
            throw std::runtime_error("core tool registration failed: " + why);
        }
        // Governance action tool: needs the human gate by construction.
        omni_core::ToolManifest gov_tool;
        gov_tool.id = kGovernanceTool;
        gov_tool.version = 1;
        gov_tool.required_caps = omni_core::kGovernanceOnlyCaps;
        gov_tool.min_tier = omni_core::SandboxTier::T4_HumanApproved;
        gov_tool.deterministic = true;
        gov_tool.side_effects = true;
        gov_tool.requires_generation = false;
        gov_tool.timeout_ms = 60000;
        if (!bridge_.register_tool(gov_tool, &why)) {
            throw std::runtime_error("core gov tool registration failed: " + why);
        }
        routes_.add({"hermes", /*baseline=*/true});
        routes_.add({"lucifer", /*baseline=*/false});
    }

    static constexpr const char* default_tool_ = "bridge.dispatch";
    static constexpr const char* kGovernanceTool = "governance.action";

    omni_core::BridgeConfig config_;   // declared first: bridge_ copies it
    omni_core::OmniBridge bridge_;
    omni_core::RouteTable routes_;
    bool bridge_lucifer_enabled_ = true;
};

}  // namespace omni::bridge
