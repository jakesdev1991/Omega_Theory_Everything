// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// Behavior test for CorePolicyAdapter — proves the unified pipeline
// grants/holds exactly what the portable core guarantees, and that the
// daemon's request model maps correctly onto the core's Proposal model.

#include "omni_bridge/CorePolicyAdapter.hpp"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    using omni::bridge::CorePolicyAdapter;
    namespace c = omni_core;

    CorePolicyAdapter adapter(omni::bridge::kDaemonSystemCaps,
                              omni::bridge::kDaemonSystemCaps);

    // 1. Normal trusted request -> Authorized with automated caps only.
    auto d1 = adapter.authorize("t1", "", "TRUSTED", "smoke_test",
                                /*injected=*/false, /*child=*/false);
    assert(d1.status == c::Decision::Status::Authorized);
    assert(d1.reason == "authorized");
    assert((d1.granted_caps & c::kGovernanceOnlyCaps) == c::kCapNone);
    assert(d1.granted_caps == omni::bridge::kDispatchCaps);
    std::cout << "[OK] trusted request authorized via core; granted="
              << d1.granted_caps << " tier=" << static_cast<int>(d1.tier)
              << "\n";

    // 2. Injection -> High risk -> the risk mask strips NetDial; the
    //    request still asks for it -> APPA denial. Fail closed.
    auto d2 = adapter.authorize("t2", "", "TRUSTED", "web",
                                /*injected=*/true, /*child=*/false);
    std::cout << "     injected: status=" << static_cast<int>(d2.status)
              << " reason=" << d2.reason << "\n";
    assert(d2.status == c::Decision::Status::Rejected);
    assert(d2.granted_caps == c::kCapNone);

    // 3. Child branch (lucifer exploration): authorized but never with
    //    governance caps; write dropped from the context set.
    auto d3 = adapter.authorize("t3", "", "UNTRUSTED", "raw_socket",
                                /*injected=*/false, /*child=*/true,
                                c::Risk::Negligible);
    std::cout << "     child: status=" << static_cast<int>(d3.status)
              << " reason=" << d3.reason << " granted=" << d3.granted_caps
              << "\n";
    assert((d3.granted_caps & c::kGovernanceOnlyCaps) == c::kCapNone);
    assert((d3.granted_caps & c::kCapFsWrite) == c::kCapNone);

    // 4. The automated set structurally excludes governance-only caps.
    static_assert((omni::bridge::kDispatchCaps & c::kGovernanceOnlyCaps)
                  == c::kCapNone,
                  "dispatch caps must never include governance authority");
    static_assert((omni::bridge::kDaemonSystemCaps & c::kGovernanceOnlyCaps)
                  == c::kCapNone,
                  "system caps must never include governance authority");
    std::cout << "[OK] static guarantees hold (no governance caps in the "
                 "automated sets)\n";

    // 5. Governance resolution via the human gate: the ONLY path that
    //    can convert a governance-only stall into authorization.
    c::HumanGovernor gov{"jake"};
    auto g = adapter.resolve_governance("t5", gov, "policy-change");
    std::cout << "     governance: status=" << static_cast<int>(g.status)
              << " reason=" << g.reason << " granted=" << g.granted_caps
              << " verification_required=" << g.verification_required << "\n";
    assert(g.status == c::Decision::Status::Authorized);
    assert(g.verification_required);   // human path is still slow path
    assert(g.tier == c::SandboxTier::T4_HumanApproved);

    // 6. Route stability (§57.6/7): 3 failures pin the baseline.
    for (int i = 0; i < 3; ++i) adapter.record_route_failure("lucifer");
    assert(adapter.route_unstable());
    assert(adapter.select_route("lucifer") == "hermes");
    adapter.record_route_success("lucifer");
    assert(!adapter.route_unstable());
    std::cout << "[OK] route instability pins baseline\n";

    // 7. Ledger healthy; core chain verifies; governance events recorded.
    assert(adapter.ledger_healthy());
    assert(adapter.core().ledger().verify_chain());
    std::cout << "[OK] core ledger chain verifies\n";

    // 8. Skill lifecycle: the FSM is enforced through the adapter —
    //    walk the real path then governed promotion.
    c::Sha256::Digest digest{};
    auto sid = adapter.register_skill(digest);
    std::string why;
    assert(adapter.transition_skill(sid, c::SkillState::Candidate, &why));
    assert(adapter.transition_skill(sid, c::SkillState::Testing, &why));
    assert(adapter.transition_skill(sid, c::SkillState::Audited, &why));
    assert(adapter.transition_skill(sid, c::SkillState::Shadow, &why));
    assert(adapter.transition_skill(sid, c::SkillState::Canary, &why));
    assert(adapter.promote_skill(sid, gov, &why));   // governed step
    assert(adapter.skill_eligible(sid));
    std::cout << "[OK] governed skill promotion via full FSM path\n";

    std::cout << "CorePolicyAdapter: ALL CHECKS PASSED\n";
    return 0;
}
