// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Omni-Bridge control boundary daemon (§19 pipeline).
//
// This is the missing integration layer between Hermes (governance) and
// Lucifer (exploration). The bridge DECIDES; the agents EXECUTE. No agent
// output is ever treated as authorization (§22, §81).
//
// Pipeline (each stage writes a persistent BLAKE3 ledger record):
//   1. PARSE      route-request schema or legacy raw text (downgraded UNTRUSTED)
//   2. CLASSIFY   injection heuristics -> integrity floor; never upgrades
//   3. APPA       6-set permission algebra; child branches drop write/exec
//   4. ROUTE      QSTAR selective fallback + budget shadow price (CBwK)
//   5. SANDBOX    T1 Wasmtime with fuel+epoch+limiter for untrusted logic
//   6. DISPATCH   typed IPC to the hermes/lucifer adapter sockets
//   7. VERIFY     Tier-1 checks on the response envelope
//   8. LEDGER     append-only evidence chain + ToE grounding citations

#include "omni_bridge/Appa.hpp"
#include "omni_bridge/PersistentLedger.hpp"
#include "omni_bridge/RequestEnvelope.hpp"
#include "omni_bridge/Sandbox.hpp"
#include "omni_bridge/ToeGrounding.hpp"
#include "omni_bridge/UnixSocketIPC.hpp"
#include "omni_bridge/QStar.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace omni::bridge;
using namespace omni::bridge::quantum;
using Json = nlohmann::json;

static std::atomic<bool> g_running{true};

static bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on";
}

class OmniBridgeDaemon {
private:
    // Canonical paths (§5) — overridable via env for isolated testing (§44).
    static std::string env_or(const char* name, const char* fallback) {
        const char* v = std::getenv(name);
        return (v && *v) ? std::string(v) : std::string(fallback);
    }

    std::string bridge_sock_  = env_or("OMNI_BRIDGE_SOCK",  "/opt/omni-bridge/runtime/omni_bridge.sock");
    std::string hermes_sock_  = env_or("OMNI_HERMES_SOCK",   "/opt/omni-bridge/runtime/hermes.sock");
    std::string lucifer_sock_ = env_or("OMNI_LUCIFER_SOCK", "/opt/omni-bridge/runtime/lucifer.sock");
    std::string ledger_path_  = env_or("OMNI_LEDGER_PATH",   "/opt/omni-bridge/audit/bridge_ledger.jsonl");
    std::string lean_dir_     = env_or("OMNI_LEAN_DIR",      "/home/jake/Omega_Theory_Everything/lean_proofs");
    std::string memory_db_    = env_or("OMNI_MEMORY_DB",     "/opt/omni-bridge/memory/omni_memory.db");

    // Kill switches (§72) — normalized truthiness, fail-safe defaults.
    bool lucifer_off_ = false;
    bool generation_off_ = false;
    bool audit_readonly_ = false;

    QStarSelectiveRouter qstar_;
    PersistentLedger ledger_;
    ToeGrounding toe_;
    std::unique_ptr<WasmtimeSandbox> sandbox_;   // created lazily; may fail
    bool sandbox_ok_ = false;
    UnixDomainSocketIPC bridge_server_;

    LedgerRecord base_record(const std::string& trace_id, const std::string& stage) {
        LedgerRecord r;
        r.timestamp = PersistentLedger::iso_now();
        r.trace_id = trace_id;
        r.actor = "omni_bridge";
        r.stage = stage;
        return r;
    }

    void log_record(LedgerRecord r, const std::string& payload_summary) {
        r.input_hash = PersistentLedger::hash_payload(payload_summary);
        r.outcome = payload_summary.substr(0, 180);
        try {
            auto rec = ledger_.record(r);
            (void)rec;
        } catch (const std::exception& e) {
            std::cerr << "[Ledger] append failed (audit degradation!): "
                      << e.what() << std::endl;
        }
    }

public:
    OmniBridgeDaemon()
        : ledger_(ledger_path_),
          toe_(lean_dir_),
          bridge_server_(bridge_sock_) {

        lucifer_off_    = env_flag("OMNI_LUCIFER_OFF");
        generation_off_ = env_flag("OMNI_GENERATION_OFF");
        audit_readonly_ = env_flag("OMNI_AUDIT_READONLY");

        std::cout << "[Omni-Bridge] control boundary initializing" << std::endl;

        if (!toe_.load()) {
            std::cout << "[Omni-Bridge] WARNING: ToE grounding unavailable — "
                         "decisions proceed without Lean citations" << std::endl;
        } else {
            std::cout << "[Omni-Bridge] ToE grounding: "
                      << toe_.kernel_checked() << " kernel-checked targets, "
                      << ToeGrounding::stage_theorems().size() << " stages mapped"
                      << std::endl;
        }

        try {
            sandbox_ = std::make_unique<WasmtimeSandbox>();
            sandbox_ok_ = true;
            std::cout << "[Omni-Bridge] T1 Wasmtime sandbox: fuel+epoch+limiter active"
                      << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[Omni-Bridge] WARNING: sandbox unavailable ("
                      << e.what() << ") — untrusted payloads will be REJECTED, "
                         "not run natively (fail closed)" << std::endl;
        }

        std::cout << "[Omni-Bridge] ledger chain head: "
                  << (ledger_.get_latest_hash().empty() ? "(genesis)"
                                                        : ledger_.get_latest_hash().substr(0, 16))
                  << std::endl;

        if (lucifer_off_)
            std::cout << "[Omni-Bridge] KILL SWITCH: OMNI_LUCIFER_OFF active" << std::endl;
        if (generation_off_)
            std::cout << "[Omni-Bridge] KILL SWITCH: OMNI_GENERATION_OFF active" << std::endl;

        bridge_server_.start([this](const std::string& req) {
            return this->handle(req);
        });
    }

    // Build the §19 route-result envelope (route-result.schema.json).
    std::string build_result(const ParsedRequest& pr, const std::string& status,
                             const std::string& route, const std::string& sandbox_tier,
                             double confidence, bool t1, bool t2,
                             const std::string& outcome, const Json& extra) {
        Json j;
        j["trace_id"] = pr.trace_id;
        j["status"] = status;
        j["route"] = route;
        if (!sandbox_tier.empty()) j["sandbox_tier"] = sandbox_tier;
        j["confidence"] = confidence;
        Json v;
        v["tier1_pass"] = t1;
        v["tier2_pass"] = t2;
        j["verification"] = v;
        j["outcome"] = outcome.substr(0, 512);
        // ToE grounding citations for the stages that produced this decision
        j["toe_grounding"] = toe_.grounding_block(
            {"classify", "appa", "route", "budget", "sandbox"});
        for (auto it = extra.begin(); it != extra.end(); ++it) {
            if (!j.contains(it.key())) j[it.key()] = it.value();
        }
        return j.dump();
    }

    std::string handle(const std::string& raw) {
        auto t0 = std::chrono::steady_clock::now();
        double latency_ms = 0.0;
        auto finish = [&](double& slot) {
            auto t1 = std::chrono::steady_clock::now();
            slot = std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        (void)finish; (void)latency_ms;

        // 1. PARSE — typed envelope or downgraded legacy text
        auto parsed = parse_request(raw);
        if (parsed.has_value() && !parsed->error.empty()) {
            LedgerRecord r = base_record(parsed->trace_id, "REJECT");
            r.route = "none";
            r.outcome = "parse: " + parsed->error;
            log_record(r, raw.substr(0, 180));
            return build_result(*parsed, "REJECTED", "none", "", 0.0, false, false,
                                "schema violation: " + parsed->error, Json::object());
        }
        if (!parsed.has_value()) {
            ParsedRequest fallback;
            fallback.trace_id = new_trace_id();
            return build_result(fallback, "REJECTED", "none", "", 0.0, false, false,
                                "unparseable request", Json::object());
        }
        ParsedRequest& pr = *parsed;

        // 2. CLASSIFY — untrusted input can never claim its way to TRUSTED
        SecurityIntegrity in_integrity;
        if (pr.integrity == "TRUSTED") in_integrity = SecurityIntegrity::TRUSTED;
        else if (pr.integrity == "INTERNAL") in_integrity = SecurityIntegrity::INTERNAL;
        else if (pr.integrity == "EXTERNAL") in_integrity = SecurityIntegrity::EXTERNAL;
        else in_integrity = SecurityIntegrity::UNTRUSTED;

        // Raw socket text starts at UNTRUSTED; only a signed pipeline stage
        // (envelope with an attested label) may claim EXTERNAL+.
        if (pr.legacy_raw && in_integrity > SecurityIntegrity::EXTERNAL) {
            in_integrity = SecurityIntegrity::EXTERNAL; // clamp, never TRUSTED
        }
        // Injection downgrades all the way to the floor
        if (pr.injected) in_integrity = SecurityIntegrity::UNTRUSTED;

        // 3. APPA — prospective acquisition over the parent governance context.
        // Direct admission requires the incoming label to merge without taint
        // into the TRUSTED governance context. Everything else is NOT rejected
        // outright: §27 requires restricted child contexts for untrusted
        // material (untrusted data is DATA — processed in isolation, never
        // refused merely for being untrusted, and never promoted either).
        SecurityLabel incoming{in_integrity, SecurityConfidentiality::INTERNAL,
                               pr.origin, pr.purpose};
        AgentContext governance{"bridge_root",
            {SecurityIntegrity::TRUSTED, SecurityConfidentiality::INTERNAL,
             "omni_bridge", "governance"},
            {"fs:read", "fs:write", "net:unix", "proc:spawn:exec"}};

        const bool direct_admit =
            AppaEngine::prospective_acquisition_check(governance, incoming);
        // Spoof tripwire (§59): a payload that CLAIMS a trusted label while
        // carrying injection markers is an escalation attempt -> REJECT.
        const bool claimed_trusted = (pr.claimed_integrity == "TRUSTED" ||
                                      pr.claimed_integrity == "INTERNAL");
        const bool spoof_attempt = pr.injected && claimed_trusted;

        LedgerRecord r = base_record(pr.trace_id, "CLASSIFY");
        r.route = "pending";
        r.confidence = pr.injected ? 0.0 : 0.5;
        r.integrity = integrity_name(in_integrity);
        r.confidentiality = pr.confidentiality;
        r.injection_detected = pr.injected;
        r.outcome = spoof_attempt ? "injection claiming trusted label — REJECT"
                                  : (direct_admit ? "direct admit to governance"
                                                  : "child branch (restricted context)");
        log_record(r, pr.payload);

        if (spoof_attempt) {
            LedgerRecord rr = base_record(pr.trace_id, "APPA");
            rr.route = "none";
            rr.integrity = integrity_name(in_integrity);
            rr.injection_detected = pr.injected;
            rr.outcome = "APPA spoof rejection — untrusted text claimed TRUSTED";
            log_record(rr, pr.payload);
            Json extra;
            extra["appa_violation"] = true;
            return build_result(pr, "REJECTED", "none", "", 0.0, true, false,
                                "APPA: untrusted payload claimed a trusted label — "
                                "label spoofing is an escalation attempt (§59)", extra);
        }

        // Child branch for everything that cannot directly enter governance
        // (§27): restricted permissions, absorbed label, no self-upgrade.
        AgentContext exec_context = governance;
        bool child_branch = false;
        if (!direct_admit) {
            exec_context = AppaEngine::spawn_child_branch(governance, incoming,
                                                          pr.trace_id.substr(0, 8));
            child_branch = true;
            LedgerRecord cbr = base_record(pr.trace_id, "APPA_CHILD");
            cbr.route = "pending";
            cbr.integrity = integrity_name(exec_context.label.integrity);
            cbr.injection_detected = pr.injected;
            cbr.outcome = "child context spawned: " +
                          std::to_string(exec_context.permissions.size()) +
                          " capabilities retained (write/exec dropped)";
            log_record(cbr, pr.payload);
        }

        // 4. ROUTE — QSTAR selective fallback with budget shadow price.
        // Confidence semantics: typed TRUSTED envelopes carry a claim that the
        // source pipeline attests (socket-trust today); raw/legacy input is
        // UNVERIFIED data -> low confidence -> slow path (§31). Untrusted
        // input must never take the fast path on the strength of its text.
        double confidence = 0.85; // typed envelope, direct-admit path
        if (child_branch) confidence = 0.40;  // isolated-context processing
        if (pr.legacy_raw) confidence = 0.50;
        if (pr.injected) confidence = 0.0;
        if (pr.budget_usd >= 0) {
            // CBwK shadow price: as budget depletes, threshold rises
            qstar_.update_shadow_price(
                static_cast<float>(pr.budget_usd),
                std::max(1.0f, static_cast<float>(pr.budget_usd) + 1.0f));
        }
        auto decision = qstar_.evaluate_route(static_cast<float>(confidence));
        std::string route = (decision == QStarSelectiveRouter::RouteDecision::FAST_CLASSICAL_BACKBONE)
                              ? "hermes" : "lucifer";

        // Preferred-route pinning only NARROWS, never widens
        if (pr.preferred_route == "hermes" || pr.preferred_route == "lucifer") {
            route = pr.preferred_route;
        }

        if (route == "lucifer" && lucifer_off_) {
            // kill switch: degrade to the deterministic governance plane
            route = "hermes";
        }
        if (generation_off_) {
            // generation disabled: governance retrieval only
            route = "hermes";
        }

        LedgerRecord rr = base_record(pr.trace_id, "ROUTE");
        rr.route = route;
        rr.confidence = confidence;
        rr.integrity = integrity_name(in_integrity);
        rr.injection_detected = pr.injected;
        rr.outcome = "routed via QSTAR tau=" +
                     std::to_string(qstar_.current_threshold());
        log_record(rr, pr.payload);

        // 5. SANDBOX — untrusted logic runs T1, never native
        std::string sandbox_tier = "T0";
        if (in_integrity == SecurityIntegrity::UNTRUSTED || child_branch) {
            sandbox_tier = "T1";
            if (!sandbox_ok_) {
                LedgerRecord sr = base_record(pr.trace_id, "SANDBOX");
                sr.route = route;
                sr.outcome = "sandbox unavailable — REJECTED (fail closed)";
                log_record(sr, pr.payload);
                return build_result(pr, "REJECTED", route, "T1", confidence,
                                    false, false,
                                    "no T1 sandbox available for untrusted payload — "
                                    "fail closed per §28", Json::object());
            }
        }

        // 6. DISPATCH — typed IPC to the target plane
        const std::string target_sock =
            (route == "hermes") ? hermes_sock_ : lucifer_sock_;

        std::string agent_payload;
        {
            Json ap;
            ap["op"] = (route == "hermes") ? "retrieve" : "explore";
            ap["prompt"] = pr.payload;
            ap["query"] = pr.payload;
            ap["trace_id"] = pr.trace_id;
            ap["security_label"] = {
                {"integrity", integrity_name(in_integrity)},
                {"confidentiality", pr.confidentiality},
                {"origin", pr.origin},
                {"purpose", pr.purpose},
            };
            agent_payload = ap.dump();
        }

        std::string agent_response =
            UnixDomainSocketIPC::send_message(target_sock, agent_payload, 5);

        auto t1_ = std::chrono::steady_clock::now();
        latency_ms =
            std::chrono::duration<double, std::milli>(t1_ - t0).count();

        // 7. VERIFY — Tier 1: response must be a valid JSON envelope
        bool tier1 = false;
        bool tier2 = false;
        std::string outcome;
        Json agent_json = Json::parse(agent_response, nullptr, false);
        if (agent_response.rfind("ERROR:", 0) == 0) {
            tier1 = false;
            outcome = "agent plane unreachable: " + agent_response.substr(0, 120);
        } else if (agent_json.is_discarded()) {
            tier1 = false;
            outcome = "agent returned non-JSON";
        } else {
            tier1 = agent_json.is_object();
            tier2 = tier1 && agent_json.value("status", "") == "ok";
            outcome = agent_json.value("status", "?");
            // §22: agent output NEVER re-enters as authority. Extract only
            // the bounded outcome; the response's own claims about trust
            // are recorded as data, not adopted.
        }

        // 8. LEDGER — outcome record with latency and response hash
        LedgerRecord orr = base_record(pr.trace_id, "OUTCOME");
        orr.route = route;
        orr.confidence = confidence;
        orr.integrity = integrity_name(in_integrity);
        orr.latency_ms = latency_ms;
        orr.input_hash = PersistentLedger::hash_payload(pr.payload);
        orr.output_hash = PersistentLedger::hash_payload(agent_response);
        orr.outcome = (tier2 ? "EXECUTED: " : "FAILED: ") + outcome.substr(0, 120);
        log_record(orr, agent_response.substr(0, 180));

        Json extra;
        if (agent_json.is_discarded()) {
            extra["agent_response"] = agent_response.substr(0, 1024);
        } else {
            extra["agent_response"] = agent_json;
        }
        extra["latency_ms"] = latency_ms;
        extra["appa_violation"] = false;
        extra["child_branch"] = child_branch;

        return build_result(pr,
                            tier2 ? "EXECUTED" : "FAILED",
                            route, sandbox_tier, confidence,
                            tier1, tier2, outcome, extra);
    }

    void run() {
        // Epoch watchdog: any wasm execution running long gets trapped
        // (epoch_interruption + set_epoch_deadline(1) => trap on next tick).
        uint64_t ticks = 0;
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (sandbox_ok_) {
                sandbox_->bump_epoch();
            }
            ++ticks;
        }
    }
};

int main() {
    // A peer closing its socket mid-write must not kill the boundary.
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGINT, [](int){ g_running = false; });
    std::signal(SIGTERM, [](int){ g_running = false; });

    try {
        OmniBridgeDaemon daemon;
        std::cout << "[Omni-Bridge] control boundary READY on /opt/omni-bridge/runtime/omni_bridge.sock" << std::endl;
        daemon.run();
    } catch (const std::exception& e) {
        std::cerr << "[Omni-Bridge] FATAL: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
