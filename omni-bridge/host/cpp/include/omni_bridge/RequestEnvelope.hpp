// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

// Typed request envelope + Tier-1 schema validation (§29, §53).
//
// The bridge socket speaks the route-request.schema.json contract:
//   { trace_id, payload, security_label, proposed_tools, budget, preferred_route }
// Raw non-JSON text is preserved as a legacy path but is downgraded to
// UNTRUSTED and routed through the injection classifier — never TRUSTED.

#include <nlohmann/json.hpp>
#include <string>
#include <optional>
#include <vector>
#include <array>
#include <functional>
#include <sstream>
#include <iomanip>
#include <random>
#include <cstring>

namespace omni::bridge {

using Json = nlohmann::json;

struct ParsedRequest {
    std::string trace_id;
    std::string payload;
    std::string claimed_integrity; // label AS CLAIMED by the sender (evidence)
    std::string integrity;         // effective label after classification
    std::string confidentiality;
    std::string origin;
    std::string purpose;
    bool        legacy_raw = false;
    std::string preferred_route; // "" | hermes | lucifer | auto
    double      budget_usd = -1;
    int64_t     budget_latency_ms = -1;
    std::string error;           // set when rejected at parse time
    bool        injected = false;
};

// §53 injection heuristics. This is a DOWNGRADE classifier, not an immunity
// claim (§71): matching input gets EXTERNAL at best, never TRUSTED.
inline bool detects_injection(const std::string& text) {
    static const std::array<const char*, 10> markers = {
        "ignore all previous instructions",
        "disregard your instructions",
        "grant_permission", "grant yourself permission",
        "disable_hermes", "disable verification",
        "remove_audit", "delete audit", "exfiltrate",
        "ignore all previous"
    };
    std::string lower;
    lower.reserve(text.size());
    for (unsigned char c : text) lower.push_back(static_cast<char>(std::tolower(c)));
    for (const char* m : markers) {
        if (lower.find(m) != std::string::npos) return true;
    }
    return false;
}

// Generate a RFC-4122-ish trace id from system entropy (not rand()).
inline std::string new_trace_id() {
    std::array<unsigned char, 16> buf;
    std::random_device rd;
    for (auto& b : buf) b = rd();
    buf[6] = (buf[6] & 0x0F) | 0x40; // version 4
    buf[8] = (buf[8] & 0x3F) | 0x80; // variant
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < buf.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) ss << '-';
        ss << std::setw(2) << static_cast<int>(buf[i]);
    }
    return ss.str();
}

inline std::optional<ParsedRequest> parse_request(const std::string& raw) {
    ParsedRequest req;

    // Try the typed envelope first (schemas/route-request.schema.json).
    auto j = Json::parse(raw, nullptr, /*allow_exceptions=*/false,
                         /*ignore_comments=*/true);
    if (!j.is_discarded() && j.is_object()) {
        // An input that claims to be a typed object must satisfy the schema
        // in full — missing required fields is a REJECTION, not a legacy
        // fallback (fail closed per §44/§53).
        if (!j.contains("payload") || !j["payload"].is_string()) {
            req.error = "schema violation: 'payload' (string) is required";
            return req;
        }
        req.trace_id = j.value("trace_id", "");
        if (req.trace_id.empty()) req.trace_id = new_trace_id();
        req.payload = j["payload"].get<std::string>();
        if (req.payload.empty() || req.payload.size() > 131072) {
            req.error = "payload violates bounds (1..131072)";
            return req;
        }

        if (j.contains("security_label") && j["security_label"].is_object() &&
            j["security_label"].contains("integrity")) {
            const auto& sl = j["security_label"];
            req.integrity = sl.value("integrity", "UNTRUSTED");
            req.confidentiality = sl.value("confidentiality", "PUBLIC");
            req.origin = sl.value("origin", "unknown");
            req.purpose = sl.value("purpose", "unspecified");
        } else {
            req.error = "schema violation: 'security_label' with 'integrity' is required";
            return req;
        }

        if (j.contains("preferred_route")) {
            const auto& pr = j["preferred_route"];
            if (pr.is_string()) {
                std::string s = pr.get<std::string>();
                if (s == "hermes" || s == "lucifer" || s == "auto")
                    req.preferred_route = s;
            }
        }
        if (j.contains("budget") && j["budget"].is_object()) {
            const auto& b = j["budget"];
            if (b.contains("max_usd") && b["max_usd"].is_number())
                req.budget_usd = b["max_usd"].get<double>();
            if (b.contains("max_latency_ms") && b["max_latency_ms"].is_number())
                req.budget_latency_ms = b["max_latency_ms"].get<int64_t>();
        }

        // CLAIMED integrity is never trusted: the classifier re-derives it.
        req.claimed_integrity = req.integrity; // preserve spoof evidence
        req.injected = detects_injection(req.payload);
        if (req.injected) {
            // Downgrade regardless of the claimed label (§26: untrusted text
            // cannot gain privilege by declaring itself trusted).
            req.integrity = "UNTRUSTED";
        }
        return req;
    }

    // Legacy raw-text path: untyped input is data, not authority.
    req.legacy_raw = true;
    req.trace_id = new_trace_id();
    req.payload = raw;
    if (raw.empty() || raw.size() > 131072) {
        req.error = "payload violates bounds";
        return req;
    }
    req.claimed_integrity = "NONE"; // raw text claims nothing
    req.integrity = "UNTRUSTED";
    req.confidentiality = "PUBLIC";
    req.origin = "raw_socket";
    req.purpose = "legacy_text";
    req.injected = detects_injection(req.payload);
    return req;
}

} // namespace omni::bridge
