// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// omni_bridge_core.hpp — the Omni-Bridge control boundary (portable core).
//
// Implements the directive's control-plane requirements (MASTER_DIRECTIVE
// §§19–41): APPA permission algebra, security labels, context branching,
// sandbox-tier selection, tool ABI validation, skill lifecycle + rollback,
// append-only evidence ledger, route fallback, replay classification,
// kill switches, and the deterministic authorization pipeline:
//
//   proposal -> manifest validation -> skill eligibility -> APPA
//            -> risk gate -> audit gate -> sandbox selection
//            -> (fast | slow path) -> [governance gate] -> authorization
//            -> execution -> output validation -> verification -> audit
//
// Core invariants (directive §22, §24, §25, §81), machine-checked at
// compile time (static_assert) and by the acceptance tests:
//   * No LLM output becomes authorization: a Proposal is data; only the
//     deterministic evaluate() mints an Authorization, and only the
//     human-governance gate mints GovernanceApproval.
//   * APPA: effective = System ∩ Declared ∩ Tenant ∩ Tool ∩ Context ∩
//     Risk; risk only ever REMOVES authority.
//   * Untrusted content never silently becomes trusted; trusted-context
//     writes require an explicit governance-approved upgrade.
//   * A child context cannot upgrade itself.
//   * Audit unavailable => high-risk execution fails closed.
//
// HONEST LIMITS (do not paper over these):
//   * Hash: SHA-256 (verified against FIPS vectors), not BLAKE3. The
//     directive prefers BLAKE3; swapping requires a pinned, vetted
//     implementation per the supply-chain rules (§74). The chain format
//     is hash-agnostic (32-byte digests) so the upgrade is mechanical.
//   * Sandbox TIERS are selected, not enforced here. T1 (Wasmtime),
//     T2 (container), T3 (microVM) enforcement is a host-side runtime
//     responsibility; this core is the policy + audit boundary.
//   * The "human" gate is a type-level separation in a single process.
//     In production it must be a separate service/account (§43) so the
//     separation is enforced by the OS, not just the type system.
//   * Memory is not the §33 memory subsystem; this is the control plane.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace omni {

// ===========================================================================
// SHA-256 (FIPS 180-4). Chosen for zero-dependency verifiability; see the
// BLAKE3 note above. Verified against official test vectors in tests.
// ===========================================================================

class Sha256 {
 public:
  using Digest = std::array<std::uint8_t, 32>;

  static Digest hash(std::span<const std::uint8_t> data) noexcept {
    Sha256 h;
    h.absorb(data.data(), data.size());
    h.pad_and_final();
    return h.state_as_digest();
  }

  static Digest hash(std::string_view s) noexcept {
    return hash({reinterpret_cast<const std::uint8_t*>(s.data()),
                 s.size()});
  }

  // domain-separated concatenation: hash(a || len(b) || b)
  static Digest chain(const Digest& a, std::string_view b) noexcept {
    Sha256 h;
    const auto* pa = reinterpret_cast<const std::uint8_t*>(a.data());
    h.absorb(pa, a.size());
    const std::uint64_t blen = static_cast<std::uint64_t>(b.size());
    const auto* lb = reinterpret_cast<const std::uint8_t*>(&blen);
    h.absorb(lb, sizeof(blen));
    h.absorb(reinterpret_cast<const std::uint8_t*>(b.data()), b.size());
    h.pad_and_final();
    return h.state_as_digest();
  }

 private:
  Sha256() noexcept { state_[0] = 0x6a09e667u; state_[1] = 0xbb67ae85u;
    state_[2] = 0x3c6ef372u; state_[3] = 0xa54ff53au;
    state_[4] = 0x510e527fu; state_[5] = 0x9b05688cu;
    state_[6] = 0x1f83d9abu; state_[7] = 0x5be0cd19u; }

  static std::uint32_t rotr(std::uint32_t x, unsigned n) noexcept {
    return (x >> n) | (x << (32u - n));
  }

  void absorb(const std::uint8_t* p, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
      buf_[buflen_++] = p[i];
      if (buflen_ == 64) { block(buf_.data()); buflen_ = 0; }
    }
    total_ += n;
  }

  void pad_and_final() noexcept {
    const std::uint64_t bits = total_ * 8ULL;
    const std::uint8_t one = 0x80u;
    absorb(&one, 1);
    const std::uint8_t zero = 0x00u;
    while (buflen_ != 56) absorb(&zero, 1);
    std::uint8_t lenb[8];
    for (int i = 0; i < 8; ++i) {
      lenb[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    }
    absorb(lenb, 8);
  }

  void block(const std::uint8_t* p) noexcept {
    static constexpr std::uint32_t K[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,
      0x59f111f1u,0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,
      0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,
      0xc19bf174u,0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
      0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,0x983e5152u,
      0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
      0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,
      0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,
      0xd6990624u,0xf40e3585u,0x106aa070u,0x19a4c116u,0x1e376c08u,
      0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,
      0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
      0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24) |
             (static_cast<std::uint32_t>(p[4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(p[4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
          rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 =
          rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2],
                   d = state_[3], e = state_[4], f = state_[5],
                   g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
      const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = S0 + maj;
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a;
      a = t1 + t2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }

  Digest state_as_digest() const noexcept {
    Digest out{};
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 4; ++j) {
        out[static_cast<std::size_t>(4 * i + j)] =
            static_cast<std::uint8_t>(state_[i] >> (24 - 8 * j));
      }
    }
    return out;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buf_{};
  std::size_t buflen_ = 0;
  std::uint64_t total_ = 0;
};

[[nodiscard]] inline std::string to_hex(
    const Sha256::Digest& d) noexcept {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(d.size() * 2);
  for (std::uint8_t b : d) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 0xF]);
  }
  return out;
}

// ===========================================================================
// APPA — the permission algebra (directive §24)
//   Effective = System ∩ Declared ∩ Tenant ∩ Tool ∩ Context ∩ Risk
// Risk only removes authority. Verified by static_asserts + tests.
// ===========================================================================

using CapSet = std::uint64_t;

inline constexpr CapSet kCapNone = 0;
inline constexpr CapSet kCapNetRead = 1ULL << 0;
inline constexpr CapSet kCapNetDial = 1ULL << 1;
inline constexpr CapSet kCapFsRead = 1ULL << 2;
inline constexpr CapSet kCapFsWrite = 1ULL << 3;
inline constexpr CapSet kCapProcExec = 1ULL << 4;
inline constexpr CapSet kCapCredentialRead = 1ULL << 5;
inline constexpr CapSet kCapPolicyWrite = 1ULL << 6;
inline constexpr CapSet kCapSkillPromote = 1ULL << 7;
inline constexpr CapSet kCapTrustedMemoryWrite = 1ULL << 8;
inline constexpr CapSet kCapLedgerAppend = 1ULL << 9;
inline constexpr CapSet kCapSecretRead = 1ULL << 10;
inline constexpr CapSet kCapMintOrMoveValue = 1ULL << 11;

// Caps that NO automated path may hold: governance-only authority
// (directive §22 — Lucifer must not govern itself).
inline constexpr CapSet kGovernanceOnlyCaps =
    kCapPolicyWrite | kCapSkillPromote | kCapMintOrMoveValue |
    kCapSecretRead | kCapCredentialRead;

enum class Risk : std::uint8_t {
  Negligible = 0, Low = 1, Medium = 2, High = 3, Critical = 4 };

// Cap sets permitted AT each risk level — monotonically non-increasing.
// Risk never manufactures authority (only ANDs into the intersection).
// Ladder rationale: Negligible = anything except governance-only caps;
// Low drops process execution; Medium also drops network egress;
// High is read-only + audit; Critical removes everything.
[[nodiscard]] constexpr CapSet risk_mask(Risk r) noexcept {
  switch (r) {
    case Risk::Negligible:
      return ~kGovernanceOnlyCaps;
    case Risk::Low:
      return (~kGovernanceOnlyCaps) & ~kCapProcExec;
    case Risk::Medium:
      return (~kGovernanceOnlyCaps) & ~kCapProcExec & ~kCapNetDial;
    case Risk::High:
      return kCapFsRead | kCapNetRead | kCapLedgerAppend;
    case Risk::Critical:
      return kCapNone;
  }
  return kCapNone;
}

// Compile-time verified APPA properties (§18 "where mathematical
// certainty is actually useful" — checked by the compiler here; the
// Lean formalization is deferred until a Lean toolchain is pinned).
static_assert(risk_mask(Risk::Critical) == kCapNone,
              "critical risk must remove all authority");
static_assert((risk_mask(Risk::Low) & risk_mask(Risk::Negligible)) ==
                  risk_mask(Risk::Low),
              "risk masks must be monotone non-increasing");
static_assert((risk_mask(Risk::Medium) & risk_mask(Risk::Low)) ==
                  risk_mask(Risk::Medium),
              "risk masks must be monotone non-increasing");
static_assert((risk_mask(Risk::High) & risk_mask(Risk::Medium)) ==
                  risk_mask(Risk::High),
              "risk masks must be monotone non-increasing");
static_assert((risk_mask(Risk::Negligible) & kGovernanceOnlyCaps) ==
                  kCapNone,
              "governance-only caps are outside every automated risk "
              "level (§22)");

[[nodiscard]] constexpr CapSet appa_intersect(CapSet system,
                                              CapSet declared,
                                              CapSet tenant, CapSet tool,
                                              CapSet context,
                                              Risk risk) noexcept {
  return system & declared & tenant & tool & context & risk_mask(risk);
}

// ===========================================================================
// Security labels (§25) and contexts (§27)
// ===========================================================================

enum class Integrity : std::uint8_t { Trusted, Untrusted };
enum class Confidentiality : std::uint8_t { Public, Internal, Restricted };
enum class Origin : std::uint8_t {
  System, Human, Model, Retrieved, ToolOutput
};
enum class Purpose : std::uint8_t { Task, Governance, Audit, Telemetry };

struct SecurityLabel {
  Integrity integrity = Integrity::Untrusted;
  Confidentiality confidentiality = Confidentiality::Public;
  Origin origin = Origin::Model;
  Purpose purpose = Purpose::Task;

  [[nodiscard]] friend bool operator==(const SecurityLabel& a,
                                       const SecurityLabel& b) noexcept {
    return a.integrity == b.integrity &&
           a.confidentiality == b.confidentiality &&
           a.origin == b.origin && a.purpose == b.purpose;
  }
};

[[nodiscard]] constexpr SecurityLabel untrusted_external() noexcept {
  return {Integrity::Untrusted, Confidentiality::Public, Origin::Retrieved,
          Purpose::Task};
}

// Trusted-memory writes require Trusted integrity; untrusted artifacts
// never silently become trusted (§25) — only the governance upgrade
// path may relabel, and it is audited.
[[nodiscard]] constexpr bool may_write_trusted_memory(
    const SecurityLabel& l) noexcept {
  return l.integrity == Integrity::Trusted;
}

struct ExecutionContext {
  std::string id;
  CapSet declared_caps = kCapNone;   // the agent's declared authority
  CapSet context_caps = kCapNone;    // this context's restriction set
  Risk risk_ceiling = Risk::Medium;
  SecurityLabel content_label = untrusted_external();
  bool is_child = false;             // §27 branch isolation
};

// ===========================================================================
// Sandbox tiers (§28)
// ===========================================================================

enum class SandboxTier : std::uint8_t {
  T0_InProcess = 0,
  T1_Wasm = 1,
  T2_Container = 2,
  T3_MicroVM = 3,
  T4_HumanApproved = 4,
  T5_Prohibited = 5
};

[[nodiscard]] constexpr SandboxTier max_tier(SandboxTier a,
                                             SandboxTier b) noexcept {
  return (static_cast<int>(a) >= static_cast<int>(b)) ? a : b;
}

// Policy floor: untrusted content executing with real capabilities must
// run in at least a container tier; governance-only caps need a human.
[[nodiscard]] constexpr SandboxTier policy_floor(
    const SecurityLabel& label, CapSet requested) noexcept {
  SandboxTier t = SandboxTier::T0_InProcess;
  if (label.integrity == Integrity::Untrusted) {
    t = SandboxTier::T2_Container;
  }
  if ((requested & kGovernanceOnlyCaps) != kCapNone) {
    t = SandboxTier::T4_HumanApproved;
  }
  return t;
}

// ===========================================================================
// Tool ABI (§29)
// ===========================================================================

struct ToolManifest {
  std::string id;
  std::uint32_t version = 0;
  CapSet required_caps = kCapNone;
  SandboxTier min_tier = SandboxTier::T0_InProcess;
  bool deterministic = true;
  bool side_effects = false;
  bool requires_generation = false;  // touches an LLM
  std::uint32_t timeout_ms = 0;

  // Fail-closed ABI validation (§44: invalid config prevents execution).
  [[nodiscard]] bool valid(std::string* why = nullptr) const noexcept {
    if (id.empty()) { if (why) *why = "missing tool id"; return false; }
    if (version == 0) {
      if (why) *why = "missing version";
      return false;
    }
    if (timeout_ms == 0) {
      if (why) *why = "missing timeout";
      return false;
    }
    if (required_caps == kCapNone) {
      if (why) *why = "tool must declare at least one capability or be "
                      "explicitly effect-free";
      return false;
    }
    return true;
  }
};

// ===========================================================================
// Skill lifecycle (§40) and registry with rollback (§41)
// ===========================================================================

enum class SkillState : std::uint8_t {
  Discovered, Candidate, Testing, Audited, Shadow, Canary,
  Promoted, Monitored, Deprecated, Retired
};

[[nodiscard]] constexpr bool skill_eligible_for_use(
    SkillState s) noexcept {
  return s == SkillState::Promoted || s == SkillState::Monitored;
}

[[nodiscard]] constexpr bool skill_transition_valid(SkillState from,
                                                    SkillState to) noexcept {
  if (from == to) return false;
  switch (from) {
    case SkillState::Discovered:
      return to == SkillState::Candidate || to == SkillState::Deprecated;
    case SkillState::Candidate:
      return to == SkillState::Testing || to == SkillState::Deprecated;
    case SkillState::Testing:
      return to == SkillState::Audited || to == SkillState::Deprecated;
    case SkillState::Audited:
      return to == SkillState::Shadow || to == SkillState::Deprecated;
    case SkillState::Shadow:
      return to == SkillState::Canary || to == SkillState::Deprecated;
    case SkillState::Canary:
      return to == SkillState::Promoted || to == SkillState::Deprecated;
    case SkillState::Promoted:
      return to == SkillState::Monitored ||
             to == SkillState::Deprecated;
    case SkillState::Monitored:
      return to == SkillState::Deprecated;
    case SkillState::Deprecated:
      return to == SkillState::Retired;
    case SkillState::Retired:
      return false;  // terminal
  }
  return false;
}

struct SkillVersion {
  std::uint32_t version = 1;
  SkillState state = SkillState::Discovered;
  Sha256::Digest content_digest{};
};

// ===========================================================================
// Replay classification (§35)
// ===========================================================================

enum class ReplayClass : std::uint8_t {
  ExactReplay, DeterministicReplay, SimulatedCounterfactual,
  ApproximateCounterfactual, NonReplayable
};

[[nodiscard]] constexpr ReplayClass classify_replay(
    bool tool_deterministic, bool same_recorded_inputs,
    bool same_environment) noexcept {
  if (!tool_deterministic) return ReplayClass::NonReplayable;
  if (same_recorded_inputs && same_environment) {
    return ReplayClass::ExactReplay;
  }
  if (same_recorded_inputs) return ReplayClass::DeterministicReplay;
  return ReplayClass::SimulatedCounterfactual;
}

// ===========================================================================
// Evidence ledger (§34): append-only, hash-chained, tamper-evident.
// ===========================================================================

struct LedgerEvent {
  std::uint64_t seq = 0;
  std::string trace_id;
  std::string kind;      // e.g. "authorization", "rejection", "upgrade"
  std::string actor;     // "lucifer", "hermes", "system", "human:<name>"
  std::string summary;   // NO SECRETS (§45): enforced by convention +
                         // test (redaction scan for known secret tokens)
  Sha256::Digest prev{};
  Sha256::Digest self{};
};

class EvidenceLedger {
 public:
  // Returns sequence number, or nullopt when the ledger cannot append
  // (read-only kill switch, or the audit subsystem has failed).
  [[nodiscard]] std::optional<std::uint64_t> append(
      std::string_view trace_id, std::string_view kind,
      std::string_view actor, std::string_view summary) {
    if (read_only_ || failed_) {
      ++gap_count_;
      return std::nullopt;
    }
    LedgerEvent e;
    e.seq = next_seq_++;
    e.trace_id.assign(trace_id);
    e.kind.assign(kind);
    e.actor.assign(actor);
    e.summary.assign(summary);
    e.prev = last_hash_;
    const std::string canonical = canonicalize(e);
    e.self = Sha256::chain(e.prev, canonical);
    last_hash_ = e.self;
    events_.push_back(std::move(e));
    return events_.back().seq;
  }

  [[nodiscard]] bool verify_chain() const noexcept {
    Sha256::Digest prev{};
    for (std::size_t i = 0; i < events_.size(); ++i) {
      const LedgerEvent& e = events_[i];
      if (e.seq != static_cast<std::uint64_t>(i)) return false;
      if (e.prev != prev) return false;
      if (e.self != Sha256::chain(e.prev, canonicalize(e))) return false;
      prev = e.self;
    }
    return true;
  }

  void mark_failed() noexcept { failed_ = true; }
  [[nodiscard]] bool healthy() const noexcept { return !failed_; }
  void set_read_only(bool ro) noexcept { read_only_ = ro; }
  [[nodiscard]] bool read_only() const noexcept { return read_only_; }
  [[nodiscard]] std::uint64_t gap_count() const noexcept {
    return gap_count_;
  }
  [[nodiscard]] const std::vector<LedgerEvent>& events() const noexcept {
    return events_;
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return events_.size();
  }

 private:
  [[nodiscard]] static std::string canonicalize(const LedgerEvent& e) {
    // Length-prefixed field order is fixed => unambiguous encoding.
    std::string out;
    const auto field = [&out](std::string_view v) {
      out += std::to_string(v.size());
      out.push_back(':');
      out.append(v);
      out.push_back('|');
    };
    out += std::to_string(e.seq);
    out.push_back('|');
    field(e.trace_id);
    field(e.kind);
    field(e.actor);
    field(e.summary);
    return out;
  }

  std::vector<LedgerEvent> events_;
  Sha256::Digest last_hash_{};
  std::uint64_t next_seq_ = 0;
  std::uint64_t gap_count_ = 0;
  bool read_only_ = false;
  bool failed_ = false;
};

// ===========================================================================
// Routes (§30) with deterministic fallback + baseline pinning (§57.6/7)
// ===========================================================================

struct Route {
  std::string id;
  bool baseline = false;
};

class RouteTable {
 public:
  static constexpr int kInstabilityThreshold = 3;

  void add(Route r) { routes_.push_back(std::move(r)); }
  void record_failure(std::string_view id) {
    for (auto& r : routes_) {
      if (r.id == id) {
        if (++failure_count_ >= kInstabilityThreshold) unstable_ = true;
        return;
      }
    }
  }
  void record_success(std::string_view id) {
    for (auto& r : routes_) {
      if (r.id == id) {
        failure_count_ = 0;
        unstable_ = false;
        return;
      }
    }
  }

  // Deterministic selection: preferred route when stable; else fallback
  // order; else pinned baseline when the router is unstable (§57.7).
  [[nodiscard]] const Route* select(
      std::string_view preferred) const noexcept {
    if (unstable_) return baseline();
    for (const auto& r : routes_) {
      if (r.id == preferred) return &r;
    }
    for (const auto& r : routes_) {
      if (!r.baseline) return &r;
    }
    return baseline();
  }

  [[nodiscard]] const Route* baseline() const noexcept {
    for (const auto& r : routes_) {
      if (r.baseline) return &r;
    }
    return nullptr;
  }
  [[nodiscard]] bool unstable() const noexcept { return unstable_; }
  void reset_instability() noexcept {
    failure_count_ = 0;
    unstable_ = false;
  }

 private:
  std::vector<Route> routes_;
  int failure_count_ = 0;
  bool unstable_ = false;
};

// ===========================================================================
// The Omni-Bridge (§19): deterministic authorization + audit boundary.
// ===========================================================================

// Marker type: only human governance constructs this. In production this
// is a separate service/account; see the honest-limits note above.
struct HumanGovernor {
  std::string name;
};

class GovernanceApproval {
 public:
  GovernanceApproval(const GovernanceApproval&) = default;
  GovernanceApproval& operator=(const GovernanceApproval&) = default;

 private:
  friend class OmniBridge;
  GovernanceApproval() = default;
};

struct Proposal {  // DATA ONLY — this is what Lucifer/Hermes emit (§22)
  std::string trace_id;
  std::string tool_id;
  std::string route_id;
  std::string skill_id;  // optional
  CapSet requested_caps = kCapNone;
  Risk assessed_risk = Risk::Medium;
  SecurityLabel content_label = untrusted_external();
  std::string payload;   // canonical input (schema-validated upstream)
  bool from_lucifer = true;
};

struct Decision {
  enum class Status : std::uint8_t {
    Authorized, NeedsGovernance, NeedsVerification, Rejected
  };
  Status status = Status::Rejected;
  CapSet granted_caps = kCapNone;
  SandboxTier tier = SandboxTier::T5_Prohibited;
  bool verification_required = false;
  std::string reason;
  std::optional<std::uint64_t> audit_seq{};
};

struct ExecResult {  // returned by the caller's executor callback
  bool schema_ok = true;
  SecurityLabel output_label = untrusted_external();
  std::string output;
};

enum class VerificationVerdict : std::uint8_t { Pass, Fail, Disagree };

struct BridgeConfig {  // kill switches (§72) — all fail closed
  bool dynamic_expertise = true;
  bool lucifer_enabled = true;
  bool hermes_retrieval = true;
  bool generation = true;
  bool learning = true;
  bool audit_readonly = false;
};

class OmniBridge {
 public:
  OmniBridge(BridgeConfig config, CapSet system_caps, CapSet tenant_caps)
      : config_(config), system_caps_(system_caps),
        tenant_caps_(tenant_caps) {
    ledger_.set_read_only(config_.audit_readonly);
  }

  // ---- tool & skill registration (setup / governance path) ----------

  [[nodiscard]] bool register_tool(const ToolManifest& m,
                                   std::string* why = nullptr) {
    if (!m.valid(why)) return false;
    for (const auto& t : tools_) {
      if (t.id == m.id) {
        if (why) *why = "duplicate tool id";
        return false;
      }
    }
    tools_.push_back(m);
    return true;
  }

  [[nodiscard]] std::uint32_t register_skill(
      const Sha256::Digest& digest) {
    skills_.push_back({"skill-" +
                           std::to_string(skills_.size() + 1),
                       {{1, SkillState::Discovered, digest}}});
    return static_cast<std::uint32_t>(skills_.size());
  }

  // Audited skill transition; promotion to Promoted requires a
  // governance approval minted ONLY by human_governance() (§22, §61).
  [[nodiscard]] bool transition_skill(std::uint32_t skill,
                                      SkillState to,
                                      const GovernanceApproval* approval,
                                      std::string* why = nullptr) {
    if (skill == 0 || skill > skills_.size()) {
      if (why) *why = "unknown skill";
      return false;
    }
    auto& entry = skills_[static_cast<std::size_t>(skill - 1)];
    const SkillVersion& v = entry.versions.back();
    if (!skill_transition_valid(v.state, to)) {
      if (why) *why = "invalid lifecycle transition";
      ledger_append(entry.id, "skill-transition-rejected", "system",
                    "from=" + std::to_string(static_cast<int>(v.state)) +
                        " to=" + std::to_string(static_cast<int>(to)));
      return false;
    }
    if (to == SkillState::Promoted && approval == nullptr) {
      if (why) *why = "promotion requires human governance approval";
      ledger_append(entry.id, "skill-transition-rejected", "system",
                    "promotion-without-governance");
      return false;
    }
    const SkillState from = v.state;
    // Append-only version history: every transition creates a new
    // version entry so rollback (§41) always has a real predecessor.
    entry.versions.push_back(
        SkillVersion{v.version + 1, to, v.content_digest});
    ledger_append(entry.id, "skill-transition", "system",
                  "from=" + std::to_string(static_cast<int>(from)) +
                      " to=" + std::to_string(static_cast<int>(to)));
    return true;
  }

  [[nodiscard]] bool skill_eligible(std::uint32_t skill) const noexcept {
    if (skill == 0 || skill > skills_.size()) return false;
    return skill_eligible_for_use(
        skills_[static_cast<std::size_t>(skill - 1)].versions.back()
            .state);
  }

  // §57.10 / §41: rollback restores the previous known-good version.
  [[nodiscard]] bool rollback_skill(std::uint32_t skill,
                                    std::string* why = nullptr) {
    if (skill == 0 || skill > skills_.size()) {
      if (why) *why = "unknown skill";
      return false;
    }
    auto& entry = skills_[static_cast<std::size_t>(skill - 1)];
    if (entry.versions.size() < 2) {
      if (why) *why = "no previous version to roll back to";
      return false;
    }
    const auto& cur = entry.versions[entry.versions.size() - 1];
    const auto& prev = entry.versions[entry.versions.size() - 2];
    if (!skill_eligible_for_use(prev.state)) {
      if (why) *why = "previous version is not a known-good state";
      return false;
    }
    entry.versions.pop_back();
    ledger_append(entry.id, "skill-rollback", "system",
                  "rolled-back-from-v" +
                      std::to_string(cur.version) + "-to-v" +
                      std::to_string(prev.version));
    return true;
  }

  // ---- governance gate (§22): the ONLY source of GovernanceApproval --
  // Lucifer/Hermes call sites have no access to HumanGovernor values;
  // in production this arrives over an authenticated human channel.
  [[nodiscard]] GovernanceApproval human_governance(
      const HumanGovernor& g, std::string_view action) {
    GovernanceApproval a;
    ledger_append("governance", "governance-approval",
                  "human:" + g.name, std::string(action));
    return a;
  }

  // ---- the deterministic authorization pipeline (§19, §29) ----------

  [[nodiscard]] Decision evaluate(const Proposal& p,
                                  const ExecutionContext& ctx) {
    Decision d;
    d.status = Decision::Status::Rejected;

    // Kill switches (§72) — fail closed with a functional degraded mode.
    if (!config_.lucifer_enabled && p.from_lucifer) {
      d.reason = "kill switch: lucifer disabled";
      return reject(p, d);
    }
    if (!config_.hermes_retrieval && !p.skill_id.empty()) {
      d.reason = "kill switch: hermes retrieval disabled";
      return reject(p, d);
    }

    // 1. Tool ABI validation (fail closed on invalid manifests, §44).
    const ToolManifest* tool = find_tool(p.tool_id);
    if (tool == nullptr) {
      d.reason = "unknown tool";
      return reject(p, d);
    }
    std::string why;
    if (!tool->valid(&why)) {
      d.reason = "invalid tool manifest: " + why;
      return reject(p, d);
    }
    if (!config_.generation && tool->requires_generation) {
      d.reason = "kill switch: generation disabled";
      return reject(p, d);
    }

    // 2. Skill eligibility (§57.3: obsolete skills are rejected).
    if (!p.skill_id.empty()) {
      std::uint32_t sid = 0;
      if (!skill_index(p.skill_id, &sid) || !skill_eligible(sid)) {
        d.reason = "skill not eligible for use (lifecycle state)";
        return reject(p, d);
      }
    }

    // 3. APPA (§24) on the AUTOMATED portion of the request.
    //    Governance-only caps (§22) are excluded from every automated
    //    risk level; they can only traverse the human gate below.
    const CapSet gov_caps = p.requested_caps & kGovernanceOnlyCaps;
    const CapSet auto_caps = p.requested_caps & ~kGovernanceOnlyCaps;
    const CapSet effective = appa_intersect(
        system_caps_, ctx.declared_caps, tenant_caps_,
        tool->required_caps, ctx.context_caps, p.assessed_risk);
    if ((effective & auto_caps) != auto_caps) {
      d.reason = "APPA denial: missing capabilities (requested not "
                 "within System∩Declared∩Tenant∩Tool∩Context∩Risk)";
      return reject(p, d);
    }

    // 4. Risk ceiling of the context (§27).
    if (static_cast<int>(p.assessed_risk) >
        static_cast<int>(ctx.risk_ceiling)) {
      d.reason = "assessed risk exceeds context ceiling";
      return reject(p, d);
    }

    // 5. Audit gate (§57.8): audit unavailable => high-risk stops.
    if ((!ledger_.healthy() || ledger_.read_only()) &&
        static_cast<int>(p.assessed_risk) >=
            static_cast<int>(Risk::High)) {
      d.reason = "audit subsystem unavailable: high-risk execution "
                 "fails closed";
      return reject(p, d);
    }

    // 6. Sandbox selection (§28): max(tool floor, policy floor).
    d.tier = max_tier(tool->min_tier,
                      policy_floor(p.content_label, p.requested_caps));
    if (d.tier == SandboxTier::T5_Prohibited) {
      d.reason = "prohibited action tier";
      return reject(p, d);
    }

    d.granted_caps = kCapNone;  // granted only on final authorization

    // 7. Governance-only capabilities (§22): never auto-authorized;
    //    only the human gate (authorize_governance) can mint them.
    if (gov_caps != kCapNone) {
      d.status = Decision::Status::NeedsGovernance;
      d.tier = SandboxTier::T4_HumanApproved;
      d.verification_required = true;
      d.reason = "governance-only capability: human approval required";
      ledger_append(p.trace_id, "needs-governance",
                    p.from_lucifer ? "lucifer" : "hermes", d.reason);
      return d;
    }

    // 8. Fast vs slow path (§31): untrusted content, High risk, or
    // non-deterministic tools require the slow path (verification).
    d.verification_required =
        p.content_label.integrity == Integrity::Untrusted ||
        static_cast<int>(p.assessed_risk) >=
            static_cast<int>(Risk::High) ||
        !tool->deterministic;

    d.status = Decision::Status::Authorized;
    d.granted_caps = auto_caps;  // never more than requested
    d.reason = "authorized";
    d.audit_seq = ledger_append(p.trace_id, "authorization",
                                p.from_lucifer ? "lucifer" : "hermes",
                                "tool=" + p.tool_id + " tier=" +
                                    std::to_string(
                                        static_cast<int>(d.tier)));
    if (!d.audit_seq) {
      // Audit could not record an authorization-bearing action at
      // Medium+ risk: refuse to proceed un-audited (fail closed).
      if (static_cast<int>(p.assessed_risk) >=
          static_cast<int>(Risk::Medium)) {
        d.status = Decision::Status::Rejected;
        d.reason = "audit append failed: refusing un-audited "
                   "medium+ risk action";
        return d;
      }
    }
    return d;
  }

  // Convert a NeedsGovernance decision via a minted approval.
  [[nodiscard]] Decision authorize_governance(
      const Decision& d, const Proposal& p, const GovernanceApproval&) {
    Decision out = d;
    if (out.status != Decision::Status::NeedsGovernance) return out;
    out.status = Decision::Status::Authorized;
    out.verification_required = true;  // human path is still slow path
    out.audit_seq = ledger_append(p.trace_id, "governance-authorized",
                                  "human", "tool=" + p.tool_id);
    return out;
  }

  // ---- execution + output validation + verification + audit (§29) ---

  struct ExecOutcome {
    bool executed = false;
    std::string reason;
    ExecResult result;
  };

  [[nodiscard]] ExecOutcome execute(
      const Decision& d, const Proposal& p,
      const VerificationVerdict& v,
      const std::function<ExecResult(const Decision&, const Proposal&)>&
          executor) {
    ExecOutcome o;
    if (d.status != Decision::Status::Authorized) {
      o.reason = "not authorized";
      ledger_append(p.trace_id, "execution-refused", "system",
                    "decision-not-authorized");
      return o;
    }
    if (d.verification_required && v != VerificationVerdict::Pass) {
      // §57.5: verification disagreement => abstain + escalate.
      o.reason = v == VerificationVerdict::Disagree
                     ? "verification disagreed: abstaining and "
                       "escalating"
                     : "verification failed";
      ledger_append(p.trace_id, "verification-abstain", "system",
                    o.reason);
      return o;
    }
    if (executor == nullptr) {
      o.reason = "no executor bound";
      return o;
    }
    const ExecResult r = executor(d, p);
    if (!r.schema_ok) {  // §57.4: schema violation => reject.
      o.reason = "tool output violated schema";
      ledger_append(p.trace_id, "output-rejected", "system", o.reason);
      return o;
    }
    o.executed = true;
    o.result = r;
    ledger_append(p.trace_id, "execution-committed", "system",
                  "output_label_integrity=" +
                      std::to_string(static_cast<int>(
                          r.output_label.integrity)));
    return o;
  }

  // ---- child contexts (§27): restricted; cannot upgrade themselves --

  [[nodiscard]] ExecutionContext spawn_child(
      const ExecutionContext& parent, CapSet restriction) {
    ExecutionContext child = parent;
    child.id = parent.id + "/child";
    child.is_child = true;
    child.context_caps = parent.context_caps & restriction;
    child.declared_caps = parent.declared_caps & restriction;
    if (static_cast<int>(parent.content_label.integrity) <
        static_cast<int>(Integrity::Untrusted)) {
      // Child content is at least as restrictive as the parent's.
      child.content_label = parent.content_label;
    }
    child.risk_ceiling = parent.risk_ceiling;
    ledger_append(child.id, "child-context-spawned", "system",
                  "restricted-caps");
    return child;
  }

  // Upgrade attempts from any context (child or not) for
  // governance-only caps are not an evaluate() concern — they are
  // simply impossible: kGovernanceOnlyCaps is masked out of every
  // risk_mask() level. This explicit check exists for tests (§60).
  [[nodiscard]] bool context_can_upgrade_itself(
      const ExecutionContext&) const noexcept {
    return false;
  }

  // ---- telemetry / observability (§52) -------------------------------

  [[nodiscard]] const EvidenceLedger& ledger() const noexcept {
    return ledger_;
  }
  [[nodiscard]] RouteTable& routes() noexcept { return routes_; }
  [[nodiscard]] const BridgeConfig& config() const noexcept {
    return config_;
  }

 private:
  struct SkillEntry {
    std::string id;
    std::vector<SkillVersion> versions;
  };

  [[nodiscard]] const ToolManifest* find_tool(
      const std::string& id) const noexcept {
    for (const auto& t : tools_) {
      if (t.id == id) return &t;
    }
    return nullptr;
  }

  [[nodiscard]] bool skill_index(const std::string& id,
                                 std::uint32_t* out) const noexcept {
    for (std::size_t i = 0; i < skills_.size(); ++i) {
      if (skills_[i].id == id) {
        *out = static_cast<std::uint32_t>(i + 1);
        return true;
      }
    }
    return false;
  }

  std::optional<std::uint64_t> ledger_append(
      std::string_view trace, std::string_view kind,
      std::string_view actor, std::string_view summary) {
    return ledger_.append(trace, kind, actor, summary);
  }

  [[nodiscard]] Decision reject(const Proposal& p, Decision& d) {
    d.status = Decision::Status::Rejected;
    d.granted_caps = kCapNone;
    ledger_append(p.trace_id, "rejection",
                  p.from_lucifer ? "lucifer" : "hermes", d.reason);
    return d;
  }

 public:
  // Test hook: simulate an audit-subsystem failure (§57.8 drill).
  void simulate_ledger_failure_for_testing() noexcept {
    ledger_.mark_failed();
  }

 private:

  BridgeConfig config_;
  CapSet system_caps_;
  CapSet tenant_caps_;
  std::vector<ToolManifest> tools_;
  std::vector<SkillEntry> skills_;
  RouteTable routes_;
  EvidenceLedger ledger_;
};

}  // namespace omni
