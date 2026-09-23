# Omni-Bridge (Hermes ⇄ Lucifer) — implementation status

Control-boundary reference implementation of the Master Installation and
Integration Directive. **What is here now compiles and passes its
acceptance suite.** What is not here is listed just as plainly. This
file follows the §80 evidence rule: no "complete/secure/proven/immune/
verified/production-ready" language without evidence.

## Build and run (C++23, no dependencies)

```bash
cd cpp && ./build.sh        # strict + TSan + ASan/UBSan
./build.sh --quick          # strict only
```

Latest measured run (this repo, this branch): **all 3 build
configurations pass; 18/18 test groups green.**

```
U1 sha256 vectors             U6 duplicate skill/tool
U2 APPA algebra               B1–B10 §57.1–§57.10 acceptance
U3 ledger chain + tamper      P1 §59 prompt injection (5 payloads)
U4 replay classification      P2 §60 capability escalation
U5 kill switches              S1 §58 end-to-end smoke
M1 evaluate() micro-benchmark
```

## Implemented and tested (here, in this repo)

| Directive section | What exists | Test evidence |
|---|---|---|
| §19–§23 APPA | `appa_intersect` = System∩Declared∩Tenant∩Tool∩Context∩Risk; risk only removes; governance-only caps masked from every automated level (static_asserts) | U2, B1, P2 |
| §22 governance gate | `kGovernanceOnlyCaps` = PolicyWrite\|SkillPromote\|MintOrMoveValue\|SecretRead\|CredentialRead → `NeedsGovernance`; only `human_governance()`→`authorize_governance()` converts to Authorized | P2 (+positive control), B1 |
| §24–§27 risk + tiers | Risk ladder Negligible→Critical (monotonic, static_asserts); SandboxTier T0–T5 selected = max(tool floor, policy floor) | U2, S1 |
| §32–§35 tool ABI | `ToolManifest` fail-closed validation (schema, caps, limits, determinism, rollback plan) | B4 (output schema gate), B9 |
| §36–§38 evidence ledger | hash-chained append-only, length-prefixed canonicalization, gap counting, read-only/failed states, tamper detection | U3, B8 |
| §39 replay classes | EXACT / DETERMINISTIC / SIMULATED_COUNTERFACTUAL / APPROXIMATE_COUNTERFACTUAL / NON_REPLAYABLE | U4 |
| §40–§41 skill lifecycle | DISCOVERED→…→RETIRED state machine, append-only version history, governance-gated promotion, rollback | B3, B9, B10 |
| §42 routes | RouteTable: 3 consecutive failures → unstable → baseline pinned | B6/B7 |
| §43 kill switches | OMNI_*_OFF + pinned versions, functional degraded modes | U5 |
| §44 pipeline | 8-step `evaluate()`: proposal→schema→APPA→risk-ceiling→audit-gate→tier→governance-split→authorize | S1 |
| §45–§46 execution | `execute()`: verification gate (Pass/Fail/Disagree→abstain+escalate), schema validation, rollback hooks | B4, B5 |
| §51–§52 contexts | parent→child context restriction; child can never exceed parent | B2 |
| §57.1–§57.10 | all ten B-scenarios | B1–B10 |
| §58–§60 | S1 smoke w/ trace correlation; P1 injection; P2 escalation | all green |
| §72 kill-switch env | parsed once at construction; no re-parse escape hatch | U5 |
| §84 first action | `scripts/omni_host_audit.py` — read-only host audit kit | run on Gentoo host |

## Known deviations (documented, not hidden)

- **SHA-256, not BLAKE3.** No vendored BLAKE3 in this environment;
  SHA-256 verified against FIPS 180-4 vectors (U1). The ledger chain
  format is hash-agnostic (32-byte digests); swapping in BLAKE3 is
  mechanical.
- **T1/T2/T3 enforcement is host-side.** The core *selects* the tier
  and refuses to authorize below the floor; actually spawning
  Wasmtime/container/microVM sandboxes is host runtime work.
- **The "human" in the governance gate is a process-internal call in
  this single-process implementation.** In production the approval
  must come from a separate service/account the LLM cannot reach. The
  type system (`GovernanceApproval` private ctor + friend) models the
  boundary; the deployment must enforce it.
- **M1 is a sanity bound, not a benchmark claim**: ~1.2 µs/op for
  `evaluate()` incl. ledger append, single thread, this container. The
  directive's "sub-ms routing" target is NOT claimed from this.

## Not implemented (host-side, blocked, or research)

- §1–§14, §43, §65, §75 host installation (partition audit, Btrfs
  subvolumes, systemd units, Wasmtime, containers, microVMs) —
  requires the Gentoo host; `scripts/omni_host_audit.py` is the first
  step to run there.
- `framework_complete bridge.json` — **never supplied**. Sections
  gated on it stay gated.
- Fisher-Rao, RCOD, Swarm Dissonance, ToM diagnostics (§71) — research
  tracks, some prior art in `../rcod/`; not wired into the bridge, and
  audit tiers are **not** truth oracles.
- L0/L1/L2 progressive disclosure loader for the Hermes skill store —
  the skill registry and lifecycle exist; the catalog/procedure/file
  tiering of actual skill content is not built.
- Lucifer's four exploration roles and Hermes's expert curation are
  *callers* of this boundary; they live outside it by design.

## Files

- `MASTER_DIRECTIVE.md` — the directive, verbatim, all 84 sections.
- `cpp/omni_bridge_core.hpp` — header-only core (namespace `omni`).
- `cpp/test_omni_bridge.cpp` — acceptance suite.
- `cpp/build.sh` — 3-config build.
- `scripts/omni_host_audit.py` — §84 host audit kit (read-only).

## Security model in one paragraph

Lucifer proposes; the bridge decides; Hermes governs. No LLM output is
authorization: every capability grant is computed by `evaluate()` from
intersections of declared/system/tenant/tool/context/risk sets, never
from proposal text. Governance-only capabilities are unreachable
without a `GovernanceApproval` that only `human_governance()` can mint.
Untrusted content (P1) never gains privilege — it can only make
execution *stricter* (slow path, verification, higher tier). The
ledger's chain integrity is checked on every append; audit failure
fails closed for Medium+ risk and gaps are counted for Low.
