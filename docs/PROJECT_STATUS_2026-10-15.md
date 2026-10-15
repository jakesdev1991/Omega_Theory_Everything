<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE and ../LICENSES/MIT.txt. -->

# Project status inventory — 2026-10-15

**Revision reviewed:** `a764f34e7368bbc81166119c3f2516e732bebf89` — the current
`main` tip and the starting tip of `arena/4fb39d71-omega-theory-everything`.
**Working-tree state at the start:** clean; no local changes or untracked deliverables.
**Scope:** repository source and documents, current GitHub PR/CI/release metadata, and
reproducible local checks. This is a readiness inventory, not a scientific, legal, security,
or financial audit.

## Executive read

The project has moved past an unverified collection of ideas and prototypes: it has a broad,
recently green CI surface, reproducible research checks, formal-methods guardrails, several
well-tested offline/testnet-oriented prototypes, and a real unsigned wallet prerelease.

It is **not** at a theory-validation, public-financial-product, or mainnet-launch milestone.
The physics program remains exploratory; the repository's own simple Q-region benchmarks do
not recover the desired Schwarzschild behavior, the map from Q-region observables to a physical
metric is unresolved, no unique covariant theory or independent empirical prediction has been
frozen, and the empirical preregistration gate is correctly closed. The token/economy components
are local or valueless-pilot software only: no Sepolia deployment, Solana Devnet mint, AMITY
Taproot Asset, or value-bearing mainnet deployment is claimed.

The useful shorthand is:

| Stage | State | Meaning |
|---|---|---|
| Repository integrity and reproducibility | **Operational** | Clean tip; local consistency gate and current GitHub CI are green. |
| Formal mathematics | **Kernel-checked statements, carefully scoped** | Stronger than prose, but not a proof that Omega describes nature. |
| Physics research | **Active hypothesis program** | Candidate models and diagnostics exist; derivation and empirical validation do not. |
| Product prototypes | **Offline/testnet-ready software** | Wallet, web, governance, Nostr, bridge, and pilot tooling can be exercised safely. |
| Public value-bearing launch | **No-go / deliberately gated** | Deployment, independent review, legal work, and governance decisions are still prerequisites. |

---

## 1. Evidence collected for this snapshot

### Source and CI state

- At the start of this inventory, `origin/main` and the working checkout both resolved to `a764f34` (merged PR #83 on 2026-10-12).
- The most recent observed `main` CI run is
  [`38255777137`](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255777137):
  **14/14 jobs passed**. It covers Python quality/tests/simulations, Rust, EVM, Solana, web,
  mobile node, AMITY, Nostr, C++ acceptance suites, MCP, AI governor, desktop-wrapper checks,
  secret/vulnerability scanning, and documentation.
- The matching Lean CI run is
  [`38255776907`](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255776907):
  **passed**. The configured workflow runs `lake build ToE`, source-honesty gates, the
  transitive kernel-axiom audit, the statement audit, and the pinned-source cited-name check.
  Lean toolchains/sources are unavailable in this workspace, so this CI result is the evidence
  for those compiler-dependent checks.
- The local tree passed `python research/consistency_gate.py` with **16 passed, 0 failed,
  0 skipped**. This includes derivation, covariant, TFIM, LaTeX, claim, Lean-static,
  parity/mirror/statement-audit, pytest, Ruff, and MyPy checks.
- The local Python suite reported **215 passed, 81 subtests passed**.
- The strict, non-sanitizer (`--quick`) C++ acceptance paths also passed locally for the CBwK
  shadow-price pacer and Omni-Bridge. The sanitizer variants remain covered by the green CI job.

### Corpus size and evidence boundaries

| Item | Current count / state |
|---|---|
| Repository tracked files | 589 (about 43 MB of tracked content) |
| Lean modules | 69 |
| Lean theorem declarations (source count) | 930 |
| Plain-text Lean mirrors | 68, checked byte-identical by the mirror gate |
| LaTeX volumes | 41, checked for structural balance and cited declarations |
| Statement-audit rows | 163: 73 witness, 43 conditional, 14 renamed, 33 carried (cap 33) |
| Current public desktop release | `wallet-v0.1.0` prerelease, four installer formats plus `SHA256SUMS.txt` |

These counts demonstrate breadth and traceability; they do not independently establish physical
truth, security, legal clearance, or production readiness.

---

## 2. Research and formalization: where the theory actually is

### What is real and reproducible

- The repository has machine-checked algebra/numerics for the stated radial-metric identities,
  Markov model, finite transverse-field Ising diagnostic, covariant scalar equations, and the
  scalar–Gauss–Bonnet linear zero-mode calculation.
- The Markov benchmark is useful precisely because it is falsifiable at the model level: its
  specified mutual-information/correlation mapping drives the radial coefficient in the
  opposite direction from the selected Schwarzschild dictionary. That is a limitation of this
  simple model, not an empirical falsification of all possible Q-region models.
- The curvature-coupled scalar calculation finds a **linear** s-wave threshold near
  `eta / M^2 = 0.725613` for the selected candidate action. It is not a nonlinear black-hole
  solution, a stability result, an observational constraint, or a derivation from Q-regions.
- The Lean corpus has meaningful compiler and source-honesty coverage: no permitted unexpected
  axioms in the selected audit, no `sorry`-style escape hatches under the static gates, parity
  protection for comment-only changes, and explicit statement classifications.

### What remains scientifically open

1. **Ontology and map:** no derivation maps Q-region mutual information/correlations to
   `Phi`, a macroscopic metric, physical distance, or gravity.
2. **Dynamics:** the minimally coupled scalar action has a conditional GR branch; the
   curvature-coupled gravitational-scalar branch is the provisional next model. Neither is
   derived from the Q-region construction, and no nonlinear branch/stability solution is in
   hand.
3. **Prediction:** no parameter-fixed observable prediction has been frozen against independent
   holdout data. The historical `H_0 = 73.5` depletion result is a calibration input, not an
   independent prediction.
4. **Empirical gate:** theory choice, full equations/priors, observable map, immutable data and
   comparator, decision rule, and a clean reproduction must be frozen before a valid empirical
   test can be registered.
5. **Interpretation of Lean:** a kernel-checked conditional theorem is evidence for its formal
   statement only. It cannot convert a toy model, a stipulated premise, or a renamed claim into
   a derivation of nature.

**Scientific bottom line:** the appropriate external description is an exploratory,
well-instrumented research program—not a completed Theory of Everything and not an empirically
validated gravitational theory.

---

## 3. Engineering and product workstreams

| Workstream | Current state | Readiness boundary |
|---|---|---|
| Web app / Economy Test Console | Type-checked, tested, built in CI; provides offline scenario, audit-export, faucet, wallet, reader, and store surfaces | Suitable for local/offline test workflows; not evidence that rails or economics are live. |
| Wallet / desktop wrapper | Client-side wallet flows and a `wallet-v0.1.0` prerelease with checksums exist | Installers are unsigned, unaudited prototypes and target only valueless rails. Manual cross-platform launch/use evidence is still needed. |
| EVM `$OMEGA` | Sepolia-only fixed-supply `tOMEGA`, locking, Governor/Timelock, and claim-gate code are tested | No Sepolia deployment manifest or deployment; needs named test treasury/guardian, test ETH, preflight/drill, and independent review. |
| Solana TWC | Devnet-only `tTWC` fixed-supply SPL issuer/verifier and offline plans are tested | No Devnet mint/manifest; needs authorized signer/treasury, approved metadata URI/hash, preflight, drill, and review. |
| AMITY | Testnet-only operator/configuration scaffold, proof formatting, policy calculator, and non-broadcasting preflight are tested | No `litd`/`tapd`, channels, Universe service, Taproot Asset, or value-bearing service exists. |
| Mobile node / Nostr | NIP-90 daemon/client and static App Store surfaces are exercised in CI | A real relay, operating environment, observability, and policy review are needed before claiming a live service. |
| Omni-Bridge / CBwK / MCP / AI governor | C++ acceptance suites and MCP smoke test are covered by CI; the governor reports its negative prototype result honestly | These are engineering prototypes, not evidence for autonomous authority or optimizer benefit. |
| Novel / public access | The current manuscript is public under CC BY 4.0; optional wallet proofs do not restrict access | Rights/provenance and third-party-material review remain open; the public grant is not an exclusivity or ownership proof. |

The engineering portfolio is unusually broad for one repository. That makes **sequencing** more
important than adding another prototype: each rail has enough code to justify a deliberate test,
audit, and ownership plan, but none should be represented as a deployed value-bearing product.

---

## 4. Governance, safety, rights, and launch gates

The repository's own launch and C.A.R.E. documents correctly retain hard gates:

- **No mainnet/value-bearing release:** C.A.R.E., AMITY, OMEGA, and TWC remain research or
  valueless testnet components until social usability, machine-readable rules, voluntary/pilot
  evidence, simulations, independent security/privacy/safeguarding/legal reviews, and absence
  of unverified clinical/economic/performance claims are demonstrated.
- **No unreviewed token launch:** no independent EVM/Solana/AMITY audit, public testnet
  deployment, or final mainnet specification exists.
- **Key business decision still open:** select the future `$OMEGA` mainnet venue (Ethereum L1
  versus an EVM L2). Sepolia is only the chosen pilot network.
- **Rights and compliance are material blockers:** contributor/assignment history, third-party
  material, SBOM/license review, novel references, trademark clearance for TWC/project marks,
  securities/regulatory posture, and counsel review are unresolved. The path-specific split
  licensing describes intended scope; it is not proof of complete ownership.
- **Human authority remains required:** system documents prohibit silently expanding agent
  authority or using these models for automatic financial, medical, or public-benefit decisions.

These are not minor paperwork items. They are the boundary between a safe technical prototype
and a public, value-bearing system.

---

## 5. Project-management health

### GitHub backlog at the time of inventory

- **39 open pull requests; 0 open issues.**
- **36** open PRs are Dependabot updates; **3** are older human-authored/agent PRs.
- Merge-state split: **19 clean, 14 unstable, 6 dirty**.
- The open PRs include major-version migrations. For example, PR
  [#81](https://github.com/jakesdev1991/Omega_Theory_Everything/pull/81) (Hardhat 2 → 3) is
  red in the EVM CI job and must be treated as a migration investigation, not an automatic
  dependency bump.
- The three non-Dependabot PRs are stale and dirty against current `main` (#14, #27, #34).
  Their work should be explicitly superseded, re-evaluated, or closed rather than left as an
  ambiguous alternate history.

### Interpretation

The source tree is healthy; **maintenance attention is fragmented**. There is a successful CI
baseline, but the open dependency wave needs a triage policy: group compatible minor updates,
handle major framework changes in isolated migration branches with full test evidence, and close
obsolete/conflicting proposals. Do not merge a large backlog merely to make the PR count smaller.

---

## 6. Recommended order of work

1. **Keep the scientific truth boundary intact.** Choose one next research question—most
   importantly, a non-fitted Q-region-to-observable/dynamics proposal or a clearly independent
   falsifiable prediction—and write the assumptions and rejection criteria before implementation.
2. **Use testnet only for the next operational milestone.** If the owner elects to activate a
   rail, begin with the existing preflight and a valueless Sepolia/Devnet drill; record the
   manifest and incident/reversal procedure; then commission independent review. Do not jump to
   a mainnet decision through a prototype.
3. **Resolve launch ownership questions in parallel.** Engage appropriate legal/security/privacy
   reviewers, establish an SBOM/provenance process, and decide whether the public manuscript is
   the near-term product independent of token utility.
4. **Triage the PR queue.** Close/rebase the three stale manual PRs, then make a risk-ranked
   Dependabot plan. Start with tested compatible updates; isolate Hardhat, TypeScript, ESLint,
   Next/React, and cryptographic-library major changes.
5. **Treat the wallet prerelease as a field-test candidate, not a launch.** Verify installer
   provenance/checksums and manual behavior on each target OS; decide signing/notarization,
   support, telemetry/privacy, and incident ownership before wider distribution.

---

## 7. Limits of this inventory

- The local environment cannot run Lean, Rust, the pinned Mathlib source audit, or real-chain
  deployments; current green GitHub CI is cited for the first three, and no deployment is
  inferred for the last.
- Passing tests indicate that coded behavior matches the tests. They do not replace an
  independent security audit, legal review, empirical replication, or production-operating
  evidence.
- GitHub reports no currently open issue, but that is not a substitute for a risk register,
  owner assignments, or a release decision.

## Final assessment

**The project is strongest today as a disciplined research-and-prototyping platform.** Its next
success should be depth and decision quality—not more scope: one scientifically falsifiable
modeling pass, one deliberately bounded testnet drill if desired, and one focused pass over
legal/security/maintenance ownership. Until those gates move, the honest status remains:
**green code and useful prototypes; open science; no value-bearing launch.**
