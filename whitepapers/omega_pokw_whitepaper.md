<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# $OMEGA Proof of Kernel-Verified Work (PoKW)

**Version:** 0.1 research draft
**Status:** Design specification; not an offer, investment recommendation, or live
monetary system.
**Relationship:** instantiates the Proof of Useful Work (§4) framework of
[`../tri_token_sovereign_economy_blueprint.md`](../tri_token_sovereign_economy_blueprint.md)
for `$OMEGA`, where the useful work is **formal mathematics**: proving,
sharpening, or refuting statements in the Omega formalization corpus
([`../lean_proofs/`](../lean_proofs/)).

> **One-paragraph summary.** After bootstrap, new `$OMEGA` enters existence by
> exactly one mechanism: **governance-ratified batches of kernel-verified
> mathematical work.** A "miner" is any human or agent that closes an open
> proof task, produces a witness, or retires a misleading statement in the
> Lean corpus. Verification is not an oracle's opinion: a submission either
> passes `lake build ToE`, the kernel audit, and the honesty ratchets, or it
> does not. The corpus's own anti-vacuity gates — already CI-enforced and at
> zero — serve as the difficulty adjustment: trivial, restated, or misleading
> work earns nothing, by construction.

## 1. What this is and is not

**Is:** a claim-verification protocol in the blueprint's sense — a bounded,
auditable, appealable mechanism converting verified formal-mathematics
contributions into issuance rights, under governance control.

**Is not:**
- a claim that the Omega Theory of Everything is true, complete, or
  empirically validated (see [`../lean_proofs/PROOF_AUDIT.md`](../lean_proofs/PROOF_AUDIT.md):
  kernel-checked ≠ physically derived; weak model ≠ intended claim);
- a claim that physical interpretation is purchased by proofs — physical
  interpretation remains a separate, human, blinded decision
  ([`../conjecture_pilot/PILOT_PREREGISTRATION.md`](../conjecture_pilot/PILOT_PREREGISTRATION.md) §4);
- a token sale. The pilot remains Sepolia-only and valueless, per
  [`../evm/README.md`](../evm/README.md).

## 2. Position in the four-plane model

The blueprint separates money (SOV), contribution receipts (USE/TWC), and
governance scarcity (`$OMEGA`). PoKW respects that separation:

| Layer | What PoKW does there |
|---|---|
| **TWC (work receipts)** | Each accepted contribution mints a **work receipt** on the TWC plane via the blueprint §4.2 schema (`WorkClaim`, attestations, expiry, appeal). Receipts record *that* work happened and by whom. |
| **`$OMEGA` (macro scarcity)** | Ratified batches of receipts authorize a **bounded science-mint epoch payout**. `$OMEGA` never pays per-proof directly; it pays per *ratified batch*, so issuance remains a governance action with delay and appeal, not an automated faucet. |
| **CARE** | Untouched. No proof work may alter care, consent, or appeal guarantees. |
| **AMITY** | Untouched. Conversion boundaries unchanged. |

This keeps the blueprint's invariant — no oracle, validator, administrator, or
model unilaterally mints value — because three independent things must agree:
the kernel (math), the attestors (attribution), and the Governor+Timelock
(authorization).

## 3. The mining loop

```text
PULL     miner selects an open task from the Proof Backlog (§4)
   |
WORK     human or agent produces a Lean proof / witness / counterexample
   |
VERIFY   deterministic, public, reproducible — no discretion:
         lake build ToE  +  audit_kernel.py (260-target transitive audit)
         + audit_vacuity ratchets + statement-audit + cited-name gates
   |
ATTEST   ≥2 independent attestations for high-impact classes (blueprint §4.1)
         attestors stake reputation/bond; appeals preserved
   |
BATCH    receipts accumulate into an epoch batch (merkle root)
   |
RATIFY   $OMEGA Governor votes + TimelockController delay
   |
MINT     bounded epoch emission per §6; receipts minted on TWC plane
```

The verification function is the part most "useful work" schemes lack: it is
**binary, deterministic, and already running in CI** on this repository
(evidence: Lean CI runs 38251369698 / 38254475070, green). Anyone can replay a
submission's verification on identical pinned toolchains (Lean/mathlib
v4.32.0). There is no verifier trust problem for the mathematics itself —
only for attribution, which is what the attestation layer exists for.

## 4. Work classes (the Proof Backlog)

The backlog is not invented for this paper; it is the live "Next:" ledger of
the audit passes, which the corpus maintains per module. Initial classes:

| Class | Work | Verified by | Weight |
|---|---|---|---|
| **A — close an open proof task** | e.g. Vol04 general-H dynamics; Vol18 arbitrary finite ensembles; Vol13 constraint-from-action; Vol30 derive Kepler from two-body ODE; CBwK ownership/concurrency refinement | kernel + gates | highest |
| **B — registered conjecture** | candidate from [`../conjecture_pilot/`](../conjecture_pilot/) transforms (weaken/converse/generalize) that survives numeric screening; held-out freeze enforced | numeric gate + well-formedness | small on registration; main weight if later proven (Class A) |
| **C — counterexample / retirement** | witness that retires or sharpens a misleading statement (the mechanism the corpus itself uses; e.g. the `*_Stmt` retirements) | kernel + vacuity ratchets | high — negative results are first-class |
| **D — model upgrade** | replace a degenerate/wrong model with a non-degenerate one passing all gates (e.g. the Vol25 marginal-sum correction) | kernel + gates + regression suite | high |
| **E — empirical benchmark** | physics-adjacent validation (telemetry, simulation) | preregistration + independent replication | **zero monetary weight until independently validated** (blueprint §5: isolated from monetary finality) |

Backlog items carry difficulty estimates from their audit-provenance
(declarations touched, prior pass history), which governance uses to set
per-class caps. The backlog is public, versioned, and derived from
`PROOF_AUDIT.md` — not from a private queue.

## 5. Difficulty adjustment is the honesty layer

Most proof-of-work schemes adjust difficulty by making the puzzle harder.
PoKW cannot — a theorem is either proven or not. Instead, difficulty is the
**ratchet thresholds tightening as the corpus grows**, exactly as the audit
passes have already done six times over:

- `--max-misleading 0 --max-axiom-names 0 --max-unit-stubs 0
  --max-trivial-proofs 0` (all enforced in CI today);
- the `carried_cap` on legacy-disposition rows (statement_audit.json) —
  re-deriving renamed variants earns nothing;
- the cited-name gate and `legacy_statement_aliases.json` baseline —
  retired names must survive nowhere.

Consequence: the strategy "farm many small restatements" is not merely
unprofitable; it is **rejected by the same gates that accept the work**.
Farming pressure tightens the gates further. The anti-gaming control is the
mathematics itself.

Remaining gaming vectors and their controls (blueprint §4.4):

| Vector | Control |
|---|---|
| Volume farming (lemma-splitting, renames) | vacuity + statement-audit gates; `carried_cap` |
| Conjecture overfitting to known results | held-out freeze + preregistration (conjecture_pilot); leakage scan |
| Attestor collusion | ≥2 independent attestations, rotating review, bonds, proportional appealable penalties |
| Authorship sybil (one entity, many hats) | receipt weighting bounded and reliability-decayed; duplicate-evidence detection |
| Governance capture of ratification | TimelockController delay; public batch roots; per-epoch mint caps (§6) bound the worst case even under capture |
| Claiming physical meaning for math | Class E isolation; disposition taxonomy (witness/conditional) carried on receipts |

## 6. Emission

Extends the `$OMEGA` whitepaper's curve:

```text
issuance(t) = bootstrap(t) + reserve_adjustment(t) + science_mint(t)
```

with `science_mint` the only post-bootstrap emission path:

```text
epoch_payout  = min(epoch_cap, Σ_receipt_weights(batch) × ratification_scale)
epoch_cap      = science_emission_remaining × k_epoch     (k_epoch ≤ 2%)
global_cap     = S_sci                                        (e.g. ≤ 15% of
                                                               initial supply,
                                                               governance-set)
```

Properties (mirroring CBwK, which the corpus already proves for this exact
shape — `CBwK.runCosts_conservation`, `runCosts_eq_some_iff`):

- payout ≤ cap always; unratified weight never mints (guard = `Some`);
- remaining balance is conserved and monotonically decreasing;
- a batch's weight is committed before ratification, so late additions
  cannot retroactively dilute earlier epochs.

Receipts decay/revalidate per blueprint §2.2; decayed receipts lose ratification
eligibility, not historical attribution.

## 7. Sepolia pilot form

Valueless, Sepolia-only, mirroring the existing pilot's guard pattern:

```solidity
interface IOmegaScienceMinter {
    /// @notice merkle root of a governance-ratified receipt batch
    function submitRatifiedBatch(bytes32 batchRoot, uint256 epochPayoutWei) external;
    /// @notice claim epoch payout against a receipt leaf in a ratified batch
    function claimEpoch(bytes32[] calldata proof, Receipt calldata leaf) external;
}
```

with `SepoliaTestnetOnly` construction guard, no owner-mint path, epoch caps
enforced in-contract, and the timelock as the only `submitRatifiedBatch`
proposer. **Not yet implemented** — see status table. The pilot does not
verify Lean on-chain; it verifies *governance's ratification of an
off-chain, publicly reproducible CI run*, which is the correct division of
labor (chain stores authorization and attribution; kernel stores truth).

## 8. Why this mechanism fits this corpus specifically

1. **Verification already exists and is adversarially hardened.** The gates
   were built by sixteen audit passes whose explicit purpose was defeating
   overstated claims. A mining mechanism built on plausibility would need to
   invent its own referees; this one inherits referees that have already
   survived an adversarial audit.
2. **The corpus rewards refutation.** Most sci-token schemes collapse because
   only positive results pay. Here Class C pays for *retiring* claims — the
   corpus's own history (RCOD's honest negative verdict, the `*_Stmt`
   retirements) is the proof-of-concept that this incentive is coherent.
3. **Negative externalities are bounded by construction.** If the research
   program stalls, `science_mint` simply pays nothing; the asset does not
   need to pretend progress. Emission tracks verified mathematical output
   one-for-one or not at all.
4. **Agents can mine.** An AI that produces kernel-checked proofs is a
   legitimate miner — the Omni-Bridge's governance boundary
   ([`../omni-bridge/`](../omni-bridge/)) already defines the capability,
   sandbox, and audit constraints under which such an agent may act. A
   mining agent is a Lucifer-proposing, Hermes-governed worker whose output
   is a PR.

## 9. What this paper does not claim

- No claim that proof volume measures scientific truth. It measures
  **formalization progress** — a different, honest metric.
- No claim of miner decentralization guarantees; sybil resistance here is the
  blueprint's (bonds, attestor rotation, bounded weights), not a novel
  consensus protocol.
- No mainnet authorization. Nothing here modifies the pilot's valueless,
  Sepolia-only boundary.
- Open problems: receipt privacy for pseudonymous miners; cross-referencing
  receipts to Lean declaration hashes (planned: `decl_name@commit` in the
  receipt's `evidence_root`); emissions during the bootstrap window;
  governance-set cap parameters; counsel review before any public description
  of this mechanism.

## Implementation status

| Piece | Status | Where |
|---|---|---|
| Proof backlog (work classes) | **live** — per-module "Next:" ledger | `lean_proofs/PROOF_AUDIT.md` |
| Deterministic verification | **live in CI** — kernel audit + gates, green | `.github/workflows/lean-ci.yml` |
| Conjecture generation + preregistration | **live** — 24 tests, held-out freeze | `conjecture_pilot/` |
| Work-receipt schema | **specified** | blueprint §4.2 |
| Attestation layer | **specified**, not implemented | blueprint §4.1 |
| `science_mint` emission + caps | **specified here** (§6), not implemented | this paper |
| Sepolia pilot minter contract | **not implemented** — interface sketch §7 | future `evm/contracts/` |
| Governance wiring (Governor+Timelock) | **pilot exists** for other actions; ratify-mint action not added | `evm/` |
