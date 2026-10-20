<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE and ../LICENSES/MIT.txt. -->

# Project status inventory — 2026-10-20

**Revision inventoried:** `91dc170d1bf7a7f82efda2f39d6c6ef953c5ee0c` (`main` HEAD:
"Add OmegaScienceMinter: PoKW receipt-claim leg for the Sepolia pilot").
**Method:** read-only audit of the local tree plus GitHub state via the public API,
performed by Muse (AI assistant) at the owner's request. No test suites were
re-run for this snapshot; where a claim below rests on execution, it says so.
This document supersedes [`PROJECT_STATUS_2026-10-08.md`](PROJECT_STATUS_2026-10-08.md)
as the current-state pointer; the October 8 inventory remains as the historical
record of its date.

---

## 1. Bottom line

The two open wounds of the October 8 inventory are closed: **PR #51** (Security
Scan fix) and **PR #56** (honesty-remediation passes ten and eleven) both merged
on 2026-10-10. Development since then has been additive — the Omni-Bridge host
layer and the $OMEGA Proof-of-Kernel-Verified-Work (PoKW) pilot work landed on
`main` on 2026-10-16/17 — but the coordination layer has not kept up: the
October 15 status and supply-chain train (**PR #84**) sits unmerged while `main`
moves under it, 35 pull requests are open (there are no open issues), and a few
present-tense references in the tree are stale (Section 6).

Separately, the RCOD §5 metrics-as-features claim was tested under a
pre-registered design (five runs, four variants) and is **negative in every
variant**; that thread is closed (Section 4).

## 2. What changed since 2026-10-08

- **PR #51 merged 2026-10-10** — CI Security Scan fixed; the Python suite is
  enforced in CI.
- **PR #56 merged 2026-10-10** — honesty-remediation passes ten and eleven
  landed, closing the October 8 inventory's headline gap.
- **2026-10-16** — Omni-Bridge host layer: C++23 daemon plus Python planes,
  PoKW mining, grounded chat; the host daemon was then unified with the
  portable core (`evaluate()` is the policy authority). This adds a Lean
  module outside `lean_proofs/` (`omni-bridge/host/lean/SecureRoute.lean`).
- **2026-10-16** — $OMEGA PoKW whitepaper v0.1.
- **2026-10-16** — `docs/SECURITY-REVIEW-2026-10-16.md` added.
- **2026-10-17** — OmegaScienceMinter: PoKW receipt-claim leg for the Sepolia
  pilot.

## 3. Open pull requests (35) — triage

| Group | PRs | State / needed action |
|---|---|---|
| Status + supply chain (Arena) | #84 | Large: `PROJECT_STATUS_2026-10-15`, SBOM, launch-readiness gates, workflow and README changes. Open since 2026-10-15; `main` has moved since — expect conflicts. Owner decision: rebase and merge, or close and re-extract the pieces. |
| Likely superseded | #14 (Security Scan fix, 2026-09-23) | PR #51 fixed the same job and merged; #14 is a closure candidate. |
| Overlapping #84 | #1 (setup-python 5→7), #2 (checkout 4→7), #3 (codeql 3→4) | Touch the same workflow files as #84; resolve together with the #84 decision. |
| Substantive, stale | #34 (theory/whitepaper alignment, Sep 25), #27 (simulation physics v4.0 alignment, Sep 24) | A month old against a moved `main`; need owner review or closure. |
| Dependabot majors | #93/#88 (react 19.3), #67 (typescript 7), #73/#66/#60 (@noble/hashes 2.x), #64/#59 (@noble/curves 2.x), #41/#39 (solana-program), #9 (pandas 3.x), #8 (scipy), #7 (matplotlib), #12 (networkx), #37 (sympy), #58 (mcp), #24 (solc) | Review individually; majors can break builds. |
| Dependabot minors/patches | #92, #91, #90, #89, #87, #86, #82, #80, #61, #21, #10 | Low risk; batch-merge candidates once CI is green. |

## 4. RCOD §5 — closed, negative

`RCOD_Operational_Logic.pdf` §5 proposes the RCOD swarm metrics as ML features
for shock-onset prediction; [`rcod/RESULTS.md`](../rcod/RESULTS.md) never tested
that use (it tested the governor intervention). A pre-registered benchmark
(synthetic AR(1) series with precursor ramps; PR-AUC primary; 10 seeds; paired
Wilcoxon) tested four variants: single-level swarm features, stacked 3-level
swarms, nested temporal partitions, and a parameter-free dyadic ladder. **All
four failed their pre-registered criteria against a plain rolling-statistics
baseline** (best variant: Wilcoxon p = 0.37). Per the design's pre-registered
one-shot rule the thread is closed — iterating redesigns on the same benchmark
would manufacture false positives. The full note, code, design, and per-seed
results are filed at `rcod/SECTION5_RESULTS.md` with artifacts under
`rcod/section5/` (branch `claude/rcod-section5-results`). The F5
damped-optimizer observation in `rcod/RESULTS.md` is a different mechanism
(optimizer dynamics, not time-series features) and is unaffected by this result.

## 5. Verified consistent (spot checks, this audit)

- README module counts: `lean_proofs/` holds **69 Lean modules** and
  `latex_docs/` holds **41 TeX volumes** — both match README.
- All **51 relative links in README** resolve to files in the tree.
- `web/public/book/full-book.md` contains **17 chapters** (see Section 6).
- The local clone of `main` was current and clean at the inventoried revision.

## 6. Known stale references

- **"16-chapter" descriptions of *Crucible: The Satoshi Protocol*** — 6
  occurrences: `launch/novel_day_one_plan.md` (2), `novel/README.md` (1),
  `whitepapers/care_economy_whitepaper.md` (3). The book has 17 chapters.
  Corrections are prepared on branch `claude/stale-reference-fixes`.
- `PROJECT_STATUS_2026-10-08.md` speaks of PRs #51/#56 as open in the present
  tense — accurate for its date; superseded by this inventory.

## 7. Owner actions recommended

1. **Enable branch protection on `main`** — require a pull request and passing
   CI before merge. As of this inventory `main` is **unprotected**: the
   repository's restraints are CI-after-the-fact plus process discipline, not
   enforcement. No agent or contributor should be able to push directly to
   `main`.
2. **Decide PR #84** — rebase and merge, or close and re-extract; it blocks
   #1/#2/#3 either way.
3. **Close #14** as superseded by #51; review or close #27 and #34.
