# Lean proof audit — 2026-11-05

## Verdict: INCONCLUSIVE

No proof was checked by the requested external auditor in this environment.
The audit was run on **all 75 top-level `lean_proofs/*.lean` files** (74 proof
modules and `lakefile.lean`); every invocation exited 3 with `INCONCLUSIVE` and
zero Lean runs because neither Lean nor Lake was available. This does **not**
mean the proofs failed, nor that they passed. The per-file paths, SHA-256
hashes and verdicts are in [LEAN_AUDIT_INVENTORY.json](LEAN_AUDIT_INVENTORY.json).

Auditor: [`jakesdev1991/lean-proof-audit`](https://github.com/jakesdev1991/lean-proof-audit),
commit `d91d47f57d0c6a140ee68473f75498bfab7ce20b`, version 2.0.0. Target:
Omega repository commit `dd83f48c560986162cbe256589c20f8c64c55274`, including
local commit `dd83f48`, which was not on the remote PR branch when audited.
No structured claim files were supplied. No claim-to-statement comparison,
mutation-based dependence check, elaborated-statement inspection, or transitive
axiom inspection was performed by the requested auditor.

**Subsequent CI update:** After this audit snapshot, the remote push succeeded.
The normalized-overlap additions, including a namespace-only adjustment for the
repository's lexical name checker, passed the full Lean build and selected
288-target kernel-axiom audit in [run 42132144272](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/42132144272)
at commit `c770dce`. This does not change the external auditor's inconclusive
result for the original snapshot or replace its unrun full-corpus checks.

## Findings

### 1. Auditor report can incorrectly say “yes” when nothing ran

**Reproduced; affects the auditor's reporting, not proof validity.** On
`QRegionChannel.lean`, the JSON correctly gives `overall: inconclusive`,
`lean_runs: 0`, and `no_toolchain`, but simultaneously gives `sound: true` and
`statements_honest: true`. The Markdown renders these as “Is every theorem sound
(kernel + axioms)? yes” and “Do the statements say what they appear to say?
yes”. Both are unsupported: the report itself says every check was skipped.

The cause is visible in the auditor's pinned `lean_audit/report.py`:
`build_report` sets these booleans from the absence of findings rather than
requiring that the relevant checks completed; `render_markdown` renders `true`
as “yes”. `cli.py`'s no-toolchain path supplies only an inconclusive build
result. This contradicts the intended unknown-is-not-pass discipline. Treat
these fields and rendered answers as **unknown**, not as evidence. A proper fix
in the separate auditor repository would give each question an explicit
pass/fail/unknown state and test missing, skipped and inconclusive checks; this
audit did not modify that repository.

### 2. Normalized-overlap additions were not kernel-verified at audit time

The earlier Lean CI [run 42128816079](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/42128816079)
passed the full `ToE` build and its selected **284-target** transitive axiom
audit at commit `ab6ad24`. That includes the binary benchmark's mutual
information and bit-flip monotonicity, and the Planck dictionary. It does not
include the four additional normalized-overlap declarations at local commit
`dd83f48`; the push had been rejected by the remote at audit time, so their
compilation and axiom status were then **unknown**. The 288-target kernel audit
subsequently passed at `c770dce`, as recorded above; CI's earlier 284-target
pass alone did not verify those additions.

### 3. Model-interface mismatch remains a real conceptual issue

`OmegaProtocol.Φ_pos` states that existing Q-region pair overlap is strictly
positive, while the classical two-bit benchmark's independent endpoint has
zero mutual information. A proposed normalized benchmark overlap therefore
cannot equal the existing pairwise overlap at that endpoint. The local
`independent_not_existing_overlap` declaration expresses this and was later
kernel-checked as noted above. It is a mismatch between these particular
definitions—not evidence that all possible Q-region
models fail. No identification of the classical channel with physical Q-region
evolution is established.

Similarly, the existing pairwise definition sets `Φ R R = 1`; the chosen COD
profile vanishes at `Φ = 1`. Treating a self-pair as a spacetime point would
therefore give a zero operational scale. No such physical point-mapping is
established; this is a warning against silently imposing one.

### 4. Carried assumptions and interpretation remain open

The independent ledger census reports **14 open entries** and **16 distinct
carried `law_*` names**. Source scans for declaration-level axioms are not a
substitute for tracing these structure fields, checking theorem assumptions, or
showing that a model represents nature. Both Planck profiles are chosen forms;
their mathematical matching does not derive either profile or determine Φ
from physical Q-region data.

## Independent checks performed locally

These are the Omega repository's existing checks, **not results from the
external Lean auditor**:

- `audit_axioms.py --max 0`: zero source declarations matching its axiom/opaque
  scan. This is not a full transitive kernel axiom audit.
- `audit_vacuity.py`: all eight configured lexical counters zero. This does not
  prove semantic non-vacuity or detect every unused hypothesis.
- `check_statement_audit.py`: 161 physics-sounding volume declarations
  covered; 33 carried rows at the existing cap. The rows are human judgements,
  not proofs that hypotheses are satisfiable.
- Declaration parity: 75 files match the current baseline; 74 text mirrors are
  in sync. The baseline was updated when the source changed, so this establishes
  consistency with the updated snapshot, not that no code changed.
- Lean tooling unit tests: 111 passed. Auditor's own Python unit tests: 62 passed;
  neither set supplies a Lean toolchain or proves the corpus.

## Required follow-up

1. **Completed after the audit:** the push succeeded, and the full Lean build
   and selected transitive kernel audit passed at `c770dce`.
2. Run the requested auditor on each proof module in an environment with the
   pinned Lean 4.32.0, a built Mathlib project and the auditor's conformance
   probes. Retain machine-readable per-file results and account for mutation
   budget, timeouts, private declarations and all inconclusive checks. A static
   scan or the earlier selected 284-target kernel audit is not equivalent.
3. Provide independently authored structured claim files for the important
   Planck, Q-region, metric and gravity theorems. The auditor cannot infer
   intended physical claims from prose, and copying the formal statement into a
   claim file would not validate intent.
4. Fix the external auditor's inconclusive-report booleans before relying on
   its yes/no summary. Do not silently interpret missing evidence as a pass.

The full requested kernel-backed audit remains **unfinished**.
