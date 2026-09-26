<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# conjecture_pilot — measurement-first conjecture generator (no learning)

Scaffold for the two-week pilot described in
[`PILOT_PREREGISTRATION.md`](PILOT_PREREGISTRATION.md). It turns the Lean
statements in `lean_proofs/` into candidate conjectures by four naive
transformations, screens them numerically under Lean's total-function
semantics, and (where a Lean toolchain is present) checks well-formedness and
whether cheap automation already closes them. Every event goes to an
append-only JSONL ledger with provenance and an explicit epistemic category.

```bash
python -m conjecture_pilot.run_pilot freeze                  # held-out inventory + leakage scan
python -m conjecture_pilot.run_pilot generate --split train   # 7 core modules
python -m conjecture_pilot.run_pilot generate --split heldout # refuses if held-out files changed
python -m conjecture_pilot.run_pilot report                   # conjecture_pilot/out/report.md
python -m pytest conjecture_pilot -q                          # 24 tests
```

Files: `extract.py` (lexical signature extraction), `transform.py`
(weaken / converse / generalize), `numeric.py` (falsifier), `gates.py`
(numeric + Lean gates, outcome labels), `freeze.py` (held-out freeze),
`schema.py` (records, ladder, ledger), `run_pilot.py` (CLI).
`out/` and `.scratch/` are git-ignored; `heldout_freeze.json` and the dry-run
report are committed as pre-registration artifacts.

Nothing here is a proof of anything about the Omega theory. A numeric PASS is
evidence; a Lean PASS is a kernel check of a *mathematical* statement; physical
significance is a separate, human, blinded decision (pre-registration §4).
