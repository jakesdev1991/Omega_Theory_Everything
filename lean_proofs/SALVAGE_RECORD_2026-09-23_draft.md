#!/usr/bin/env python3
"""Salvage preservation record for local-draft-2026-09-23-0703 (3dcfa10).

Branch disposition (2026-10-16):
  SALVAGED  — auto_repair.py (as a read-only provenance/status tool,
              write paths deliberately not preserved; see its header)
  DEPRECATED — everything else on the branch, by explicit audit:

  1. OmegaUnifiedFoundation.lean draft version (1988-line diff): the draft
     carries 96 `*_Stmt` vacuous unification propositions
     (MasterUnification_Stmt, TheoryOfEverything_Stmt, ...) that the P0
     audit RETIRED as vacuous — main's file records this at the top:
     "`*_Stmt` unification theorems were retired as vacuous in the P0 audit".
     Restoring them would regress the claim-tightening pass.

  2. Per-Vol `*_from_omega` corollaries (draft-only theorems in Vol21-Vol54):
     definitional restatements (bodies are `rfl`-chains over the definitions)
     that PROOF_AUDIT.md documents as "retired (tenth pass)" — e.g.
     consciousness_from_omega, fermi_from_omega, compilerall,
     holographiccode, nothingboundsall. Sample bodies confirm tautologies:
     `mutualInformation R₁ R₂ = mutualInformation R₁ R₂`.

  3. lakefile.lean draft version: downgrades mathlib4 v4.32.0 → v4.8.0 and
     deletes the explicit 59-module-root ToE target — would break the
     kernel-check build definition (lake build ToE).

  4. Sim1/Sim4 .py restores of pre-rename scripts, LICENSES/ deletions,
     README/CONTRIBUTING regressions, ci.yml/lean-ci.yml/wallet-desktop.yml
     reversions, .lake build artifacts (excluded by .gitignore), trufflehog
     exclusion removal: all superseded by later passes on main.

The auto_repair.py salvage keeps only the read-only stub-counting function;
its write paths (which produced the retired definitional restatements) are
deliberately omitted so this historical tool cannot regress the audit
cleanup. The audit_vacuity.py / audit_axioms.py / audit_kernel.py /
check_* harnesses on main are the living replacements.

This file exists so the disposition decision is auditable in-repo.
"""
import subprocess
import sys

def main():
    print(__doc__)
    return 0

if __name__ == "__main__":
    sys.exit(main())
