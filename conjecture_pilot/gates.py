# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Gates. Each returns PASS / FAIL / UNAVAILABLE / NOT_APPLICABLE, never a guess.

Lean-backed gates write a scratch file *outside* lean_proofs/ (so the CI grep
for ``sorry`` never sees it) and run ``lake env lean`` from lean_proofs/. When
the toolchain is absent they report UNAVAILABLE — the same policy as
lean_proofs/leandojo_tactic_harness.py.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .extract import _LEAN_PROOFS, RealDef
from .numeric import check
from .schema import Candidate, GateResult, Status

SCRATCH = Path(__file__).resolve().parent / ".scratch"
AUTOMATION = "by first | rfl | (norm_num) | (simp) | (positivity) | (linarith) | (nlinarith) | (aesop) | (exact?)"


def lean_available(root: Path = _LEAN_PROOFS) -> bool:
    return shutil.which("lake") is not None and (root / "lakefile.lean").exists()


def _scratch_source(candidate: Candidate, proof: str) -> str:
    st = candidate.statement
    imports = ["import Mathlib", f"import {st.module}"]
    opens = " ".join(st.namespaces) or st.module
    return "\n".join(imports) + f"\n\nopen {opens}\n\n" + st.render(proof=proof)


def run_lean(
    candidate: Candidate, proof: str, timeout: int = 120, root: Path = _LEAN_PROOFS
) -> tuple[Status, dict[str, object]]:
    if not lean_available(root):
        return Status.UNAVAILABLE, {"reason": "lake not on PATH in this environment"}
    SCRATCH.mkdir(exist_ok=True)
    path = SCRATCH / f"{candidate.id}.lean"
    path.write_text(_scratch_source(candidate, proof), encoding="utf-8")
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Status.FAIL, {"reason": "timeout", "timeout": timeout}
    finally:
        path.unlink(missing_ok=True)
    out = proc.stdout + proc.stderr
    errors = [line for line in out.splitlines() if ": error:" in line]
    uses_sorry = "declaration uses 'sorry'" in out
    evidence: dict[str, object] = {
        "returncode": proc.returncode,
        "errors": errors[:10],
        "uses_sorry": uses_sorry,
    }
    if proc.returncode != 0 or errors:
        return Status.FAIL, evidence
    return Status.PASS, evidence


def gate_wellformed(candidate: Candidate) -> GateResult:
    t0 = time.time()
    status, evidence = run_lean(candidate, "by sorry")
    if status is Status.PASS and not evidence.get("uses_sorry"):
        evidence["note"] = (
            "elaborated without sorry warning; statement may be closed by rfl-like elaboration"
        )
    return GateResult(candidate.id, "wellformed", status, evidence, time.time() - t0)


def gate_automation(candidate: Candidate) -> GateResult:
    """Rung 2: PASS here means *not closable* by cheap automation."""
    t0 = time.time()
    status, evidence = run_lean(candidate, AUTOMATION)
    if status is Status.UNAVAILABLE:
        return GateResult(
            candidate.id,
            "not_closable_by_automation",
            status,
            evidence,
            time.time() - t0,
        )
    closable = status is Status.PASS and not evidence.get("uses_sorry")
    return GateResult(
        candidate.id,
        "not_closable_by_automation",
        Status.FAIL if closable else Status.PASS,
        {**evidence, "closable_by_automation": closable},
        time.time() - t0,
    )


def gate_numeric(
    candidate: Candidate, defs: dict[str, RealDef], samples: int = 2000
) -> GateResult:
    t0 = time.time()
    verdict = check(candidate.statement, defs, samples=samples)
    status = {
        "pass": Status.PASS,
        "fail": Status.FAIL,
        "inconclusive": Status.INCONCLUSIVE,
        "not_applicable": Status.NOT_APPLICABLE,
    }[verdict.status]
    return GateResult(
        candidate.id, "numeric", status, verdict.to_dict(), time.time() - t0
    )


def evaluate(
    candidate: Candidate,
    defs: dict[str, RealDef],
    use_lean: bool = True,
    samples: int = 2000,
) -> list[GateResult]:
    """Cheap numeric screen first; Lean gates only for numeric survivors."""
    results = [gate_numeric(candidate, defs, samples)]
    numeric = results[0]
    if numeric.status is Status.FAIL:
        return results
    if use_lean:
        results.append(gate_wellformed(candidate))
        if results[-1].status is Status.PASS:
            results.append(gate_automation(candidate))
    return results


def outcome(results: list[GateResult]) -> str:
    """One label per candidate for the yield table."""
    by_gate = {r.gate: r for r in results}
    numeric = by_gate.get("numeric")
    if numeric is not None and numeric.status is Status.FAIL:
        reason = numeric.evidence.get("reason")
        return (
            "refuted_numerically"
            if reason == "counterexample"
            else "hypotheses_unsatisfiable"
        )
    wf = by_gate.get("wellformed")
    if wf is None or wf.status is Status.UNAVAILABLE:
        if numeric is not None and numeric.status is Status.PASS:
            return "numeric_pass_lean_unavailable"
        if numeric is not None and numeric.status is Status.INCONCLUSIVE:
            return "numeric_inconclusive_lean_unavailable"
        return "numeric_na_lean_unavailable"
    if wf.status is Status.FAIL:
        return "ill_formed"
    auto = by_gate.get("not_closable_by_automation")
    if auto is not None and auto.status is Status.FAIL:
        return "closable_by_automation"
    return "open_task"
