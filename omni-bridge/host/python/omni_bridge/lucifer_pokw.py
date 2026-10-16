#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Lucifer PoKW capabilities: verify / conjecture / backlog / mine ops.

Extends the exploration plane with the Proof-of-Kernel-Verified-Work loop
(whitepapers/omega_pokw_whitepaper.md):

  backlog    — list/claim tasks from the machine-readable proof backlog
  verify     — kernel-check a candidate proof text locally (lake build)
  conjecture — run the repo's conjecture_pilot transforms + numeric gate
  mine       — one autonomous loop iteration: pull → verify → record

Deterministic, no network, no LLM. Every result carries provenance and an
explicit epistemic category (NUMERICAL vs KERNEL vs CANDIDATE), matching the
pilot's preregistration discipline: a numeric PASS is evidence, never a proof.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path.home() / "Omega_Theory_Everything"
LEAN_DIR = REPO / "lean_proofs"
CONJ = REPO / "conjecture_pilot"
BACKLOG = Path("/opt/omni-bridge/telemetry/proof_backlog.json")
MINE_LOG = Path("/opt/omni-bridge/telemetry/mine_ledger.jsonl")
CLAIMS = Path("/opt/omni-bridge/telemetry/claimed_tasks.json")
# Scratch dir must be writable under the systemd unit's ProtectHome=read-only
# (the repo is read-only to the service; /opt/omni-bridge is ReadWritePaths).
SCRATCH = Path("/opt/omni-bridge/telemetry/pokw_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)

LEAN_ENV = dict(os.environ)
LEAN_ENV["PATH"] = f"{Path.home() / '.elan/bin'}:{LEAN_ENV.get('PATH', '')}"

DECL_RE = re.compile(
    r"^(?:@\[[^\]]*\]\s+)?(?:private\s+|protected\s+)?"
    r"(theorem|lemma|def|abbrev|instance)\s+([A-Za-z_][A-Za-z0-9_?!*]*)",
    re.M,
)


def _sh(cmd, cwd, timeout=120, env=None):
    r = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or LEAN_ENV,
    )
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------- backlog
def op_backlog(req: dict) -> dict:
    acts = req.get("action", "list")
    if not BACKLOG.exists():
        return {"status": "error", "error": f"backlog missing: {BACKLOG}"}
    doc = json.loads(BACKLOG.read_text())

    if acts == "list":
        open_tasks = [t for t in doc["tasks"] if t.get("status") == "open"]
        return {
            "status": "ok",
            "op": "backlog",
            "open_tasks": len(open_tasks),
            "total": len(doc["tasks"]),
            "tasks": [
                {k: t[k] for k in ("task_id", "class", "module", "task", "status")}
                for t in open_tasks[: int(req.get("limit", 20))]
            ],
        }

    if acts == "claim":
        tid = req.get("task_id", "")
        task = next((t for t in doc["tasks"] if t["task_id"] == tid), None)
        if not task:
            return {"status": "error", "error": f"unknown task {tid}"}
        if task.get("status") != "open":
            return {"status": "error", "error": f"task {tid} is {task.get('status')}"}
        task["status"] = "claimed"
        task["claimed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        task["claimed_by"] = req.get("miner", "local-miner")
        BACKLOG.write_text(json.dumps(doc, indent=2))
        claims = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}
        claims[tid] = task
        CLAIMS.write_text(json.dumps(claims, indent=2))
        return {"status": "ok", "op": "backlog", "claimed": task}
    elif acts == "release":
        tid = req.get("task_id", "")
        task = next((t for t in doc["tasks"] if t["task_id"] == tid), None)
        if not task:
            return {"status": "error", "error": f"unknown task {tid}"}
        task["status"] = "open"
        task.pop("claimed_at", None)
        task.pop("claimed_by", None)
        BACKLOG.write_text(json.dumps(doc, indent=2))
        claims = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}
        claims.pop(tid, None)
        CLAIMS.write_text(json.dumps(claims, indent=2))
        return {"status": "ok", "op": "backlog", "released": tid}

    return {"status": "error", "error": f"unknown action: {acts}"}


# ---------------------------------------------------------------- verify
def op_verify(req: dict) -> dict:
    """Kernel-check candidate Lean text. Two modes:

    standalone : the text must be self-contained (imports allowed).
                 Compiled as a temp module inside the lake env.
    module      : req['module'] names an existing corpus file whose current
                 on-disk content should be re-kernel-checked (incremental).
    """
    mode = req.get("mode", "standalone")
    t0 = time.perf_counter()

    if mode == "module":
        mod = req.get("module", "")
        if not re.fullmatch(r"[A-Za-z0-9_]+", mod or ""):
            return {"status": "error", "error": "bad module name"}
        rc, out, err = _sh(["lake", "build", mod], cwd=LEAN_DIR, timeout=1800)
        dt = time.perf_counter() - t0
        ok = rc == 0 and "Build completed successfully" in (out + err)
        return {
            "status": "ok",
            "op": "verify",
            "mode": "module",
            "verified": ok,
            "returncode": rc,
            "duration_s": round(dt, 2),
            "output_tail": (err or out)[-800:],
        }

    code = req.get("code", "")
    if not code.strip():
        return {"status": "error", "error": "no code provided"}
    # Write to the service-writable scratch dir; lake env resolves imports
    # from the repo's lakefile regardless of the candidate file's location.
    tmp = SCRATCH / f"candidate_{uuid.uuid4().hex[:10]}.lean"
    tmp.write_text(code)
    mod = tmp.stem
    try:
        rc, out, err = _sh(["lake", "env", "lean", str(tmp)], cwd=LEAN_DIR, timeout=900)
        dt = time.perf_counter() - t0
        errors = [
            ln
            for ln in (out + err).splitlines()
            if "error:" in ln or "unsolved goals" in ln
        ]
        decls = [f"{k} {n}" for k, n in DECL_RE.findall(code)]
        return {
            "status": "ok",
            "op": "verify",
            "mode": "standalone",
            "verified": rc == 0,
            "returncode": rc,
            "duration_s": round(dt, 2),
            "declarations": decls,
            "error_count": len(errors),
            "errors": errors[:12],
            "output_tail": (out + err)[-600:],
        }
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------ conjecture
def op_conjecture(req: dict) -> dict:
    """Run the repo's preregistered conjecture transforms over a module.

    Numeric gate = evidence only (epistemic category NUMERICAL), per the
    pilot's preregistration. Class-B registration requires a later KERNEL
    verification (the verify op) to earn full weight.
    """
    module = req.get("module", "")
    if not re.fullmatch(r"[A-Za-z0-9_]+", module or ""):
        return {
            "status": "error",
            "error": "module required (e.g. Vol18_StatisticalMechanics)",
        }
    src = LEAN_DIR / f"{module}.lean"
    if not src.exists():
        return {"status": "error", "error": f"no such module: {module}"}

    sys.path.insert(0, str(REPO))
    try:
        from conjecture_pilot.extract import extract_real_defs, extract_statements
        from conjecture_pilot.numeric import check
        from conjecture_pilot.transform import generate
    except ImportError as e:
        return {"status": "error", "error": f"conjecture_pilot import failed: {e}"}

    try:
        statements, unparsed = extract_statements(src)
        defs = extract_real_defs(src)
    except Exception as e:
        return {"status": "error", "error": f"extract failed: {e}"}

    limit = int(req.get("limit", 12))
    results = []
    for stmt in statements:
        for cand in generate(stmt, family=module):
            try:
                v = check(cand.statement, defs)
            except Exception:
                continue
            if v.status == "not_applicable":
                continue
            rec = {
                "source_decl": cand.source,
                "transform": cand.transformation,
                "statement": cand.statement.conclusion[:240],
                "verdict": v.status,  # pass | fail | inconclusive
                "epistemic": "NUMERICAL",
                "counterexample": v.counterexample,
            }
            if v.status == "pass":
                rec["note"] = (
                    "evidence only — numeric PASS under Lean total-"
                    "function semantics; requires kernel check"
                )
            results.append(rec)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    passed = [r for r in results if r["verdict"] == "pass"]
    refuted = [r for r in results if r["verdict"] == "fail"]
    return {
        "status": "ok",
        "op": "conjecture",
        "module": module,
        "statements": len(statements),
        "unparsed": len(unparsed),
        "candidates": len(results),
        "numeric_pass": len(passed),
        "refuted": len(refuted),
        "inconclusive": len(results) - len(passed) - len(refuted),
        "results": results,
        "note": "preregistered transforms; PASS is evidence, not proof",
    }


# ------------------------------------------------------------------ mine
def _mine_record(rec: dict) -> None:
    MINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MINE_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def op_mine(req: dict) -> dict:
    """One autonomous mining iteration (advisory pass — proposes, never
    merges). Attempts: backlog refresh → claim an open Class-A task →
    conjecture screen (Class B) → record everything to the mine ledger.

    Deliberately does NOT write to the repo: output is a proposal artifact
    under /opt/omni-bridge/telemetry/ for human/governance review — the
    PoKW loop's RATIFY stage is not this op's to perform (§22).
    """
    it: dict[str, object] = {
        "iter_id": f"mine_{uuid.uuid4().hex[:12]}",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _mine_record({**it, "phase": "start"})

    # 1. refresh backlog from the audit ledger
    try:
        rc, out, err = _sh(
            [
                "python3",
                "/opt/omni-bridge/source/omni-bridge/scripts/extract_backlog.py",
            ],
            cwd=REPO,
            timeout=60,
        )
        it["backlog_refresh"] = "ok" if rc == 0 else f"exit {rc}"
    except Exception as e:
        it["backlog_refresh"] = str(e)[:120]

    # 2. claim next open task
    doc = json.loads(BACKLOG.read_text()) if BACKLOG.exists() else {"tasks": []}
    open_tasks = [t for t in doc["tasks"] if t.get("status") == "open"]
    if not open_tasks:
        it["phase"] = "done"
        it["result"] = "backlog empty"
        _mine_record({**it, "phase": "done", "result": "backlog empty"})
        return {"status": "ok", "op": "mine", **it}
    task = open_tasks[0]
    op_backlog(
        {"action": "claim", "task_id": task["task_id"], "miner": "autonomous-loop"}
    )
    it["claimed"] = task["task_id"]
    it["module"] = task.get("module", "")
    # 3. conjecture screen on the target module (Class-B evidence pass)
    module_name = str(it["module"] or "")
    mod = module_name.replace(".lean", "")
    conj = None
    if mod and re.fullmatch(r"[A-Za-z0-9_]+", mod):
        try:
            conj = op_conjecture({"module": mod, "limit": 8})
        except Exception as e:
            conj = {"status": "error", "error": str(e)[:120]}
    it["conjecture"] = (
        {
            "numeric_pass": conj.get("numeric_pass", 0),
            "candidates": conj.get("candidates", 0),
            "refuted": conj.get("refuted", 0),
        }
        if conj and conj.get("status") == "ok"
        else conj
    )
    # 4. kernel re-verify the module's current state (baseline evidence)
    ver = op_verify({"mode": "module", "module": mod})
    it["module_baseline_verified"] = (
        ver.get("verified") if ver.get("status") == "ok" else None
    )

    # 5. propose (never merge): write a proposal artifact for review
    proposal = {"task": task, "conjecture": conj, "baseline": ver, "iter": it}
    ppath = Path("/opt/omni-bridge/telemetry/proposals") / f"{it['iter_id']}.json"
    ppath.parent.mkdir(parents=True, exist_ok=True)
    ppath.write_text(json.dumps(proposal, indent=2))
    it["proposal"] = str(ppath)

    # 6. advance the queue: mark the task done for this advisory pass so
    #    the next iteration pulls the NEXT open task (claims preserved by
    #    the extractor; 'done' = proposal emitted, awaiting human review).
    doc = json.loads(BACKLOG.read_text())
    for t in doc["tasks"]:
        if t["task_id"] == task["task_id"]:
            t["status"] = "done"
            t["proposal"] = str(ppath)
            t["completed_at"] = it["ts"]
    BACKLOG.write_text(json.dumps(doc, indent=2))
    claims = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}
    claims.pop(task["task_id"], None)
    CLAIMS.write_text(json.dumps(claims, indent=2))

    _mine_record({**it, "phase": "proposed"})
    return {"status": "ok", "op": "mine", **it}


OPS = {
    "backlog": op_backlog,
    "verify": op_verify,
    "conjecture": op_conjecture,
    "mine": op_mine,
}
