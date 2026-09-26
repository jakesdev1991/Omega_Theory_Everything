# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""CLI for the measurement-first pilot.

    python -m conjecture_pilot.run_pilot freeze
    python -m conjecture_pilot.run_pilot generate --split train
    python -m conjecture_pilot.run_pilot generate --split heldout
    python -m conjecture_pilot.run_pilot report

No learning, no adaptive allocation: every source statement gets every
transformation, every candidate gets every applicable gate, in fixed order.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .extract import _LEAN_PROOFS, extract_real_defs, extract_statements
from .freeze import HELDOUT_MODULES, TRAIN_MODULES, verify_freeze, write_freeze
from .gates import evaluate, lean_available, outcome
from .schema import Ledger
from .transform import TRANSFORMATIONS, generate

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out"
FREEZE_PATH = HERE / "heldout_freeze.json"


def cmd_freeze(args: argparse.Namespace) -> int:
    data = write_freeze(FREEZE_PATH, _LEAN_PROOFS)
    n = sum(len(v["statements"]) for v in data["inventory"].values())
    print(
        f"frozen {len(data['inventory'])} held-out modules, {n} statements, commit {data['frozen_at_commit'][:12]}"
    )
    print(f"upstream deps (allowed): {data['upstream_dependencies_allowed']}")
    for entry in data["downstream_leakage_masked"]:
        print(f"masked: {entry['module']}: {len(entry['references'])} references")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    modules = TRAIN_MODULES if args.split == "train" else HELDOUT_MODULES
    if args.split == "heldout":
        if not FREEZE_PATH.exists():
            print("refusing: run `freeze` first so the held-out inventory is fixed")
            return 2
        changed = verify_freeze(
            json.loads(FREEZE_PATH.read_text(encoding="utf-8")), _LEAN_PROOFS
        )
        if changed:
            print(f"refusing: held-out files changed since freeze: {changed}")
            return 2
    out = Path(args.out)
    ledger = Ledger(out / f"ledger_{args.split}.jsonl")
    use_lean = lean_available() and not args.no_lean
    started = time.time()
    ledger.append(
        {
            "kind": "run",
            "split": args.split,
            "modules": list(modules),
            "transformations": list(TRANSFORMATIONS),
            "lean_available": use_lean,
            "samples": args.samples,
            "started": started,
        }
    )
    totals: Counter[str] = Counter()
    for module in modules:
        path = _LEAN_PROOFS / f"{module}.lean"
        statements, unparsed = extract_statements(path, module)
        defs = extract_real_defs(path)
        for u in unparsed:
            ledger.append(
                {
                    "kind": "unparsed",
                    "module": u.module,
                    "name": u.name,
                    "line": u.line,
                    "reason": u.reason,
                }
            )
        seen: set[str] = set()
        for s in statements:
            ledger.append(
                {
                    "kind": "source",
                    "module": module,
                    "name": s.name,
                    "line": s.line,
                    "hypotheses": len(s.hypotheses()),
                    "lean": s.signature(),
                }
            )
            for cand in generate(s, module):
                if cand.id in seen:
                    ledger.append(
                        {
                            "kind": "duplicate",
                            "candidate_id": cand.id,
                            "source": cand.source,
                            "transformation": cand.transformation,
                        }
                    )
                    continue
                seen.add(cand.id)
                ledger.append(cand.to_record())
                results = evaluate(cand, defs, use_lean=use_lean, samples=args.samples)
                for r in results:
                    ledger.append(r.to_record())
                label = outcome(results)
                ledger.append(
                    {
                        "kind": "outcome",
                        "candidate_id": cand.id,
                        "module": module,
                        "transformation": cand.transformation,
                        "label": label,
                    }
                )
                totals[label] += 1
        print(
            f"{module}: {len(statements)} statements parsed, {len(unparsed)} unparsed, {len(seen)} candidates"
        )
    ledger.append(
        {
            "kind": "run_end",
            "split": args.split,
            "seconds": time.time() - started,
            "outcomes": dict(totals),
        }
    )
    print(f"outcomes: {dict(totals)}")
    print(f"ledger: {ledger.path} sha256={ledger.sha256()[:16]}")
    return 0


def _report_split(ledger: Ledger) -> str:
    records = list(ledger.records())
    if not records:
        return "_no ledger_\n"
    run = next((r for r in records if r["kind"] == "run"), {})
    sources = [r for r in records if r["kind"] == "source"]
    unparsed = [r for r in records if r["kind"] == "unparsed"]
    cands = {r["id"]: r for r in records if r["kind"] == "candidate"}
    outcomes = [r for r in records if r["kind"] == "outcome"]
    gates = [r for r in records if r["kind"] == "gate"]
    dupes = [r for r in records if r["kind"] == "duplicate"]
    end = next((r for r in records if r["kind"] == "run_end"), {})

    by_cell: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for o in outcomes:
        by_cell[(o["module"], o["transformation"])][o["label"]] += 1
    labels = sorted({o["label"] for o in outcomes})
    seconds_by_gate: Counter[str] = Counter()
    for g in gates:
        seconds_by_gate[g["gate"]] += g["seconds"]
    relevance: Counter[str] = Counter()
    for g in gates:
        if g["gate"] == "numeric":
            for v in g["evidence"].get("relevance", {}).values():
                relevance[v] += 1

    lines = []
    lines.append(
        f"- Lean toolchain available during run: **{run.get('lean_available')}**; samples per candidate: {run.get('samples')}"
    )
    lines.append(
        f"- Source statements parsed: **{len(sources)}**; unparsed: **{len(unparsed)}**; candidates: **{len(cands)}** (+{len(dupes)} duplicates dropped)"
    )
    lines.append(
        f"- Wall time: {end.get('seconds', 0):.1f} s; gate time by gate: "
        + ", ".join(f"{k} {v:.1f}s" for k, v in seconds_by_gate.items())
    )
    if relevance:
        lines.append(
            "- Hypothesis relevance (numeric, per hypothesis): "
            + ", ".join(f"{k} {v}" for k, v in sorted(relevance.items()))
        )
    lines.append("")
    lines.append(
        "| module | transformation | candidates | " + " | ".join(labels) + " |"
    )
    lines.append("|---|---|---:|" + "---:|" * len(labels))
    for (module, transformation), counter in sorted(by_cell.items()):
        total = sum(counter.values())
        lines.append(
            f"| {module} | {transformation} | {total} | "
            + " | ".join(str(counter.get(label, 0)) for label in labels)
            + " |"
        )
    total_counter: Counter[str] = Counter()
    for counter in by_cell.values():
        total_counter.update(counter)
    lines.append(
        f"| **all** | | **{sum(total_counter.values())}** | "
        + " | ".join(f"**{total_counter.get(label, 0)}**" for label in labels)
        + " |"
    )
    if unparsed:
        reasons = Counter(u["reason"] for u in unparsed)
        lines.append("")
        lines.append(
            "Unparsed by reason: " + ", ".join(f"{k} {v}" for k, v in reasons.items())
        )
    examples = [o for o in outcomes if o["label"] == "refuted_numerically"][:5]
    if examples:
        lines.append("")
        lines.append(
            "First numerically refuted candidates (counterexample recorded in ledger):"
        )
        gate_by_cand = {g["candidate_id"]: g for g in gates if g["gate"] == "numeric"}
        for o in examples:
            c = cands[o["candidate_id"]]
            ce = gate_by_cand[o["candidate_id"]]["evidence"].get("counterexample")
            lines.append(
                f"- `{c['statement']['name']}` ← {c['source']} [{c['transformation']}] counterexample {ce}"
            )
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    parts = ["# Pilot yield report (auto-generated; measurement only, no learning)\n"]
    for split in ("train", "heldout"):
        parts.append(f"## Split: {split}\n")
        parts.append(_report_split(Ledger(out / f"ledger_{split}.jsonl")))
    text = "\n".join(parts)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    gen = sub.add_parser("generate")
    gen.add_argument("--split", choices=("train", "heldout"), required=True)
    gen.add_argument("--out", default=str(DEFAULT_OUT))
    gen.add_argument("--samples", type=int, default=2000)
    gen.add_argument(
        "--no-lean", action="store_true", help="skip Lean gates even if available"
    )
    gen.set_defaults(func=cmd_generate)
    rep = sub.add_parser("report")
    rep.add_argument("--out", default=str(DEFAULT_OUT))
    rep.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    result: Any = args.func(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
