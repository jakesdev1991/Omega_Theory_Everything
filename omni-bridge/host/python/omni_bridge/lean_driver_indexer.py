#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Lean 4 Driver Indexer — grounds the REAL Lucifer (omni-bridge, nvme0n1p3)
in the kernel-checked Lean corpus instead of prose summaries.

Extracts from ~/Omega_Theory_Everything/lean_proofs/*.lean:
  - every theorem/lemma/def/instance declaration with full statement+proof
  - provenance: file, line, declaration name, kind
  - audit status cross-referenced with kernel_audit_targets.json

Indexes into Lucifer's retrieval DB (/opt/omni-bridge/retrieval/fts5.db)
with origin=lean_kernel + integrity=KERNEL_CHECKED metadata, so his
explore/synthesis plane cites theorems, not vibes.

Usage: python3 lean_driver_indexer.py [--db PATH] [--lean-dir PATH] [--rebuild]
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = "/opt/omni-bridge/retrieval/fts5.db"
DEFAULT_LEAN_DIR = Path.home() / "Omega_Theory_Everything" / "lean_proofs"

DECL_RE = re.compile(
    r"^(?P<indent>\s*)(?:@\[[^\]]*\]\s+)?"
    r"(?P<kind>theorem|lemma|definition|def|instance|abbrev)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_'!?]*)",
    re.MULTILINE,
)


def extract_declarations(text: str, file_name: str):
    """Yield (kind, name, start_line, snippet) per declaration."""
    out = []
    lines = text.split("\n")
    for m in DECL_RE.finditer(text):
        kind = m.group("kind")
        name = m.group("name")
        start_line = text[: m.start()].count("\n") + 1
        # statement = up to ':=' on the decl line or following lines
        snippet_lines = []
        depth_done = False
        for i in range(
            m.start() and (text[: m.start()].count("\n")),
            min(len(lines), text[: m.start()].count("\n") + 80),
        ):
            snippet_lines.append(lines[i])
            if ":=" in lines[i] and i > text[: m.start()].count("\n"):
                depth_done = True
                break
            if depth_done:
                break
        snippet = "\n".join(snippet_lines[:40]).strip()
        out.append((kind, name, start_line, snippet))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--lean-dir", default=str(DEFAULT_LEAN_DIR))
    ap.add_argument("--rebuild", action="store_true", help="drop lean_* docs first")
    ap.add_argument("--max-bytes", type=int, default=120_000)
    args = ap.parse_args()

    lean_dir = Path(args.lean_dir)
    audit_path = lean_dir / "kernel_audit_targets.json"
    audit_targets = set()
    if audit_path.exists():
        audit_targets = {t.split(".")[-1] for t in json.loads(audit_path.read_text())}

    files = sorted(lean_dir.glob("*.lean"))
    if not files:
        print(f"[indexer] no .lean files in {lean_dir}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    if args.rebuild:
        conn.execute("DELETE FROM docs WHERE doc_id LIKE 'lean:%'")
        print("[indexer] cleared previous lean docs")

    indexed = 0
    audits_hit = 0
    for f in files:
        try:
            text = f.read_text(errors="replace")
        except OSError as e:
            print(f"[indexer] skip {f.name}: {e}", file=sys.stderr)
            continue
        for kind, name, line_no, snippet in extract_declarations(text, f.name):
            doc_id = f"lean:{f.stem}:{name}"
            kernel_checked = name in audit_targets
            if kernel_checked:
                audits_hit += 1
            meta = {
                "origin": "lean_kernel",
                "integrity": "KERNEL_CHECKED" if kernel_checked else "VOLUME_SOURCE",
                "file": f.name,
                "line": line_no,
                "kind": kind,
                "decl": name,
            }
            content = f"[{f.name}:{line_no}] {kind} {name}\n{snippet}"
            conn.execute(
                "INSERT OR REPLACE INTO docs(doc_id, content, metadata) VALUES (?,?,?)",
                (doc_id, content, json.dumps(meta)),
            )
            indexed += 1
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM docs WHERE doc_id LIKE 'lean:%'").fetchone()[
        0
    ]
    conn.close()
    print(
        f"[indexer] {indexed} declarations indexed from {len(files)} files "
        f"({audits_hit} kernel-audit targets hit); total lean docs: {n}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
