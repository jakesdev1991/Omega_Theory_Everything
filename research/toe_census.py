# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Census of what the programme has not derived -- the distance to a TOE.

`docs/theory_ledger.json` lists, by hand, every quantity or choice the theory
needs but does not derive: free functions, free constants, chosen profiles,
matching conditions, and deliberate postulates. This script checks that the
ledger is well formed, that every evidence pointer still resolves, that the
count of carried `law_*` fields in the Lean corpus matches, and that neither
number has grown past its ratchet.

Why this exists. "Closer to a theory of everything" is otherwise a mood. A
candidate theory of everything has, by construction, no free *functions* and
very few free constants; everything else is derived or fixed by measurement.
So the honest distance metric is a count, and the only way a count means
something is if it cannot quietly go up. `counts must not rise` is enforceable,
`progress` is not, and pretending otherwise is how a research programme spends
five years getting further away from its own claims.

It is a ledger of judgement, not a proof: the same epistemic position as
`lean_proofs/statement_audit.json`. The script checks shape, evidence and the
ratchet. It cannot check that a disposition is *right*, and it says so in its
output.

Run: python research/toe_census.py [--list] [--json]
Exit status: 0 clean, 1 findings, 2 unreadable ledger.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "theory_ledger.json"

OPEN_STATUSES = ("undetermined", "chosen", "identified")
STATUSES = OPEN_STATUSES + ("postulate", "external", "derived")
KINDS = (
    "constant",
    "function",
    "profile",
    "dictionary",
    "ansatz",
    "model",
    "postulate",
)
REQUIRED = ("id", "kind", "name", "status", "evidence", "note")
NOTE_MIN_LENGTH = 40

LEAN_FIELD = re.compile(r"\blaw_[A-Za-z0-9_]+\b")


def load(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"{path.name}: expected an object with an 'items' list")
    if not isinstance(payload["items"], list) or not payload["items"]:
        raise ValueError(f"{path.name}: 'items' must be a non-empty list")
    if not isinstance(payload.get("ratchet"), dict):
        raise ValueError(f"{path.name}: a 'ratchet' object is required")
    return payload


def _heading_matches(heading: str, anchor: str) -> bool:
    """An anchor is either a section number (`#7`) or text (`#scope`)."""
    if anchor.isdigit():
        # Not an f-string: `{1,6}` would be read as a format field, not a quantifier.
        pattern = r"^#{1,6}\s+" + re.escape(anchor) + r"\."
        return bool(re.match(pattern, heading))
    return anchor.replace("-", " ").lower() in heading.lower()


def check_evidence(pointer: str, root: pathlib.Path) -> list[str]:
    """A pointer must resolve: `file:Declaration` or `file#anchor` or `file`."""
    problems: list[str] = []
    target, _, fragment = pointer.partition("#")
    if not fragment:
        target, _, fragment = pointer.partition(":")
    path = root / target
    if not path.is_file():
        return [f"`{pointer}`: no such file `{target}`"]
    text = path.read_text(encoding="utf-8")
    if not fragment:
        return problems
    if "#" in pointer:
        headings = [line for line in text.splitlines() if line.startswith("#")]
        if not any(_heading_matches(line, fragment) for line in headings):
            problems.append(f"`{pointer}`: no heading matches `{fragment}`")
    else:
        if not re.search(rf"\b{re.escape(fragment)}\b", text):
            problems.append(f"`{pointer}`: `{fragment}` does not occur in `{target}`")
    return problems


def validate(items: list[dict[str, Any]], root: pathlib.Path) -> list[str]:
    """Schema, uniqueness and evidence for every row."""
    problems: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        where = f"row {index + 1}"
        if not isinstance(item, dict):
            problems.append(f"{where}: not an object")
            continue
        for field in REQUIRED:
            if field not in item:
                problems.append(f"{where}: missing `{field}`")
        if problems and any(f"{where}: missing" in p for p in problems):
            continue
        item_id = str(item["id"])
        where = f"`{item_id}`"
        if item_id in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(item_id)
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", item_id):
            problems.append(f"{where}: id must be lower-case words joined by hyphens")
        if item["kind"] not in KINDS:
            problems.append(f"{where}: unknown kind `{item['kind']}`")
        if item["status"] not in STATUSES:
            problems.append(f"{where}: unknown status `{item['status']}`")
        if len(str(item["note"])) < NOTE_MIN_LENGTH:
            problems.append(f"{where}: note too short to be a judgement")
        pointer = str(item["evidence"])
        pointer_problems = check_evidence(pointer, root)
        problems.extend(f"{where}: {p}" for p in pointer_problems)
    return problems


def carried_laws(root: pathlib.Path) -> dict[str, list[str]]:
    """Carried `law_*` fields in the Lean corpus, by name -> declaring files."""
    found: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.lean")):
        for match in LEAN_FIELD.finditer(path.read_text(encoding="utf-8")):
            found.setdefault(match.group(0), []).append(path.name)
    return {name: sorted(set(files)) for name, files in sorted(found.items())}


def tally(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for item in items:
        counts[item["status"]] += 1
    counts["open"] = sum(counts[s] for s in OPEN_STATUSES)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=pathlib.Path, default=LEDGER)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--list", action="store_true", help="one line per row")
    parser.add_argument("--json", action="store_true", help="machine-readable tally")
    parser.add_argument(
        "--max-open", type=int, default=None, help="override the open-items ratchet"
    )
    parser.add_argument(
        "--max-carried-laws",
        type=int,
        default=None,
        help="override the carried-law ratchet",
    )
    args = parser.parse_args(argv)

    try:
        payload = load(args.ledger)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    items = payload["items"]
    ratchet = payload["ratchet"]
    problems = validate(items, args.root)

    laws = carried_laws(args.root / "lean_proofs")
    counts = tally(items)
    max_open = args.max_open if args.max_open is not None else ratchet.get("max_open")
    max_laws = (
        args.max_carried_laws
        if args.max_carried_laws is not None
        else ratchet.get("max_carried_laws")
    )
    if max_open is None or max_laws is None:
        print(
            "error: the ledger must state max_open and max_carried_laws",
            file=sys.stderr,
        )
        return 2

    if counts["open"] > max_open:
        problems.append(
            f"open items {counts['open']} exceed the ratchet {max_open}: a new "
            "undetermined choice was added without raising it in a visible diff"
        )
    if len(laws) > max_laws:
        problems.append(
            f"carried law fields {len(laws)} exceed the ratchet {max_laws}: a new "
            "`law_*` postulate was introduced without raising it"
        )
    if counts["open"] < max_open:
        problems.append(
            f"open items {counts['open']} are below the ratchet {max_open}: lower "
            "the ratchet, or the next added gap goes unnoticed"
        )
    if len(laws) < max_laws:
        problems.append(
            f"carried law fields {len(laws)} are below the ratchet {max_laws}: lower "
            "the ratchet, or the next added postulate goes unnoticed"
        )

    if args.json:
        print(
            json.dumps(
                {
                    "open": counts["open"],
                    "max_open": max_open,
                    "carried_laws": len(laws),
                    "max_carried_laws": max_laws,
                    "by_status": {k: v for k, v in counts.items() if k != "open"},
                    "problems": problems,
                },
                indent=1,
            )
        )
        return 1 if problems else 0

    print(
        f"theory ledger: {len(items)} tracked items -- "
        + ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s])
        + f"; plus {len(laws)} carried `law_*` fields in the Lean corpus"
    )
    print(
        f"open distance: {counts['open']} (ratchet {max_open}) | "
        f"carried laws: {len(laws)} (ratchet {max_laws})"
    )
    if args.list:
        for item in items:
            if item["status"] in OPEN_STATUSES:
                print(f"  [{item['status']:<12}] {item['id']:<24} {item['name']}")
        print("  carried law fields:")
        for name, files in laws.items():
            print(f"  [postulate   ] {name:<40} {', '.join(files)}")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1

    print(
        "OK: ledger well formed, every evidence pointer resolves, both ratchets hold. "
        "The rows are judgements; this checks that they exist and agree with the tree, "
        "not that any of them is right."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
