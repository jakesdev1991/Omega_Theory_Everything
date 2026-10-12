#!/usr/bin/env python3
"""Comment-insensitive parity check for the Lean corpus.

Roadmap Phase 5 replaced overclaiming labels in ``lean_proofs/`` by editing
comments and docstrings only. This module is the machine check that such an
edit cannot have changed a declaration: every ``.lean`` file is reduced to a
canonical *code projection* -- nested block comments, docstrings and line
comments removed, string and character literals preserved, blank lines and
trailing whitespace dropped -- and that projection is hashed.

Two ways to use it:

* default / ``--update``: compare the working tree with the committed baseline
  ``declaration_parity.json``. A comment-only edit leaves every hash
  untouched; any change to a name, statement or proof moves a hash and fails
  until the baseline is deliberately regenerated with ``--update``.
* ``--against <git-ref>``: compare the working tree with a git revision and
  leave the baseline alone. Use it to show that a branch differs from ``main``
  by comments alone.

This is a lexical check, not a Lean parser and not a kernel check. It never
decides whether a file compiles, and it cannot see semantic equivalence: two
spellings of the same proof produce two hashes. What it does answer, exactly,
is "did the code change, independent of the prose?".

Known limitation: comment markers that appear inside a character literal or
between two primed identifiers on one line are not recognised as such. That
direction is conservative -- text may be kept as code -- and no file in this
corpus currently hits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from lean_source import lean_sources

ROOT = Path(__file__).resolve().parent
BASELINE_NAME = "declaration_parity.json"
BASELINE_VERSION = 1

# Declaration headers, as they appear once comments are stripped. Used only for
# the human-readable counts in the report; the hash is what the gate enforces.
DECLARATION = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\]\s*)*"
    r"(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|axiom|opaque|structure|class|inductive)"
    r"\s+\S"
)


def strip_comments(text: str) -> str:
    """Remove line comments and nested block comments, preserving literals.

    Unlike :func:`lean_source.mask_comments_and_strings`, string and character
    literals are kept verbatim: the parity check must notice a change to a
    literal, since that is code and not prose.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        pair = text[index : index + 2]

        if pair == "/-":
            depth = 1
            index += 2
            while index < length and depth > 0:
                if text[index : index + 2] == "/-":
                    depth += 1
                    index += 2
                elif text[index : index + 2] == "-/":
                    depth -= 1
                    index += 2
                else:
                    if text[index] == "\n":
                        out.append("\n")
                    index += 1
            continue

        if pair == "--":
            while index < length and text[index] != "\n":
                index += 1
            continue

        if char == '"':
            out.append(char)
            index += 1
            while index < length:
                current = text[index]
                if current == "\\" and index + 1 < length:
                    out.append(text[index : index + 2])
                    index += 2
                    continue
                out.append(current)
                index += 1
                if current == '"':
                    break
            continue

        # Character literals are one char or one escape, so they are three or
        # four bytes long: '"', '\n'. Primed identifiers such as `h'` are not
        # literals and are left to the ordinary path below.
        if char == "'" and (
            (index + 2 < length and text[index + 2] == "'")
            or (index + 1 < length and text[index + 1] == "\\")
        ):
            out.append(
                text[index : index + 3]
                if text[index + 1] != "\\"
                else text[index : index + 4]
            )
            index += 3 if text[index + 1] != "\\" else 4
            continue

        out.append(char)
        index += 1
    return "".join(out)


def code_projection(text: str) -> str:
    """Canonical, comment-free, formatting-insensitive form of a Lean file."""
    lines = [
        line.replace("\t", "    ").rstrip()
        for line in strip_comments(text).splitlines()
    ]
    return "\n".join(line for line in lines if line.strip())


def projection_digest(text: str) -> str:
    return hashlib.sha256(code_projection(text).encode("utf-8")).hexdigest()


def declaration_count(text: str) -> int:
    return sum(
        1 for line in code_projection(text).splitlines() if DECLARATION.match(line)
    )


def project_tree(root: Path) -> dict[str, dict[str, object]]:
    """Per-file projection digests, keyed by path relative to ``root``."""
    tree: dict[str, dict[str, object]] = {}
    for path in lean_sources(root):
        text = path.read_text(encoding="utf-8")
        projection = code_projection(text)
        tree[path.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(projection.encode("utf-8")).hexdigest(),
            "declarations": sum(
                1 for line in projection.splitlines() if DECLARATION.match(line)
            ),
            "code_lines": len(projection.splitlines()),
        }
    return tree


def git_show(root: Path, ref: str, relative: str) -> str | None:
    """Text of ``relative`` at ``ref``, or None when absent there."""
    repo = root.parent
    target = (Path(root.name) / relative).as_posix()
    command = ["git", "show", f"{ref}:{target}"]
    try:
        result = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:  # pragma: no cover - git missing is an operator error
        raise SystemExit(f"cannot run git: {error}") from error
    if result.returncode != 0:
        return None
    return result.stdout


def project_reference(
    root: Path, ref: str, keys: list[str]
) -> dict[str, dict[str, object]]:
    tree: dict[str, dict[str, object]] = {}
    for relative in keys:
        text = git_show(root, ref, relative)
        if text is None:
            continue
        projection = code_projection(text)
        tree[relative] = {
            "sha256": hashlib.sha256(projection.encode("utf-8")).hexdigest(),
            "declarations": sum(
                1 for line in projection.splitlines() if DECLARATION.match(line)
            ),
            "code_lines": len(projection.splitlines()),
        }
    return tree


def compare(
    current: dict[str, dict[str, object]],
    expected: dict[str, dict[str, object]],
) -> list[str]:
    """Describe every difference between two projection trees."""
    problems: list[str] = []
    for relative in sorted(set(current) | set(expected)):
        if relative not in expected:
            problems.append(
                f"{relative}: added (no reference entry)"
                if expected
                else f"{relative}: added (no baseline entry)"
            )
            continue
        if relative not in current:
            problems.append(f"{relative}: removed")
            continue
        before = expected[relative]
        after = current[relative]
        if before["sha256"] != after["sha256"]:
            problems.append(
                f"{relative}: code changed "
                f"(declarations {before['declarations']} -> {after['declarations']}, "
                f"code lines {before['code_lines']} -> {after['code_lines']})"
            )
    return problems


def load_baseline(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise SystemExit(
            f"missing baseline {path}; generate it with "
            f"'python3 {path.parent.name}/check_declaration_parity.py --update'"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != BASELINE_VERSION:
        raise SystemExit(f"unsupported baseline version {version!r} in {path}")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"baseline {path} has no file entries")
    return files


def write_baseline(path: Path, tree: dict[str, dict[str, object]]) -> None:
    payload = {"version": BASELINE_VERSION, "files": tree}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="Lean source directory")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=f"baseline file (default: <root>/{BASELINE_NAME})",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from the working tree",
    )
    parser.add_argument(
        "--against",
        metavar="REF",
        default=None,
        help="compare against a git revision instead of the baseline",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    baseline_path = args.baseline or (args.root / BASELINE_NAME)
    current = project_tree(args.root)

    if args.update:
        write_baseline(baseline_path, current)
        print(f"wrote {len(current)} file digests to {baseline_path}")
        return 0

    if args.against:
        expected = project_reference(args.root, args.against, sorted(current))
        source = f"git ref {args.against}"
    else:
        expected = load_baseline(baseline_path)
        source = str(baseline_path)

    problems = compare(current, expected)
    if args.json:
        print(
            json.dumps({"source": source, "files": len(current), "problems": problems})
        )
    elif problems:
        print(f"declaration parity against {source}: {len(problems)} difference(s)")
        for problem in problems:
            print(f"  {problem}")
        if not args.against:
            print(
                "\nIf the code change is intentional, regenerate the baseline:\n"
                "  python3 lean_proofs/check_declaration_parity.py --update"
            )
    else:
        print(f"declaration parity against {source}: {len(current)} files unchanged")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
