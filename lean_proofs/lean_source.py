"""Small lexical helpers for source audits, not a Lean parser or kernel checker."""

from __future__ import annotations

import re
from pathlib import Path


def mask_comments_and_strings(text: str) -> str:
    """Blank nested comments and string literals, preserving offsets/newlines.

    Lean block comments nest. Ignoring them with a non-greedy regex can expose
    commented-out declarations as code. Strings also may contain comment markers.
    Unterminated comments/strings are masked to EOF; Lean must reject the syntax.
    """
    out = list(text)
    i = 0
    depth = 0
    string = False
    line_comment = False
    while i < len(text):
        char = text[i]
        pair = text[i : i + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
            else:
                out[i] = " "
            i += 1
        elif depth:
            if pair in {"/-", "-/"}:
                depth += 1 if pair == "/-" else -1
                out[i : i + 2] = "  "
                i += 2
            else:
                if char != "\n":
                    out[i] = " "
                i += 1
        elif string:
            if char == "\\" and i + 1 < len(text):
                out[i] = " "
                if text[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
            else:
                if char == '"':
                    string = False
                if char != "\n":
                    out[i] = " "
                i += 1
        elif pair in {"/-", "--"}:
            depth = int(pair == "/-")
            line_comment = pair == "--"
            out[i : i + 2] = "  "
            i += 2
        elif char == '"':
            string = True
            out[i] = " "
            i += 1
        else:
            i += 1
    return "".join(out)


def lean_sources(root: Path) -> list[Path]:
    """Include nested local modules, but never downloaded packages/build output."""
    return sorted(
        path
        for path in root.rglob("*.lean")
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def module_imports(root: Path) -> dict[str, set[str]]:
    """Lexical local-module import map, including external names as leaf edges."""
    imports = re.compile(r"^[ \t]*(?:public[ \t]+)?import[ \t]+([^\n]+)", re.M)
    graph: dict[str, set[str]] = {}
    for path in lean_sources(root):
        if path == root / "lakefile.lean":
            continue
        name = ".".join(path.relative_to(root).with_suffix("").parts)
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        graph[name] = {
            dependency
            for match in imports.finditer(text)
            for dependency in match.group(1).split()
        }
    return graph


def find_import_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return a closed local dependency cycle, or None; external leaves are ignored."""
    complete: set[str] = set()
    active: set[str] = set()
    stack: list[str] = []

    def visit(name: str) -> list[str] | None:
        if name in active:
            return stack[stack.index(name) :] + [name]
        if name in complete:
            return None
        active.add(name)
        stack.append(name)
        for dependency in sorted(graph[name] & graph.keys()):
            cycle = visit(dependency)
            if cycle is not None:
                return cycle
        stack.pop()
        active.remove(name)
        complete.add(name)
        return None

    for name in sorted(graph):
        cycle = visit(name)
        if cycle is not None:
            return cycle
    return None
