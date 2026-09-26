# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Data model for the pilot: statements, candidates, gate results, ledger.

Every record carries an explicit epistemic category so that a kernel-checked
proof is never silently promoted to a physical claim (pre-registration §3).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class Epistemic(str, Enum):
    """The four categories a record may belong to. They never merge."""

    FORMAL = "formal"  # kernel-checked mathematical consequence
    CONJECTURE = "conjecture"  # proposed relationship, not yet verified
    NUMERICAL = "numerical"  # numerical result under stated assumptions
    PHYSICAL = "physical"  # interpretation requiring empirical support


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"  # checker ran, evidence too thin either way
    UNAVAILABLE = "unavailable"  # the checker (e.g. Lean) is not installed
    NOT_APPLICABLE = "not_applicable"  # the checker cannot express the claim
    ERROR = "error"


# Discovery ladder (pre-registration §4). Rung 3 is *utility*, not discovery.
RUNGS: dict[int, str] = {
    0: "kernel_checked",
    1: "hypotheses_satisfiable_and_load_bearing",
    2: "not_closable_by_automation_from_library",
    3: "reused_or_shortens_existing_proofs",
    4: "physically_significant_moves_a_tag",
}

_IDENT = re.compile(r"[^\W\d][\w'₀-₉.]*", re.UNICODE)


@dataclass(frozen=True)
class Binder:
    """One bracket group of a Lean signature, e.g. ``(h0 : 0 ≤ Φ)``."""

    names: tuple[str, ...]
    type: str
    bracket: str  # one of ( { [ ⦃
    is_hypothesis: bool  # lexical heuristic, see extract.is_hypothesis

    def render(self) -> str:
        close = {"(": ")", "{": "}", "[": "]", "⦃": "⦄"}[self.bracket]
        if self.bracket == "[" and not self.names:
            return f"[{self.type}]"
        if not self.type:
            return f"{self.bracket}{' '.join(self.names)}{close}"
        return f"{self.bracket}{' '.join(self.names)} : {self.type}{close}"


@dataclass(frozen=True)
class Statement:
    module: str
    name: str
    kind: str  # theorem | lemma
    line: int
    binders: tuple[Binder, ...]
    conclusion: str
    namespaces: tuple[str, ...] = ()

    def hypotheses(self) -> list[Binder]:
        return [b for b in self.binders if b.is_hypothesis]

    def variables(self) -> list[Binder]:
        return [b for b in self.binders if not b.is_hypothesis]

    def signature(self, name: str | None = None) -> str:
        parts = [self.kind, name or self.name]
        parts.extend(b.render() for b in self.binders)
        return " ".join(parts) + " :\n    " + self.conclusion.strip()

    def render(self, name: str | None = None, proof: str = "by sorry") -> str:
        """Lean text. The default proof is a placeholder used only in scratch
        files for elaboration checks; scratch files live outside lean_proofs/."""
        return f"{self.signature(name)} := {proof}\n"

    def identifiers(self) -> set[str]:
        text = " ".join(b.type for b in self.binders) + " " + self.conclusion
        names = {n for b in self.binders for n in b.names}
        return names | set(_IDENT.findall(text))


@dataclass
class Candidate:
    id: str
    family: str  # module the source statement came from
    transformation: str
    source: str  # module.name of the source statement
    params: dict[str, Any]
    statement: Statement
    epistemic: Epistemic = Epistemic.CONJECTURE
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def make(
        family: str,
        transformation: str,
        source: Statement,
        params: dict[str, Any],
        statement: Statement,
    ) -> Candidate:
        digest = hashlib.sha256(statement.signature("_").encode("utf-8")).hexdigest()
        return Candidate(
            id=digest[:16],
            family=family,
            transformation=transformation,
            source=f"{source.module}.{source.name}",
            params=params,
            statement=statement,
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["epistemic"] = self.epistemic.value
        record["kind"] = "candidate"
        record["lean"] = self.statement.signature()
        return record


@dataclass
class GateResult:
    candidate_id: str
    gate: str
    status: Status
    evidence: dict[str, Any]
    seconds: float

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": "gate",
            "candidate_id": self.candidate_id,
            "gate": self.gate,
            "status": self.status.value,
            "evidence": self.evidence,
            "seconds": self.seconds,
            "at": time.time(),
        }


class Ledger:
    """Append-only JSONL trajectory store. One line per event, never rewritten."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        with self.path.open(encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        return iter(json.loads(line) for line in lines)

    def sha256(self) -> str:
        if not self.path.exists():
            return ""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()
