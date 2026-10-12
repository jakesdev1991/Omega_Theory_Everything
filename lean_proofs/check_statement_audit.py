#!/usr/bin/env python3
"""Phase 6 statement-level audit: every physics-sounding name has a recorded judgement.

`docs/research/CLAIM_TIGHTENING_ROADMAP.md` Phase 6 asks for a pass over the Lean
physics volumes that decides, per theorem, whether the name overstates the
statement, whether the hypotheses are satisfiable, and whether a non-degenerate
instance exists. Its exit gate is: every physics-volume theorem with a
physics-sounding name has a row in the audit, and the row is one of

* **witness** -- the statement is about a declared, non-degenerate object or
  instance (the row names it in `witness`; the witness is never the theorem
  itself), or
* **conditional** -- the statement carries its hypotheses in its own signature
  (the row quotes the literal text in `stated_condition`), or
* **renamed** -- the name was changed so it no longer overstates (the row names
  the name that carries the content now in `renamed_to`, and `previous_name`,
  the retired name, which must not survive as a declaration or as a live token
  anywhere in the corpus), or
* **carried** -- the statement is true only from data the model carries -- a
  structure field, a postulate, a placeholder or constant definition, a
  definitional restatement, or a degenerate instance -- so no non-degenerate
  witness exists. The row names the declaration that carries the data in
  `source`.

The fourth disposition is not in the roadmap's one-line sketch. Implementing the
sketch against the corpus showed that a three-way split has no honest place for
`theorem photon_massless : PhotonMass = 0 := rfl` or for `|0| <= |0|` in a
placeholder layer: forcing those rows to name a "witness" would report a
degenerate witness as a non-degenerate one. The audit file therefore records the
count it carries in a `carried_cap` field, and the gate fails when the number of
`carried` rows exceeds that cap, so the count is a ratchet that may only go down
(fix a row, lower the cap in the same commit). The one thing that does raise it
is adding rows: the Phase 6 rename pass renamed three declarations the lexicon
never matched and gave them rows, so the cap moved 30 -> 33 in that commit, in
plain sight, with no existing row reclassified.

The scratch sketch's third reading is preserved: a rename performed by an audit
pass lands under its new name with `previous_name` set, exactly like the renames
recorded for earlier passes. Because such a rename is orthogonal to the
statement's disposition, `previous_name` is validated on *any* row, not only on
`renamed` rows: the retired name must be a declaration nowhere and must not
survive as a live token in the corpus (comments are masked) or in any other JSON
in this directory. The Phase 6 rename pass (2026-10-12) renamed sixteen
declarations this way -- the six `zeroCurvature_*` restatements in Vol05, the
Vol02 postulate and Maxwell pair, the two gauge-group definitional equalities,
the placeholder optical identity, the Vol09 definitional entropy bridge, the two
legacy-constant Vol14 rows, the Vol34 gate restatement and the Vol37 diffusivity
ratio -- each one a name that asserted a result the statement does not contain.

This script is the gate. It is *lexical*, like the rest of this directory's
audits: it decides which declarations need a row from a documented lexicon of
physics words, and it checks the rows for shape and existence. It cannot decide
whether a judgement is correct -- satisfiability and non-degeneracy are
mathematical claims, and a `witness` row is checked to *exist*, not to entail
the theorem. What the kernel can check is checked by CI's `lake build ToE`,
which has to compile any witness declaration the audit names.

The lexicon is a floor, not a ceiling. It is deliberately generous (a false
positive costs one row; a false negative hides a claim), the rule is spelled
out below, and adding a word to it immediately demands rows for everything it
matches. A name matches when one of its words -- split on `_`, `.`, and
camelCase boundaries -- is a token in `PHYSICS_TOKENS`.

Usage:
    python3 check_statement_audit.py                 # gate
    python3 check_statement_audit.py --review        # what needs a row, with statements
    python3 check_statement_audit.py --list          # one line per row
Exit status: 0 when every candidate has a valid row and no row is invalid,
1 when something is missing or malformed, 2 when the audit file cannot be read.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from check_mathlib_names import MODIFIER  # noqa: E402  (one definition, not two)
from lean_source import mask_comments_and_strings  # noqa: E402

AUDIT_NAME = "statement_audit.json"
VOLUME_GLOB = "Vol*.lean"

# The word list that decides which declarations need a row. Kept in groups so a
# reviewer can see the rule rather than guess it, and built from the code lines
# only: a comment inside the literal would otherwise become a lexicon word.
PHYSICS_TOKENS = frozenset(
    " ".join(
        line.split("#", 1)[0]
        for line in """
    # quantities, processes and their standard names
    entropy energy temperature thermal heat mass force momentum pressure density
    expansion contraction curvature horizon redshift luminosity power flux charge
    current potential orbit period decay relaxation diffusion instability
    viscosity dissipation oscillation harmonic radiation wave spacetime spacelike
    causal lightcone geodesic
    # relativity, quantum theory, thermodynamics, information
    quantum photon gauge vacuum planck hawking bekenstein kardashev dyson
    gravitational gravity schwarzschild kretschmann ricci friedmann flrw
    desitter cosmological cosmology cosmic inflation inflaton singularity
    hamiltonian schrodinger robertson poisson weyl symplectic lagrangian
    kms gibbs boltzmann carnot clausius landauer thermodynamic decoherence
    entanglement decoherent qubit spin magnetic electric dirac optical
    commutator isotony microcausality graviton wheeler dewitt chern hall
    haldane landau higgs ryu takayanagi epr tsirelson chsh bell singlet
    partition frame novikov omega
    # structure of the physical world and the disciplines in the volume titles
    universe multiverse brane compactification boson fermion atom nuclear
    stellar galactic metric
    """.splitlines()
    ).split()
)

DISPOSITIONS = ("witness", "conditional", "renamed", "carried")
ROW_FIELDS = ("name", "file", "disposition", "note")
NOTE_MIN_LENGTH = 20
CARRY_CAP_FIELD = "carried_cap"

DECL_KEYWORDS = (
    "theorem",
    "lemma",
    "def",
    "abbrev",
    "structure",
    "class",
    "inductive",
    "instance",
    "axiom",
    "opaque",
)

DECL_RE = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\]\s*)*"
    + MODIFIER
    + r"(?:"
    + "|".join(DECL_KEYWORDS)
    + r")\s+([A-Za-z_][\w'.]*)",
    re.M,
)

# `theorem foo` / `private lemma bar` with the kind kept for the review output.
KINDED_DECL_RE = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\]\s*)*"
    + MODIFIER
    + r"("
    + "|".join(DECL_KEYWORDS)
    + r")\s+([A-Za-z_][\w'.]*)",
    re.M,
)

CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def words(name: str) -> list[str]:
    """Split a declaration name into lowercase words on `_`, `.` and case."""
    out: list[str] = []
    for part in re.split(r"[._]", name):
        out.extend(match.group(0).lower() for match in CAMEL_RE.finditer(part))
    return out


def physics_words(name: str) -> list[str]:
    """The lexicon tokens a declaration name carries, sorted."""
    return sorted(set(words(name)) & PHYSICS_TOKENS)


def _statement_end(masked: str, start: int) -> int:
    """Offset where a declaration's signature ends: the proof term or `where`."""
    depth = 0
    lines = 0
    i = start
    while i < len(masked):
        char = masked[i]
        if char == "\n":
            lines += 1
            if lines > 40:
                return i
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            if masked.startswith(":=", i):
                return i
            if masked.startswith("where", i) and not masked[i + 5 : i + 6].isalnum():
                return i
            if char == "\n" and DECL_RE.match(masked, i + 1):
                return i
        i += 1
    return len(masked)


def _docstring(raw: str, decl_start: int) -> str:
    """The `/-- ... -/` block attached to a declaration, whitespace-collapsed."""
    head = raw.rfind("/--", 0, decl_start)
    if head == -1:
        return ""
    close = raw.find("-/", head)
    if close == -1:
        return ""
    between = raw[close + 2 : decl_start]
    if re.sub(r"@\[[^\]]*\]", "", between).strip():
        return ""
    return re.sub(r"\s+", " ", raw[head + 3 : close]).strip()


class Declaration:
    """One declaration in the corpus, with the text this audit reasons about."""

    def __init__(
        self,
        file: str,
        kind: str,
        name: str,
        statement: str,
        docstring: str,
    ) -> None:
        self.file = file
        self.kind = kind
        self.name = name
        self.statement = statement
        self.docstring = docstring

    def matches(self, candidates: set[str]) -> bool:
        return self.name in candidates


def extract_declarations(path: pathlib.Path) -> list[Declaration]:
    """Every declaration in one file, with its signature and attached docstring."""
    raw = path.read_text(encoding="utf-8")
    masked = mask_comments_and_strings(raw)
    out: list[Declaration] = []
    for match in KINDED_DECL_RE.finditer(masked):
        kind, name = match.group(1), match.group(2)
        end = _statement_end(masked, match.end())
        statement = re.sub(r"\s+", " ", masked[match.start() : end]).strip()
        statement = re.sub(r"^(?:@\[[^\]]*\]\s*)*", "", statement)
        out.append(
            Declaration(
                path.name, kind, name, statement, _docstring(raw, match.start())
            )
        )
    return out


def corpus(root: pathlib.Path) -> list[Declaration]:
    """Every declaration in `lean_proofs`, dependencies excluded."""
    out: list[Declaration] = []
    for path in sorted(root.glob("*.lean")):
        out.extend(extract_declarations(path))
    return out


def candidates(root: pathlib.Path) -> list[Declaration]:
    """Theorems and lemmas in the volumes whose names carry a physics word."""
    out: list[Declaration] = []
    for path in sorted(root.glob(VOLUME_GLOB)):
        for decl in extract_declarations(path):
            if decl.kind in ("theorem", "lemma") and physics_words(decl.name):
                out.append(decl)
    return out


def load_audit(path: pathlib.Path) -> tuple[list[dict[str, object]], object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError(f"{path.name}: expected an object with a 'rows' list")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"{path.name}: 'rows' must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}: row {index} is not an object")
    if CARRY_CAP_FIELD not in payload:
        raise ValueError(
            f"{path.name}: missing the {CARRY_CAP_FIELD!r} ratchet "
            "(the number of `carried` rows the audit is allowed to contain)"
        )
    return rows, payload[CARRY_CAP_FIELD]


# A retired name is matched on identifier boundaries.  A new name that contains
# the old one -- `zeroCurvature_bianchi_identity` from `bianchi_identity`, or a
# deliberately kept law field like `law_metric_is_qfim` beside the retired
# `metric_is_qfim` -- is a different identifier and not a survivor.
RETIRED_TOKEN = "(?<![A-Za-z0-9_']){}(?![A-Za-z0-9_'])".format


def stale_uses(root: pathlib.Path, name: str) -> list[str]:
    """Files that still mention a name this pass renamed away, as a token."""
    pattern = re.compile(RETIRED_TOKEN(re.escape(name)))
    hits: list[str] = []
    for path in sorted(root.glob("*.lean")) + sorted(root.glob("*.json")):
        if path.name == AUDIT_NAME:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".lean":
            text = mask_comments_and_strings(text)
        if pattern.search(text):
            hits.append(path.name)
    return hits


def validate(
    rows: list[dict[str, object]],
    corpus_decls: list[Declaration],
    candidate_decls: list[Declaration],
    root: pathlib.Path,
) -> tuple[list[str], list[str]]:
    """Return (problems, candidates without a row)."""
    problems: list[str] = []
    by_name: dict[str, Declaration] = {decl.name: decl for decl in corpus_decls}
    by_file_name: dict[tuple[str, str], Declaration] = {
        (decl.file, decl.name): decl for decl in corpus_decls
    }
    seen: set[str] = set()
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"row without a name: {row!r}")
            continue
        label = f"row {name}"
        if name in seen:
            problems.append(f"{label}: duplicate row")
            continue
        seen.add(name)
        for field in ROW_FIELDS:
            if not isinstance(row.get(field), str) or not str(row.get(field)).strip():
                problems.append(f"{label}: missing or empty '{field}'")
        note = row.get("note")
        if isinstance(note, str) and len(note.strip()) < NOTE_MIN_LENGTH:
            problems.append(f"{label}: note is too short to record a judgement")
        disposition = row.get("disposition")
        if disposition not in DISPOSITIONS:
            problems.append(
                f"{label}: disposition {disposition!r} is not one of {DISPOSITIONS}"
            )
            continue
        decl = by_file_name.get((str(row.get("file")), name)) or by_name.get(name)
        if decl is None:
            problems.append(f"{label}: no such declaration in the corpus")
            continue
        if row.get("file") != decl.file:
            problems.append(
                f"{label}: declared in {decl.file}, row says {row.get('file')}"
            )
        if disposition == "witness":
            witness = row.get("witness")
            if not isinstance(witness, str) or not witness:
                problems.append(f"{label}: a witness row must name its witness")
            elif witness == name:
                problems.append(f"{label}: the witness is the theorem itself")
            elif witness not in by_name:
                problems.append(f"{label}: witness {witness} is not a declaration")
        elif disposition == "conditional":
            condition = row.get("stated_condition")
            if not isinstance(condition, str) or not condition.strip():
                problems.append(
                    f"{label}: a conditional row must quote stated_condition"
                )
            elif re.sub(r"\s+", " ", condition).strip() not in decl.statement:
                problems.append(
                    f"{label}: stated_condition {condition!r} does not occur in the statement"
                )
        elif disposition == "carried":
            source = row.get("source")
            if not isinstance(source, str) or not source:
                problems.append(f"{label}: a carried row must name its source")
            elif source == name:
                problems.append(f"{label}: the source is the declaration itself")
            elif source not in by_name:
                problems.append(f"{label}: source {source} is not a declaration")
        else:  # renamed
            target = row.get("renamed_to")
            if not isinstance(target, str) or target not in by_name:
                problems.append(f"{label}: renamed_to {target!r} is not a declaration")
        # A name retired by a rename pass is checked on every disposition: the
        # rename records what the declaration is now called, not what it proves.
        previous = row.get("previous_name")
        if isinstance(previous, str) and previous:
            if previous == name:
                problems.append(f"{label}: previous_name is the row's own name")
            elif previous in by_name:
                problems.append(f"{label}: previous_name {previous} still exists")
            else:
                hits = stale_uses(root, previous)
                if hits:
                    problems.append(
                        f"{label}: previous_name {previous} is still mentioned in "
                        + ", ".join(hits)
                    )
    missing = [decl.name for decl in candidate_decls if decl.name not in seen]
    return problems, missing


def summarize(rows: list[dict[str, object]]) -> str:
    counts = {disposition: 0 for disposition in DISPOSITIONS}
    for row in rows:
        disposition = row.get("disposition")
        if disposition in counts:
            counts[disposition] += 1
    return ", ".join(f"{key} {counts[key]}" for key in DISPOSITIONS)


def review(candidate_decls: list[Declaration], corpus_decls: list[Declaration]) -> None:
    """The review document: every candidate statement, per file, with neighbours."""
    by_file: dict[str, list[Declaration]] = {}
    others: dict[str, list[Declaration]] = {}
    for decl in candidate_decls:
        by_file.setdefault(decl.file, []).append(decl)
    for decl in corpus_decls:
        others.setdefault(decl.file, []).append(decl)
    for file in sorted(by_file):
        print(f"\n### {file} ({len(by_file[file])} candidate(s))")
        neighbours = [
            f"{decl.name} [{decl.kind}]"
            for decl in others[file]
            if decl.kind != "theorem" or physics_words(decl.name)
        ]
        print(f"neighbours: {', '.join(neighbours)}")
        for decl in by_file[file]:
            print(f"- {decl.name}  words={physics_words(decl.name)}")
            print(f"    stmt: {decl.statement}")
            if decl.docstring:
                print(f"    doc:  {decl.docstring[:220]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="the lean_proofs directory")
    parser.add_argument("--audit", default="", help=f"default {AUDIT_NAME}")
    parser.add_argument(
        "--max-missing",
        type=int,
        default=0,
        help="fail when more than this many candidates lack a row (default 0)",
    )
    parser.add_argument("--review", action="store_true", help="print what needs a row")
    parser.add_argument("--list", action="store_true", help="print one line per row")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.repo).resolve()
    corpus_decls = corpus(root)
    candidate_decls = candidates(root)
    print(
        f"{len(candidate_decls)} physics-sounding declaration(s) in "
        f"{len({decl.file for decl in candidate_decls})} volume(s) "
        f"({len(PHYSICS_TOKENS)} lexicon tokens, {len(corpus_decls)} declarations indexed)"
    )

    if args.review:
        review(candidate_decls, corpus_decls)
        return 0

    audit_path = pathlib.Path(args.audit) if args.audit else root / AUDIT_NAME
    try:
        rows, carried_cap = load_audit(audit_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    carried = sum(1 for row in rows if row.get("disposition") == "carried")
    retired = sum(
        1
        for row in rows
        if isinstance(row.get("previous_name"), str) and row["previous_name"]
    )
    print(
        f"{audit_path.name}: {len(rows)} row(s) -- {summarize(rows)}; "
        f"{retired} record a retired name"
    )
    print(
        f"{carried} carried row(s), cap {carried_cap}: "
        + (
            "within the ratchet"
            if isinstance(carried_cap, int) and carried <= carried_cap
            else "OVER the ratchet"
        )
    )
    if not isinstance(carried_cap, int) or carried > carried_cap:
        print(
            f"  {CARRY_CAP_FIELD} must be an integer and at least the current "
            f"count; fix the {carried} carried row(s) or the cap."
        )
    if args.list:
        for row in sorted(
            rows, key=lambda item: (str(item.get("file")), str(item.get("name")))
        ):
            print(
                f"  {row.get('disposition'):<11} {row.get('name')}  ({row.get('file')})"
            )
        return 0

    problems, missing = validate(rows, corpus_decls, candidate_decls, root)
    for problem in problems:
        print(f"  {problem}")
    exact = [
        row for row in rows if row.get("name") in {d.name for d in candidate_decls}
    ]
    print(
        f"{len(exact)} row(s) cover candidates, {len(rows) - len(exact)} are extra; "
        f"{len(missing)} candidate(s) lack a row, limit {args.max_missing}."
    )
    if missing:
        print("Missing rows (run --review for the statements):")
        for name in missing:
            print(f"  {name}")
    over_cap = not isinstance(carried_cap, int) or carried > carried_cap
    if problems or len(missing) > args.max_missing or over_cap:
        print(
            f"FAILED: {len(problems)} invalid row(s), {len(missing)} missing row(s) "
            f"(limit {args.max_missing}), {carried} carried row(s) "
            f"(cap {carried_cap})."
        )
        return 1
    print(
        "OK: every physics-sounding volume declaration has a recorded judgement. "
        "The gate checks that the rows exist and are well formed, not that they are right."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
