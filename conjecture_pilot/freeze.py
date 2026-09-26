# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Freeze the held-out families and scan for leakage (pre-registration §2).

Output is a JSON artifact with file hashes, the exact statement inventory,
upstream dependencies (allowed, recorded) and downstream restatements
(masked from every generator/retrieval input).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .extract import _LEAN_PROOFS, extract_statements

HELDOUT_MODULES: tuple[str, ...] = (
    "RadialDilation",
    "RadialMetric",
    "Vol16_NonEquilibriumThermodynamics",
    "Vol21_ArrowOfTime",
)
TRAIN_MODULES: tuple[str, ...] = (
    "InformationPhysics",
    "DynamicPlanckScale",
    "DynamicCODScale",
    "LogCorrelationMetric",
    "OmegaAxioms",
    "CBwK_Budget_Pacer",
    "APPA_Context_Branching",
)
# Never generator or retrieval inputs: they aggregate or restate everything.
MASKED_MODULES: tuple[str, ...] = ("OmegaProtocol", "ProofRegression", "ToE")

FAMILIES: dict[str, tuple[str, ...]] = {
    "radial": ("RadialDilation", "RadialMetric"),
    "entropy": ("Vol16_NonEquilibriumThermodynamics", "Vol21_ArrowOfTime"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - freeze must still work outside git
        return "unknown"


def _namespace_names(path: Path) -> list[str]:
    return re.findall(r"^\s*namespace\s+(\S+)", path.read_text(encoding="utf-8"), re.M)


def freeze(root: Path = _LEAN_PROOFS) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    names_by_module: dict[str, list[str]] = {}
    for module in HELDOUT_MODULES:
        path = root / f"{module}.lean"
        statements, unparsed = extract_statements(path, module)
        names_by_module[module] = [s.name for s in statements]
        inventory[module] = {
            "sha256": _sha256(path),
            "namespaces": _namespace_names(path),
            "imports": re.findall(
                r"^\s*import\s+(\S+)", path.read_text(encoding="utf-8"), re.M
            ),
            "statements": [
                {"name": s.name, "line": s.line, "hypotheses": len(s.hypotheses())}
                for s in statements
            ],
            "unparsed": [
                {"name": u.name, "line": u.line, "reason": u.reason} for u in unparsed
            ],
        }

    # Downstream leakage: any other module that imports or names held-out content.
    leakage: list[dict[str, Any]] = []
    heldout_namespaces = {
        ns for module in HELDOUT_MODULES for ns in inventory[module]["namespaces"]
    } | set(HELDOUT_MODULES)
    for path in sorted(root.glob("*.lean")):
        if path.stem in HELDOUT_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        hits: list[str] = []
        for module in HELDOUT_MODULES:
            if re.search(rf"^\s*import\s+{re.escape(module)}\b", text, re.M):
                hits.append(f"import {module}")
        for ns in sorted(heldout_namespaces):
            for m in re.finditer(rf"\b{re.escape(ns)}\.([A-Za-z_][\w']*)", text):
                hits.append(f"{ns}.{m.group(1)}")
        if hits:
            leakage.append({"module": path.stem, "references": sorted(set(hits))})

    upstream = sorted(
        {
            imp
            for module in HELDOUT_MODULES
            for imp in inventory[module]["imports"]
            if imp != "Mathlib"
        }
    )
    return {
        "frozen_at_commit": _git_head(root.parent),
        "heldout_modules": list(HELDOUT_MODULES),
        "train_modules": list(TRAIN_MODULES),
        "masked_modules": list(MASKED_MODULES),
        "families": {k: list(v) for k, v in FAMILIES.items()},
        "inventory": inventory,
        "upstream_dependencies_allowed": upstream,
        "downstream_leakage_masked": leakage,
        "policy": {
            "upstream": "Held-out statements may depend on training lemmas; the statements are unseen, their prerequisites are not.",
            "downstream": "Any module listed in downstream_leakage_masked is excluded from generator inputs and from retrieval corpora during evaluation.",
        },
    }


def write_freeze(out: Path, root: Path = _LEAN_PROOFS) -> dict[str, Any]:
    data = freeze(root)
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return data


def verify_freeze(frozen: dict[str, Any], root: Path = _LEAN_PROOFS) -> list[str]:
    """Return a list of held-out files whose hash no longer matches the freeze."""
    changed = []
    for module, entry in frozen["inventory"].items():
        if _sha256(root / f"{module}.lean") != entry["sha256"]:
            changed.append(module)
    return changed
