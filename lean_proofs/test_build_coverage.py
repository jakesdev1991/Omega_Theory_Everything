"""Keep every local proof module in the full build and the source-review inventory.

These are source wiring checks, not Lean elaboration or mathematical verification.
"""

import re
import unittest
from pathlib import Path

from lean_source import (
    find_import_cycle,
    lean_sources,
    mask_comments_and_strings,
    module_imports,
)

ROOT = Path(__file__).resolve().parent


class BuildCoverageTests(unittest.TestCase):
    def test_every_proof_module_is_a_lake_root(self):
        modules = {
            ".".join(p.relative_to(ROOT).with_suffix("").parts)
            for p in lean_sources(ROOT)
            if p.name != "lakefile.lean"
        }
        roots = set(
            re.findall(
                r"`([\w.]+)",
                mask_comments_and_strings((ROOT / "lakefile.lean").read_text()),
            )
        )
        self.assertEqual(modules, roots)

    def test_entrypoint_imports_every_other_proof_module(self):
        modules = {
            ".".join(p.relative_to(ROOT).with_suffix("").parts)
            for p in lean_sources(ROOT)
            if p.name not in {"lakefile.lean", "ToE.lean"}
        }
        imports = set(
            re.findall(
                r"^import ([\w.]+)$",
                mask_comments_and_strings((ROOT / "ToE.lean").read_text()),
                re.M,
            )
        )
        self.assertEqual(modules, imports)

    def test_every_source_has_an_audit_entry(self):
        report = (ROOT / "PROOF_AUDIT.md").read_text()
        for path in lean_sources(ROOT):
            with self.subTest(module=path.name):
                self.assertIn(f"`{path.relative_to(ROOT)}`", report)

    def test_plain_lake_build_has_a_default_target(self):
        self.assertRegex(
            (ROOT / "lakefile.lean").read_text(),
            r"@\[default_target\]\s+lean_lib ToE",
        )

    def test_local_import_graph_has_no_cycles(self):
        self.assertIsNone(find_import_cycle(module_imports(ROOT)))

    def test_imports_resolve_locally_or_to_known_external_packages(self):
        graph = module_imports(ROOT)
        # A new direct external dependency needs an explicit package-prefix review.
        external = {"Mathlib", "Lean", "Init", "Lake"}
        unresolved = {
            (module, dependency)
            for module, imports in graph.items()
            for dependency in imports
            if dependency not in graph and dependency.split(".")[0] not in external
        }
        self.assertEqual(unresolved, set())


if __name__ == "__main__":
    unittest.main()
