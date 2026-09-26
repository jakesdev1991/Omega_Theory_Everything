"""Test the kernel-report gate, without pretending to run Lean in these fixtures."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_kernel import main, parse_axiom_report, run_checked, validate_targets


class KernelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke_mocked_main(self, side_effect, targets=None):
        """Exercise control flow only: the runner is mocked, never a Lean build."""
        (self.root / "kernel_audit_targets.json").write_text(
            json.dumps(["Fixture.result"] if targets is None else targets)
        )
        with patch("audit_kernel.shutil.which", return_value="/mock/lake"), patch(
            "audit_kernel.run_checked", side_effect=side_effect
        ) as runner, patch(
            "sys.argv", ["audit_kernel.py", "--root", str(self.root)]
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            code = main()
        return code, runner, output.getvalue()

    def test_foundational_axioms_and_axiom_free_declarations(self):
        report = (
            "'Model.first' depends on axioms: [propext,\n Classical.choice, Quot.sound]\n"
            "'Model.second' does not depend on any axioms\n"
        )
        parsed = parse_axiom_report(report, ["Model.first", "Model.second"])
        self.assertEqual(parsed["Model.second"], set())
        self.assertEqual(
            parsed["Model.first"], {"propext", "Classical.choice", "Quot.sound"}
        )

    def test_proof_hole_and_custom_axioms_fail(self):
        for axiom in ["sorryAx", "Model.physicalPostulate", "Lean.ofReduceBool"]:
            with self.subTest(axiom=axiom), self.assertRaisesRegex(
                ValueError, "Unapproved"
            ):
                parse_axiom_report(
                    f"'Model.result' depends on axioms: [{axiom}]", ["Model.result"]
                )

    def test_missing_and_truncated_reports_fail(self):
        for report in ["", "'Model.result' depends on axioms: [propext"]:
            with self.subTest(report=report), self.assertRaisesRegex(
                ValueError, "Missing"
            ):
                parse_axiom_report(report, ["Model.result"])

    def test_duplicate_report_fails(self):
        line = "'Model.result' does not depend on any axioms\n"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_axiom_report(line + line, ["Model.result"])

    def test_unrequested_report_fails(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            parse_axiom_report(
                "'Other.result' does not depend on any axioms", ["Model.result"]
            )

    def test_malformed_axiom_list_fails(self):
        with self.assertRaisesRegex(ValueError, "Malformed"):
            parse_axiom_report(
                "'Model.result' depends on axioms: [propext, ...]", ["Model.result"]
            )

    def test_targets_reject_empty_duplicates_and_code_injection(self):
        for names in [
            [],
            {},
            ["X", "X"],
            [1],
            ["X\naxiom bad : False"],
            ["X..Y"],
            ["X."],
        ]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_targets(names)

    def test_repository_target_inventory_is_valid(self):
        targets = json.loads(
            (Path(__file__).parent / "kernel_audit_targets.json").read_text()
        )
        self.assertEqual(validate_targets(targets), targets)

    def test_missing_lake_is_failure_not_skip_or_success(self):
        with patch("audit_kernel.shutil.which", return_value=None), patch(
            "sys.argv", ["audit_kernel.py"]
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 2)
        self.assertIn("no kernel audit was performed", output.getvalue())

    def test_failed_command_does_not_certify_plausible_output(self):
        result = subprocess.CompletedProcess(
            [], 1, "'X' does not depend on any axioms\n"
        )
        with patch(
            "audit_kernel.subprocess.run", return_value=result
        ), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            RuntimeError, "exited 1"
        ):
            run_checked(["lake", "env", "lean", "Report.lean"], Path("."), 1)

    def test_mocked_pipeline_builds_before_inspection_and_cleans_scratch(self):
        report_paths = []
        commands = []

        def runner(command, root, timeout):
            commands.append(command)
            self.assertEqual(root, self.root.resolve())
            self.assertGreater(timeout, 0)
            if len(commands) == 1:
                self.assertEqual(command[1:], ["build", "ToE"])
                self.assertFalse((root / ".lake").exists())
                return ""
            self.assertEqual(command[1:3], ["env", "lean"])
            report = Path(command[-1])
            report_paths.append(report)
            self.assertTrue(report.is_file())
            self.assertEqual(report.parent.parent, root / ".lake")
            text = report.read_text()
            self.assertTrue(text.startswith("import ToE\n"))
            self.assertIn("#print axioms Fixture.result\n", text)
            return "'Fixture.result' depends on axioms: [propext]"

        code, mock, output = self.invoke_mocked_main(runner)
        self.assertEqual(code, 0)
        self.assertEqual(mock.call_count, 2)
        self.assertIn("1 selected declarations passed", output)
        self.assertFalse(report_paths[0].exists())
        self.assertEqual(list((self.root / ".lake").iterdir()), [])

    def test_mocked_failed_build_never_inspects_stale_oleans(self):
        code, mock, output = self.invoke_mocked_main(RuntimeError("build failed"))
        self.assertEqual(code, 1)
        self.assertEqual(mock.call_count, 1)
        self.assertFalse((self.root / ".lake").exists())
        self.assertNotIn("declarations passed", output)

    def test_mocked_rejected_axiom_report_also_cleans_scratch(self):
        code, mock, output = self.invoke_mocked_main(
            ["", "'Fixture.result' depends on axioms: [Fixture.postulate]"]
        )
        self.assertEqual(code, 1)
        self.assertEqual(mock.call_count, 2)
        report = Path(mock.call_args.args[0][-1])
        self.assertFalse(report.exists())
        self.assertEqual(list((self.root / ".lake").iterdir()), [])
        self.assertIn("Unapproved", output)

    def test_mocked_timeout_is_failure(self):
        code, mock, output = self.invoke_mocked_main(
            subprocess.TimeoutExpired(["lake", "build", "ToE"], 1)
        )
        self.assertEqual(code, 1)
        self.assertEqual(mock.call_count, 1)
        self.assertIn("FAIL", output)

    def test_invalid_targets_do_not_launch_even_a_mock_build(self):
        code, mock, output = self.invoke_mocked_main(
            AssertionError("runner must not execute"), targets=["Bad..Name"]
        )
        self.assertEqual(code, 1)
        mock.assert_not_called()
        self.assertIn("FAIL", output)


if __name__ == "__main__":
    unittest.main()
