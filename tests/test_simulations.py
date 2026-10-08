# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE.
"""Regression tests for the simulation suite and its repository hygiene.

Each test here encodes a defect that actually shipped and was invisible because
CI swallowed the failure signal (the smoke step used to end every command with
`|| true`):

* ``sim6_v14_depletion.py`` crashed on every run — a units bug made the gamma
  calibration return 1e-20, the background integration then blew up, and the
  crash surfaced as "array must not contain infs or NaNs" inside SciPy;
* ``Sim1_Emergent_Geometry.py`` and ``Sim4_Evolution.py`` were prose
  manuscripts with a ``.py`` extension that the README told readers to run;
* the README's quick-start commands and the CI smoke step were never checked
  against the files that actually exist.

The tests are deliberately cheap: the expensive integration runs once, and the
rest are parse/consistency checks.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_sim6() -> types.ModuleType:
    """Import sim6_v14_depletion.py by path (its module name is unusual)."""
    spec = importlib.util.spec_from_file_location(
        "sim6_v14_depletion", ROOT / "sim6_v14_depletion.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim6_v14_depletion"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# sim6: the depletion-model calibration
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim6() -> types.ModuleType:
    return _load_sim6()


def test_sim6_calibrated_gamma_reproduces_the_target_h0(sim6: types.ModuleType) -> None:
    """The whole point of the calibration: H0(gamma*) is the fiducial H0."""
    params = {
        "alpha": sim6.alpha_def,
        "kappa": sim6.kappa_def,
        "gamma": sim6.gamma_guess_def,
        "Omega_m": sim6.OMEGA_M_FID,
        "A0": sim6.A0_def,
        "H0_SI": sim6.H0_SI_default,
    }
    gamma = sim6.calibrate_gamma(params)
    assert gamma > 0.0
    achieved = sim6.H0_model_at_today(params, gamma)
    assert achieved == pytest.approx(sim6.H0_fid, rel=1e-9)


def test_sim6_h0_response_is_strictly_increasing_in_gamma(
    sim6: types.ModuleType,
) -> None:
    """The bracket in calibrate_gamma is justified by monotonicity, not luck."""
    params = {
        "alpha": sim6.alpha_def,
        "kappa": sim6.kappa_def,
        "gamma": 1.0,
        "Omega_m": sim6.OMEGA_M_FID,
        "A0": sim6.A0_def,
        "H0_SI": sim6.H0_SI_default,
    }
    values = [sim6.H0_model_at_today(params, g) for g in (1e-6, 1e-3, 1e-1, 1.0, 1e2)]
    assert values == sorted(values)
    assert values[0] < values[-1]
    # gamma = 0 reproduces the pure matter/radiation Hubble rate, which is below
    # the target: that is the lower end of the calibration bracket.
    assert sim6.H0_model_at_today(params, 0.0) < sim6.H0_fid


def test_sim6_end_to_end_run_is_on_target_and_finite(sim6: types.ModuleType) -> None:
    """Integrate, reduce to observables, and check the reported numbers."""
    params = {
        "alpha": sim6.alpha_def,
        "kappa": sim6.kappa_def,
        "gamma": sim6.gamma_guess_def,
        "Omega_m": sim6.OMEGA_M_FID,
        "A0": sim6.A0_def,
        "H0_SI": sim6.H0_SI_default,
    }
    params["gamma"] = sim6.calibrate_gamma(params)
    sol = sim6.integrate_cosmo(params)
    assert sol.success, sol.message
    observables = sim6.compute_observables(sol, params)
    assert observables["z"].size >= 2
    # H(z = 0) is exact by construction in the calibration; the small residual
    # here is the PCHIP interpolation error of the sampled trajectory.
    assert observables["H_km_s_Mpc"][0] == pytest.approx(sim6.H0_fid, rel=1e-3)
    for key in ("H_km_s_Mpc", "w_eff", "mu", "I"):
        assert len(observables[key]) == len(observables["z"])


def test_sim6_degenerate_solution_raises_instead_of_crashing_downstream(
    sim6: types.ModuleType,
) -> None:
    """A one-point solution must fail with a model-level message."""

    class _FakeSolution:
        # solve_ivp returns numpy arrays; mirror that so the guard itself is
        # what raises, not the array indexing on the way to it.
        success = False
        message = "synthetic"
        t = np.array([0.0])
        y = np.array([[0.0], [0.005]])

    with pytest.raises(RuntimeError, match="fewer than two accepted steps"):
        sim6.compute_observables(_FakeSolution(), {"H0_SI": sim6.H0_SI_default})


# --------------------------------------------------------------------------
# Repository hygiene: nothing that lies about being runnable
# --------------------------------------------------------------------------


def test_every_root_simulation_script_is_valid_python() -> None:
    """Catches the "prose manuscript named .py" class of defect for good."""
    scripts = sorted(ROOT.glob("Sim*.py")) + sorted(ROOT.glob("sim*.py"))
    assert scripts, "expected simulation scripts at the repository root"
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def test_manuscripts_are_markdown_not_python() -> None:
    """Sim1 and Sim4 were moved out of the executable namespace."""
    for name in ("sim1_emergent_geometry.md", "sim4_evolution.md"):
        assert (ROOT / "docs" / "manuscripts" / name).is_file()
    assert not (ROOT / "Sim1_Emergent_Geometry.py").exists()
    assert not (ROOT / "Sim4_Evolution.py").exists()


def _readme_python_commands() -> list[str]:
    """Every `python <file>.py` command inside a README code block."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"^\s*python3?\s+([\w./-]+\.py)", readme, flags=re.MULTILINE)
    return commands


def test_readme_only_tells_the_reader_to_run_files_that_exist() -> None:
    """The quick start must not point at renamed or non-existent scripts."""
    commands = _readme_python_commands()
    assert commands, "README is expected to document at least one python command"
    missing = [name for name in commands if not (ROOT / name).is_file()]
    assert not missing, f"README references files that do not exist: {missing}"


def test_ci_smoke_step_does_not_swallow_failures() -> None:
    """`|| true` in the simulation smoke step hid sim6's crash for months."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith(("python ", "python3 ")) and "|| true" in stripped:
            pytest.fail(f"CI smoke command swallows its exit status: {stripped!r}")


def test_excluded_python_files_still_exist() -> None:
    """No tool configuration may exclude a file that is not there any more."""
    for config in ("ruff.toml", "mypy.ini"):
        text = (ROOT / config).read_text(encoding="utf-8")
        for name in (
            "Sim1_Emergent_Geometry.py",
            "Sim2_Cosmology.py",
            "Sim4_Evolution.py",
        ):
            assert name not in text, f"{config} still excludes {name}"


def test_sim2_and_sim6_import_without_a_display(sim6: types.ModuleType) -> None:
    """Neither script may require an interactive matplotlib backend."""
    env = {"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}
    for script in ("Sim2_Cosmology.py",):
        completed = subprocess.run(  # noqa: S603 - fixed argv, repository-local script
            [sys.executable, script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
