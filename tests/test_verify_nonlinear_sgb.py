"""Tests for research/verify_nonlinear_sgb.py — nonlinear SGB structural checks."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, '..', 'research', 'verify_nonlinear_sgb.py')


def run_script():
    r = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout


def test_all_structural_checks_pass():
    rc, out = run_script()
    assert rc == 0, out
    for tag in ['[1]', '[2]', '[3]', '[4]', '[5]']:
        line = next(ln for ln in out.split('\n') if ln.startswith(tag))
        assert 'False' not in line, line
    assert '[2]' in out and 'G(Schw) = 48M²/r⁶: True' in out


def test_horizon_regularity():
    rc, out = run_script()
    assert '[4] φ\'(1) = -3λφ(1): True' in out


def test_kretschmann_and_gauss_bonnet():
    rc, out = run_script()
    assert '[1] K(Schw) = 48M²/r⁶: True' in out
    assert 'G(Schw) = 48M²/r⁶: True' in out
