"""Tests for the nonlinear-hair negative-result record."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(REPO, 'research', 'nonlinear_hair_negative_result.py')


def test_script_runs_and_reports_done():
    r = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, timeout=600)
    assert 'NEGATIVE-RESULT SCAN DONE' in r.stdout
    # regular small-amplitude shoots must all have A > 0
    for line in r.stdout.splitlines():
        if 'regular, A>0' in line:
            a = float(line.split('A=')[1].split()[0])
            assert a > 0


def test_threshold_constant_matches_linear_solver():
    sys.path.insert(0, os.path.join(REPO, 'research'))
    import nonlinear_hair_negative_result as nr
    assert abs(nr.LAM1 - 0.7256091671) < 1e-6
