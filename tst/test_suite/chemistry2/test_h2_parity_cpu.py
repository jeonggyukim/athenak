"""
chemistry2 against chemistry: the H2 uniform and H2 advection problems, run with the
semi-implicit solver under <chemistry> and under <chemistry2> from the same input file,
give the same primitive variables at every output.

The two modules differ only in rounding (chemistry2 multiplies by inverse unit factors);
measured 1.3e-15 (uniform) and 1.7e-13 (advection) relative to t = 5 Myr, 2026-11-20
(chemistry2 integrates each step from 0, src/chemistry from the mesh time).
"""

import glob
import os

import numpy as np
import pytest
from test_suite import testutils

problems = ["uniform", "advection"]
rtol = 1.0e-12


def chemistry2_input(input_file, out_file):
    """Copy an input file with its <chemistry> block renamed <chemistry2>."""
    with open(input_file) as f:
        text = f.read()
    with open(out_file, "w") as f:
        f.write(text.replace("<chemistry>", "<chemistry2>"))


def max_relative_difference(files_a, files_b):
    worst = 0.0
    for a, b in zip(files_a, files_b):
        A = np.loadtxt(a)
        B = np.loadtxt(b)
        assert A.shape == B.shape, f"{a} and {b} differ in shape"
        scale = np.where(A != 0.0, np.abs(A), 1.0)
        worst = max(worst, float(np.max(np.abs(A - B) / scale)))
    return worst


@pytest.mark.parametrize("problem", problems)
def test_h2_parity_cpu(problem):
    input_file = f"inputs/H2_{problem}_test.athinput"
    c2_file = f"H2_{problem}_test_chemistry2.athinput"
    chemistry2_input(input_file, c2_file)
    common = ["output1/data_format=%.17e", "time/tlim=5"]
    try:
        testutils.run(input_file, ["chemistry/ode_solver=semi_implicit",
                                   f"job/basename=old_{problem}"] + common)
        testutils.run(c2_file, ["chemistry2/ode_solver=semi_implicit",
                                f"job/basename=new_{problem}"] + common)
        old = sorted(glob.glob(f"tab/old_{problem}.*.tab"))
        new = sorted(glob.glob(f"tab/new_{problem}.*.tab"))
        assert len(old) > 0 and len(old) == len(new), "outputs missing"
        worst = max_relative_difference(old, new)
        assert worst <= rtol, f"chemistry2 differs from chemistry by {worst:.2e}"
    finally:
        testutils.cleanup()
        if os.path.exists(c2_file):
            os.remove(c2_file)
