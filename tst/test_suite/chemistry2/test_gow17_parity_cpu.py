"""
chemistry2 against chemistry: the GOW17 uniform problem with the semi-implicit solver
under <chemistry> and under <chemistry2>, one substep per hydro step
(semi_implicit_nsub_max = 1), gives the same primitive variables at every output.

One substep fixes the step sequence. With free substeps, rounding changes the step
sequence and the result by up to a few percent over 1e4 Myr in both modules alike, so
the comparison would not detect a porting error. Measured 1.3e-10 relative over 11
outputs to 1e4 Myr, 2026-11-20.
"""

import glob
import os
import re

import numpy as np
from test_suite import testutils

input_file = "inputs/GOW17_uniform_test.athinput"
rtol = 1.0e-8


def write_input(block, out_file):
    """The GOW17 uniform input with the semi-implicit solver, one substep, in `block`."""
    with open(input_file) as f:
        text = f.read()
    text = re.sub(r"(ode_solver\s*=\s*)\S+",
                  r"\1semi_implicit\nsemi_implicit_nsub_max = 1", text, count=1)
    if block == "chemistry2":  # the analytic fits; plain GOW17 reads the rate table
        # and without the chemistry2 additions not in src/chemistry: He I collisional
        # ionization, hot gas
        text = re.sub(r"(network\s*=\s*)GOW17",
                      r"\1GOW17_analytic\nGOW17_he_coll_ion = false\n"
                      r"GOW17_hot_cooling = false", text, count=1)
    with open(out_file, "w") as f:
        f.write(text.replace("<chemistry>", "<" + block + ">"))


def test_gow17_parity_cpu():
    old_file, new_file = "GOW17_parity_old.athinput", "GOW17_parity_new.athinput"
    write_input("chemistry", old_file)
    write_input("chemistry2", new_file)
    common = ["output1/data_format=%.17e"]
    try:
        testutils.run(old_file, ["job/basename=old_gow17"] + common)
        testutils.run(new_file, ["job/basename=new_gow17"] + common)
        old = sorted(glob.glob("tab/old_gow17.*.tab"))
        new = sorted(glob.glob("tab/new_gow17.*.tab"))
        assert len(old) > 0 and len(old) == len(new), "outputs missing"
        worst = 0.0
        for a, b in zip(old, new):
            A, B = np.loadtxt(a), np.loadtxt(b)
            scale = np.where(A != 0.0, np.abs(A), 1.0)
            worst = max(worst, float(np.max(np.abs(A - B) / scale)))
        assert worst <= rtol, f"chemistry2 differs from chemistry by {worst:.2e}"
    finally:
        testutils.cleanup()
        for f in (old_file, new_file):
            if os.path.exists(f):
                os.remove(f)
