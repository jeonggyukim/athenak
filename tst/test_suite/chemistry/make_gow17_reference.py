"""
Rebuild data/gow17_reference.npz, the reference of the GOW17 reference test: each
condition of test_GOW17_uniform_gpu.check_time_conditions run with a converged Kokkos BDF
(rtol 1e-7, atol 1e-16) to each check time. Rerun it whenever the GOW17 rates, heating
or cooling change.

Usage, from tst/, with an AthenaK binary at tst/build/src/athena (as run_test_suite.py
builds it) and tst/inputs reachable as tst/build/src/inputs:

    python test_suite/chemistry/make_gow17_reference.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import test_suite.chemistry.test_GOW17_uniform_gpu as gow17  # noqa: E402


def main():
    os.chdir(os.path.join(HERE, "..", "..", "build", "src"))
    out = {}
    tight = ["chemistry/kokkos_BDF_rtol=1e-7", "chemistry/kokkos_BDF_atol=1e-16"]
    for name, (*_, times) in gow17.check_time_conditions.items():
        for t in times:
            s = gow17.check_time_state("kokkos_BDF", name, t, tight)
            out[f"{name}_t{t}"] = np.array(
                [s["eint"]] + [s[k] for k in gow17.check_time_species])
            print(name, t, "done")
    path = os.path.join(HERE, "data", "gow17_reference.npz")
    np.savez(path, **out)
    print("wrote", path)


if __name__ == "__main__":
    main()
