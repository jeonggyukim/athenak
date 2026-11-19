"""
chemistry2 GOW17 (rate table) against a converged reference at fixed times: four
conditions, each run from t = 0 to each check time (tlim is hit exactly), the final state
compared with a converged Kokkos BDF run of src/chemistry (rtol 1e-7, atol 1e-16) stored
in data/gow17_reference.npz. The conditions, check times and error measure are those of
the src/chemistry reference test (test_suite/chemistry, make_gow17_reference.py).

The error of a species is |x - x_ref| / max(x_ref, 1e-3 x_elem), of the energy
|e/e_ref - 1|; each tolerance is about twice the error measured on the Mac CPU.
"""

import os
import pathlib

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils

reference_input = "inputs/GOW17_reference_chemistry2.athinput"
reference_file = "../../test_suite/chemistry2/data/gow17_reference.npz"

_atomic = {"default": 1e-12, "H2": 1e-6, "CO": 1e-12, "C+": 1.6e-4, "H+": 1e-4,
           "Si+": 1.7e-6}
_molecular = {"default": 1e-12, "H2": 0.4999, "CO": 1.6e-4, "C+": 1e-12, "H+": 1e-12,
              "Si+": 1e-12}

# name: (n_H [cm^-3], xi_CR [s^-1], G0 [Draine], initial abundances, check times [Myr])
conditions = {
    "uniform_1e-6": (109.21, 2e-16, 1e-6, None, ["0.1", "0.4", "3", "30"]),
    "molecular_uv": (1e2, 2e-16, 1.0, _molecular, ["1e-4", "1e-3", "0.01", "0.1"]),
    "atomic_dense": (1e4, 2e-16, 1e-6, _atomic, ["0.02", "0.06", "0.15", "1"]),
    "atomic_high_cr": (1e3, 2e-15, 1e-6, _atomic, ["0.5", "0.8", "1.1", "5"]),
}

tolerance = {
    "kokkos_BDF": {
        "uniform_1e-6": [0.06, 0.015, 0.01, 0.01],
        "molecular_uv": [0.33, 0.3, 0.01, 0.01],
        "atomic_dense": [0.04, 0.01, 0.02, 0.01],
        "atomic_high_cr": [0.04, 0.05, 0.05, 0.01],
    },
    "semi_implicit": {
        "uniform_1e-6": [0.13, 0.08, 0.07, 0.01],
        "molecular_uv": [0.3, 0.5, 0.08, 0.01],
        "atomic_dense": [0.06, 0.06, 0.2, 0.01],
        "atomic_high_cr": [0.2, 0.35, 0.25, 0.01],
    },
}

species = {"He+": 0.1, "OHx": 3.2e-4, "CHx": 1.6e-4, "CO": 1.6e-4, "C+": 1.6e-4,
           "HCO+": 1.6e-4, "H2": 1.0, "H+": 1.0, "H3+": 1.0, "H2+": 1.0, "O+": 3.2e-4,
           "Si+": 1.7e-6}


def run_to(ode_solver, name, check_time):
    """Final internal energy and abundances of one condition at one check time."""
    n_h, xi_cr, g0, init, _ = conditions[name]
    args = [f"chemistry2/ode_solver={ode_solver}", f"time/tlim={check_time}",
            f"problem/n_H={n_h}", f"chemistry2/radiation_CR={xi_cr}",
            f"chemistry2/radiation_G0={g0}"]
    if init is not None:
        args.append(f"problem/init_default={init['default']}")
        args += [f"problem/init_{k}={v}" for k, v in init.items() if k != "default"]
    try:
        assert testutils.run(reference_input, args), (
            f"run failed: {ode_solver}, {name}, t = {check_time}")
        last = sorted(pathlib.Path("./tab/").glob("GOW17_uniform.hydro_w.*.tab"))[-1]
        data = athena_read.tab(last)
        state = [data["eint"][0]]
        for i, k in enumerate(species):
            state.append(data[f"s_{i:02d}_chem_{k}"][0])
        return np.array(state)
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("ode_solver", ["kokkos_BDF", "semi_implicit"])
def test_gow17_reference_cpu(ode_solver):
    reference = np.load(reference_file)
    report = []
    for name, (*_, times) in conditions.items():
        for i, t in enumerate(times):
            s = run_to(ode_solver, name, t)
            ref = reference[f"{name}_t{t}"]
            errors = [abs(s[0] / ref[0] - 1)]
            for j, x_elem in enumerate(species.values()):
                errors.append(abs(s[j + 1] - ref[j + 1]) / max(ref[j + 1], 1e-3 * x_elem))
            worst = int(np.argmax(errors))
            label = "eint" if worst == 0 else list(species)[worst - 1]
            report.append((name, t, label, errors[worst],
                           tolerance[ode_solver][name][i]))
    lines = [f"{ode_solver} {r[0]} t={r[1]}: worst {r[2]} {r[3]:.3g} (tol {r[4]})"
             for r in report]
    if os.environ.get("GOW17_REFERENCE_REPORT"):  # file to append the measured errors to
        with open(os.environ["GOW17_REFERENCE_REPORT"], "a") as f:
            f.write("\n".join(lines) + "\n")
    failed = [r for r in report if r[3] > r[4]]
    assert not failed, f"{len(failed)} check times exceed their tolerance: {failed}"
