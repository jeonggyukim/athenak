"""
Test for chemistry using the GOW17 network with uniform initial conditions. Runs tests for
different ODE solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import pathlib
import numpy as np

ode_solvers = ["kokkos_BDF", "semi_implicit"]

# Relative tolerances of the comparisons. The semi-implicit solver has no error
# control; its error at the default semi_implicit_dt_sub_frac = 0.1 is about 1%.
big_step_rtol = {"kokkos_BDF": 1e-5, "semi_implicit": 2e-2}
cfl_dependence_rtol = {"kokkos_BDF": 5e-3, "semi_implicit": 2e-2}

# These come from Athena++ at t=1e6. Equilibrium is reached somewhere around t=1e2 or 1e3
# but we're running the test longer to let any potential small errors compound into
# something easier to detect.
fiducial_data = {
    "time": 1e4,
    "cycle": 28,
    "dens": 109.20999908447266,
    "velx": 0.00000e00,
    "vely": 0.00000e00,
    "velz": 0.00000e00,
    "eint": 28.234391212463375,  # derived from pressure = 18.822927474975586
    "s_00_chem_He+": 4.4769834062208247e-07,
    "s_01_chem_OHx": 1.4335562809719704e-05,
    "s_02_chem_CHx": 6.079905467970548e-09,
    "s_03_chem_CO": 0.00013345545448828489,
    "s_04_chem_C+": 3.4978197049895243e-07,
    "s_05_chem_HCO+": 6.033158683749207e-08,
    "s_06_chem_H2": 0.44989413022994995,
    "s_07_chem_H+": 2.9430591439449927e-06,
    "s_08_chem_H3+": 1.2544340961540001e-06,
    "s_09_chem_H2+": 1.969339358254274e-09,
    "s_10_chem_O+": 8.677133317425145e-11,
    "s_11_chem_Si+": 6.625605806220847e-07,
}


def run_gow17_uniform_big_step(ode_solver, mpi=False):
    """Run the GOW17 uniform state test and compare to the known good results from
    AthenaK. Parameterized over the different ODE solvers that work for this network. This
    function is called by both the CPU and GPU tests."""
    if mpi:
        RUN = testutils.mpi_run
        fiducial_size = 12
        fiducial_data["cycle"] = 80
    else:
        RUN = testutils.run
        fiducial_size = 4

    try:
        cli_args = [
            f"chemistry/ode_solver={ode_solver}",
            f"mesh/nx1={fiducial_size}",
            f"mesh/nx2={fiducial_size}",
            f"mesh/nx3={fiducial_size}",
        ]
        input_file = "inputs/GOW17_uniform_test.athinput"
        results = RUN(input_file, cli_args)
        assert results, f"GOW17 uniform test run failed for {ode_solver} solver."

        data_path = pathlib.Path("./tab/GOW17_uniform.hydro_w.00010.tab")

        assert data_path.exists(), f"Output file not found at {data_path}"

        # Load the data
        test_data = athena_read.tab(data_path)

        for key in test_data:
            if key in ["i", "x1v"]:
                pass
            elif key == "time":
                assert test_data[key] == fiducial_data[key], (
                    f"Final time was not correct. Expected {fiducial_data[key]}, got "
                    "{test_data[key]}"
                )
            elif key == "cycle":
                assert test_data[key] == fiducial_data[key], (
                    "The number of time steps is not correct. Expected "
                    f"{fiducial_data[key]} but found {test_data[key]}."
                )
            else:
                test_arr = test_data[key]
                fiducial_arr = fiducial_data[key]
                assert test_arr.size == fiducial_size, (
                    f"The {key} dataset has the wrong size."
                )
                assert np.allclose(
                    test_arr, fiducial_arr, atol=0, rtol=big_step_rtol[ode_solver]
                ), (
                    f"The {key} dataset contains incorrect value(s)."
                )
    finally:
        testutils.cleanup()


# ----- Reference test -----
# Each condition is run from t = 0 to each of its check times (tlim is hit exactly) and
# the final state is compared with a converged Kokkos BDF run (rtol 1e-7, atol 1e-16)
# stored in data/gow17_reference.npz; make_gow17_reference.py rebuilds it.
reference_input = "inputs/GOW17_reference.athinput"
reference_file = "../../test_suite/chemistry/data/gow17_reference.npz"

# Initial abundances of the two physical starts; everything not listed is 1e-12
_atomic = {"default": 1e-12, "H2": 1e-6, "CO": 1e-12, "C+": 1.6e-4, "H+": 1e-4,
           "Si+": 1.7e-6}
_molecular = {"default": 1e-12, "H2": 0.4999, "CO": 1.6e-4, "C+": 1e-12, "H+": 1e-12,
              "Si+": 1e-12}

# name: (n_H [cm^-3], xi_CR [s^-1], G0 [Draine], initial abundances, check times [Myr])
check_time_conditions = {
    # all species 1e-6: far from equilibrium in every species at once
    "uniform_1e-6": (109.21, 2e-16, 1e-6, None, ["0.1", "0.4", "3", "30"]),
    # molecular gas suddenly exposed to UV: H2 and CO photodissociated within the first
    # hydro step, with photodissociation heating
    "molecular_uv": (1e2, 2e-16, 1.0, _molecular, ["1e-4", "1e-3", "0.01", "0.1"]),
    # dense shielded gas forming H2 and CO within 0.2 Myr
    "atomic_dense": (1e4, 2e-16, 1e-6, _atomic, ["0.02", "0.06", "0.15", "1"]),
    # CO chemistry driven by a high cosmic-ray ionization rate
    "atomic_high_cr": (1e3, 2e-15, 1e-6, _atomic, ["0.5", "0.8", "1.1", "5"]),
}

# Largest allowed error at each check time, about twice the measured one
check_time_tolerance = {
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

# Species compared, and the abundance per H of the element each one carries; the error
# of a species is |x - x_ref| / max(x_ref, 1e-3 x_elem), so trace species do not set it
check_time_species = {"He+": 0.1, "OHx": 3.2e-4, "CHx": 1.6e-4, "CO": 1.6e-4,
                      "C+": 1.6e-4, "HCO+": 1.6e-4, "H2": 1.0, "H+": 1.0, "H3+": 1.0,
                      "H2+": 1.0, "O+": 3.2e-4, "Si+": 1.7e-6}


def check_time_args(name, check_time, ode_solver):
    """Command-line arguments for one check-time run."""
    n_h, xi_cr, g0, init, _ = check_time_conditions[name]
    args = [f"chemistry/ode_solver={ode_solver}", f"time/tlim={check_time}",
            f"problem/n_H={n_h}", f"chemistry/radiation_CR={xi_cr}",
            f"chemistry/radiation_G0={g0}"]
    if init is not None:
        args.append(f"problem/init_default={init['default']}")
        args += [f"problem/init_{k}={v}" for k, v in init.items() if k != "default"]
    return args


def check_time_state(ode_solver, name, check_time, extra_args=()):
    """Run one condition to one check time; return its final internal energy and
    abundances."""
    try:
        args = check_time_args(name, check_time, ode_solver) + list(extra_args)
        assert testutils.run(reference_input, args), (
            f"GOW17 reference run failed: {ode_solver}, {name}, t = {check_time}"
        )
        last = sorted(pathlib.Path("./tab/").glob("GOW17_uniform.hydro_w.*.tab"))[-1]
        data = athena_read.tab(last)
        state = {"eint": data["eint"][0]}
        for k in check_time_species:
            state[k] = data[f"s_{list(check_time_species).index(k):02d}_chem_{k}"][0]
        return state
    finally:
        testutils.cleanup()


def run_gow17_reference(ode_solver):
    """Compare each condition at each check time with the converged reference."""
    reference = np.load(reference_file)
    for name, (*_, times) in check_time_conditions.items():
        for i, t in enumerate(times):
            state = check_time_state(ode_solver, name, t)
            ref = reference[f"{name}_t{t}"]
            errors = {"eint": abs(state["eint"] / ref[0] - 1)}
            for j, (k, x_elem) in enumerate(check_time_species.items()):
                errors[k] = abs(state[k] - ref[j + 1]) / max(ref[j + 1], 1e-3 * x_elem)
            worst = max(errors, key=errors.get)
            tol = check_time_tolerance[ode_solver][name][i]
            assert errors[worst] <= tol, (
                f"{ode_solver}, {name}, t = {t} Myr: {worst} error {errors[worst]:.3g} "
                f"exceeds {tol}"
            )


def run_gow17_cfl_dependence(ode_solver, mpi=False):
    """Run the GOW17 uniform state test and check if the results depend on the CFL
    number"""
    if mpi:
        RUN = testutils.mpi_run
        fiducial_size = 12
        fiducial_data["cycle"] = 80
    else:
        RUN = testutils.run
        fiducial_size = 4

    try:
        cli_args = [
            f"chemistry/ode_solver={ode_solver}",
            f"mesh/nx1={fiducial_size}",
            f"mesh/nx2={fiducial_size}",
            f"mesh/nx3={fiducial_size}",
            "mesh/x1min=-0.01",
            "mesh/x1max=0.01",
            "mesh/x2min=-0.01",
            "mesh/x2max=0.01",
            "mesh/x3min=-0.01",
            "mesh/x3max=0.01",
            "mesh/ix3_bc=outflow",
            "mesh/ox3_bc=outflow",
            "time/tlim=0.01",
        ]
        low_cfl = 0.01
        high_cfl = 0.8
        data_path = pathlib.Path("./tab/GOW17_uniform.hydro_w.00001.tab")

        # Run for the low CFL number
        input_file = "inputs/GOW17_uniform_test.athinput"
        results = RUN(input_file, cli_args + [f"time/cfl_number={low_cfl}"])
        assert results, (
            f"GOW17 uniform test run failed for {ode_solver} solver and CFL = {low_cfl}"
            " in CFL test."
        )
        low_cfl_data = athena_read.tab(data_path)

        # Run for the high CFL number
        results = RUN(input_file, cli_args + [f"time/cfl_number={high_cfl}"])
        assert results, (
            f"GOW17 uniform test run failed for {ode_solver} solver and CFL = {high_cfl}"
            " in CFL test."
        )
        high_cfl_data = athena_read.tab(data_path)

        # Now check for correct results
        ignore_list = ("i", "x1v", "time", "cycle")
        for key in low_cfl_data:
            if key in ignore_list:
                continue

            # Tolerances set to account for the tolerances given to the ODE solver
            assert np.allclose(
                low_cfl_data[key],
                high_cfl_data[key],
                atol=5e-6,
                rtol=cfl_dependence_rtol[ode_solver],
            ), f"The {key} datasets don't match with different CFL numbers"
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("ode_solver", ode_solvers)
def test_gow17_uniform_gpu(ode_solver):
    """GPU Test for GOW17 uniform test problem."""
    run_gow17_uniform_big_step(ode_solver)


@pytest.mark.parametrize("ode_solver", ode_solvers)
def test_gow17_cfl_dependence_cpu(ode_solver):
    """CPU Test for if the results of network depend on the CFL number"""
    run_gow17_cfl_dependence(ode_solver)


@pytest.mark.parametrize("ode_solver", ode_solvers)
def test_gow17_reference_gpu(ode_solver):
    """GPU Test for GOW17 abundances at fixed times against a converged reference."""
    run_gow17_reference(ode_solver)
