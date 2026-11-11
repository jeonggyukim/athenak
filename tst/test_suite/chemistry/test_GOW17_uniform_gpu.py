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
# control; its error at the default semi_implicit_cfl = 0.1 is about 1%.
big_step_rtol = {"kokkos_BDF": 1e-5, "semi_implicit": 2e-2}
cfl_dependence_rtol = {"kokkos_BDF": 5e-3, "semi_implicit": 2e-2}

# The time series is stored from a kokkos_BDF run and compared output by output;
# other solvers take a different number of cycles
time_series_ode_solvers = ["kokkos_BDF"]

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


def run_gow17_uniform_time_series(ode_solver):
    """Run the GOW17 uniform state test and run a regression test on the time series
    results."""
    try:
        cli_args = [
            f"chemistry/ode_solver={ode_solver}",
        ]

        # Run AthenaK
        input_file = "inputs/GOW17_uniform_time_series.athinput"
        results = testutils.run(input_file, cli_args)
        assert results, (
            f"GOW17 uniform time series test run failed for {ode_solver} solver."
        )

        # Get a list of all the files
        data_path = pathlib.Path("./tab/")
        files = sorted(data_path.glob("GOW17_uniform.hydro_w.*.tab"))

        # Collect all the data into a single array for easy comparison
        dataset_names = (
            "dens",
            "velx",
            "vely",
            "velz",
            "eint",
            "s_00_chem_He+",
            "s_01_chem_OHx",
            "s_02_chem_CHx",
            "s_03_chem_CO",
            "s_04_chem_C+",
            "s_05_chem_HCO+",
            "s_06_chem_H2",
            "s_07_chem_H+",
            "s_08_chem_H3+",
            "s_09_chem_H2+",
            "s_10_chem_O+",
            "s_11_chem_Si+",
        )
        test_time_series = np.empty(
            len(files), dtype=[(name, "f4") for name in dataset_names]
        )

        # Gather data
        for i, f in enumerate(files):
            data = athena_read.tab(f)
            for name in dataset_names:
                test_time_series[name][i] = data[name][0]

        # Load the fiducial data
        fiducial_time_series = np.load(
            "../../test_suite/chemistry/data/gow17_time_series.npz"
        )["time_series"]

        # Compare all results
        for name in dataset_names:
            assert np.allclose(test_time_series[name], fiducial_time_series[name]), (
                f"The {name} dataset contains incorrect value(s)."
            )

    finally:
        testutils.cleanup()


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


@pytest.mark.parametrize("ode_solver", time_series_ode_solvers)
def test_gow17_uniform_time_series_cpu(ode_solver):
    """CPU Test for GOW17 time series problem."""
    run_gow17_uniform_time_series(ode_solver)
