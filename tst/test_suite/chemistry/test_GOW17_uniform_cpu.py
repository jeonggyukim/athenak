"""
Test for chemistry using the GOW17 network with uniform initial conditions. Runs tests for
different ODE solvers
"""

# Modules
import pytest
import test_suite.chemistry.test_GOW17_uniform_gpu as gow17_uniform


@pytest.mark.parametrize("ode_solver", gow17_uniform.ode_solvers)
def test_gow17_uniform_big_step_cpu(ode_solver):
    """CPU Test for GOW17 uniform test problem."""
    gow17_uniform.run_gow17_uniform_big_step(ode_solver)


@pytest.mark.parametrize("ode_solver", gow17_uniform.ode_solvers)
def test_gow17_cfl_dependence_cpu(ode_solver):
    """CPU Test for if the results of network depend on the CFL number"""
    gow17_uniform.run_gow17_cfl_dependence(ode_solver)


@pytest.mark.parametrize("ode_solver", gow17_uniform.time_series_ode_solvers)
def test_gow17_uniform_time_series_cpu(ode_solver):
    """CPU Test for GOW17 time series problem."""
    gow17_uniform.run_gow17_uniform_time_series(ode_solver)
