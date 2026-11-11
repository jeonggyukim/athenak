
# ODE Solvers

*NOTE: This is still an area of active development and as such is subject to change. If you see any discrepancies please open an issue.*

AthenaK provides ODE solvers for solving systems of ODEs. This is still a work in progress and more ODE solvers will be added.

These ODE solvers require that the ODEs they're solving have a consistent interface. This is discussed in more detail in the [Developer Documentation](#developer-documentation)

## How to Use

### Input File

ODE solvers are chosen and initialized in the module that they are used and so the parameters for them should be defined within the block of a specific physics module. This is to accommodate using different ODE solvers with different settings in the various modules in AthenaK. For example, to use the forward euler solver in the chemistry module and in another module those sections of your input file might look like this:

```
<chemistry>
network    = H2            # Chemistry network to be used
ode_solver = forward_euler # ODE solver to be used
fe_cfl     = 0.02          # The CFL number for the forward euler ODE solver

<other physics module>
ode_solver        = forward_euler # ODE solver to be used
fe_n_subcycle_max = 1e7           # maximum number of substeps for the forward euler ODE solver
```

### Forward Euler

The simple forward Euler solver uses the explicit first-order differentiation formula to solve the ODEs by sub-cycling. Since this is an explicit method it may not be suitable for stiff ODEs. This solver is primarily intended for testing due to its simplicity and isn't necessarily expected to be performant or highly accurate.

Runtime Parameters:

| Option            | Type         | Default | Description                                        |
| ----------------- | ------------ | ------- | -------------------------------------------------- |
| fe_cfl            | Real         | 0.1     | cfl number for subcycling                          |
| fe_n_subcycle_max | unsigned int | 1e5     | maximum number of substeps                         |
| fe_yfloor         | Real         | 1e-12   | y value floor for calculating subcycling timescale |

*`fe` stands for Forward Euler.*

### Kokkos BDF

The Kokkos BDF solver is a wrapper around the implicit [Backward Differentiation Formula](https://en.wikipedia.org/wiki/Backward_differentiation_formula) solver provided by [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels). BDF methods are implicit, multi-step methods that are well suited to the stiff ODEs common in chemical networks. The solver adapts both its internal step size and order automatically as it integrates across the hydro timestep. Because it is implicit, the ODE system must additionally provide a Jacobian (see the [ODE System API](#ode-system-api) below).

Runtime Parameters:

| Option                     | Type | Default | Description                                                                                                                                                                              |
| -------------------------- | ---- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kokkos_BDF_rtol            | Real | 1.0e-3  | The allowed relative tolerance passed to the ODE solver                                                                                                                                  |
| kokkos_BDF_atol            | Real | 1.0e-6  | The allowed absolute tolerance passed to the ODE solver                                                                                                                                  |
| kokkos_BDF_first_step_frac | Real | 0.0     | fraction of the hydro timestep to use for the solver's first internal step, i.e. `dt0 = kokkos_BDF_first_step_frac * dt`. The default value of 0.0 let's the solver decide the time step |

*Note: The Kokkos BDF solver also has a maximum internal step size that is set to the hydro timestep `dt`. As of Kokkos Kernels 4.4 this is not implemented within Kokkos Kernels and so it currently has no effect.*

### Semi-Implicit

The semi-implicit solver splits the hydro timestep into substeps. In each substep the internal energy takes one linearized backward Euler step, `e += h Edot / (1 - h dEdot/de)`, and then each species takes one backward Euler step with its creation rate `C` and destruction rate `D`, `y = (y_n + h C) / (1 + h D)`. The species are updated one at a time in an order set by the network (Gauss-Seidel), so each species sees the new values of those updated before it. It needs no Jacobian and no linear solve, which makes it much cheaper than BDF, but it has no error control: its accuracy is set by `semi_implicit_cfl`.

Each substep is limited to `semi_implicit_cfl` times the shortest of the internal-energy timescale `e/|Edot|` and the timescales `y/|f|` of the species the network lists as step-limiting. Species that relax to equilibrium within a substep are not listed; backward Euler takes them there without overshooting.

Runtime Parameters:

| Option                      | Type | Default | Description                                                                                         |
| --------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------- |
| semi_implicit_cfl           | Real | 0.1     | fraction of the energy and step-limiting species timescales a substep may take                      |
| semi_implicit_yfloor        | Real | 1e-12   | abundance floor used when computing the species timescales                                          |
| semi_implicit_nsub_max      | int  | 100     | maximum number of substeps; the last one takes all the time that is left                            |
| semi_implicit_nsub_fixed    | int  | 0       | if positive, use this many equal substeps instead of choosing them from the timescales              |
| semi_implicit_refresh_rates | bool | false   | recompute the rate coefficients at the new temperature after the energy update                      |

## Developer Documentation

To facilitate easy swapping between different ODE solvers both the solvers and ODE system classes must have a very strict APIs. The forward euler solver in [forward_euler.hpp](../../blob/master/src/ode_solvers/forward_euler.hpp) and H2 network in [H2.hpp](../../blob/master/src/chemistry/network/H2.hpp) can serve as a templates but here's a description of the APIs in more detail.

### ODE Solver API

The ODE solvers should be contained in a class that is templated on the ODE system they're solving. This enables inlining which should provide a significant performance boost on GPUs. The ODE solver's interface consists of three methods:

- A `static` `GetSettings` method that is called from the host to get any parameters needed from the input file. This should return a struct that contains all the runtime parameters for the ODE solver
- A constructor with the following signature `ODESolverName(ODESettings const settings, T& ode_system, Real const t_start, Real const dt)`
  - `ODESettings const settings`: the struct that contains the runtime parameters for the solver
  - `T& ode_system`: The ODE object to evolve
  - `Real const t_start`: The start time
  - `Real const dt`: How much time to evolve the ODEs
- A `SolveODE()` method that evolves the system of ODEs by time `dt`

### ODE System API

The ODE system should be contained in a class with at minimum the following members:

- A `neqs` variable that contains the total number of equations in the system. This should probably be declared as `static constexpr int` since it's used to define loop limits
- A `y` member variable. It should be an array of type `Real` with length `neqs` that contains the current state of the values being updated by the solver. This must support accesses `operator()` and `operator[]`, the `RegisterArray` class provides this syntax if a Kokkos View doesn't work.
- A `y_new` member variable. It should be an array of type `Real` with length `neqs` that is used as scratch space to hold the updated state (and, for the Forward Euler solver, the result of evaluating the right-hand side)
- An `evaluate_function(t, dt, y_in, out)` method that computes the right-hand side of the ODEs (the derivatives) from the state `y_in` and writes them into `out`. It should be templated on the array types. The `t` and `dt` arguments are the current time and step size; solvers that do not need them (such as Forward Euler) simply pass zeros
- Many solvers require an `evaluate_jacobian(t, dt, y_in, jac)` method that computes the Jacobian matrix `jac` of the right-hand side with respect to `y_in`. Like `evaluate_function`, it should be templated on the array types. The `chemistry::numerical_jacobian` helper can be used to compute the Jacobian via finite differences if an analytic Jacobian is not available.

Specific ODE solvers might require other methods and they are noted below.

#### Semi-Implicit Solver

The semi-implicit solver additionally needs, in the ODE system class:

- `UpdateOrder`, a `std::integer_sequence<int, ...>` of species indices in the order they are updated. It is a compile-time list, so the update is unrolled with constant indices.
- `StepLimitingSpecies`, the species whose relative change limits the substep, in the same form.
- `SetupNextStep(y)`, which prepares the rate coefficients for a substep and returns the ghost species (species set by conservation, not stored), or an empty struct if there are none.
- `CDRates(y, ghosts)`, the creation and destruction rates, used as `f(k) = creation(k) - y(k) * destruction(k)`.
- `Edot(y, ghosts)`, the internal energy equation.

Optionally, a network sets `static constexpr bool custom_species_update = true` and provides `template <int K> void UpdateSpecies(y, y_n, ghosts, h)`, which then replaces the update of every species `K` in `UpdateOrder`. A network uses it to evaluate only one species' rates per update, to write a species in a more stable form, or to update several species together (those then appear once in `UpdateOrder`). The H2 network uses it to set H from hydrogen conservation; GOW17 uses it to write the ions created from neutral reservoirs implicitly and to solve CO and HCO+ together.

### Code Example

See the `Chemistry::UpdateChemistryTask` method in [chemistry_tasks.cpp](../../blob/main-chemistry/src/chemistry/chemistry_tasks.cpp) and the `Chemistry::UpdateChemistry` method in [chemistry.cpp](../../blob/main-chemistry/src/chemistry/chemistry.cpp) for a complete example. A simplified version of the most salient sections is below.

```cpp
// First we read the requested network and ODE solver from the input file and
// dispatch to the matching template instantiation of UpdateChemistry. Each
// (network, solver) pair that should be supported must be listed here.
TaskStatus Chemistry::UpdateChemistryTask(Driver* d, int stage) {
  const std::string network = my_pin->GetString("chemistry", "network");
  const std::string ode_solver = my_pin->GetString("chemistry", "ode_solver");

  if (network == "H2") {
    if (ode_solver == "forward_euler") {
      UpdateChemistry<ode_solvers::ForwardEuler, H2Network>();
    } else if (ode_solver == "kokkos_BDF") {
      UpdateChemistry<ode_solvers::KokkosBDF, H2Network>();
    }
  } else if (network == "GOW17") {
    if (ode_solver == "forward_euler") {
      UpdateChemistry<ode_solvers::ForwardEuler, GOW17Network>();
    } else if (ode_solver == "kokkos_BDF") {
      UpdateChemistry<ode_solvers::KokkosBDF, GOW17Network>();
    }
  }

  return TaskStatus::complete;
}

// Now we actually use the solver in a kernel to solve the system of ODEs. The
// method is templated on the ODE solver (itself a template on the network) and
// on the network, so the solver can be fully inlined for the GPU.
template <template <typename> class ODE_Solver_t, typename Network_t>
void Chemistry::UpdateChemistry() {
  // ------ Collect variables that we'll need -----
  // The primitive grid
  auto w0 = GetW0();
  // The time at the beginning of this timestep
  Real const t_start = pmy_pack->pmesh->time;
  // The timestep
  Real const dt = pmy_pack->pmesh->dt;

  // ----- Load the network and ODE solver settings from the input file -----
  auto const ode_settings =
      ODE_Solver_t<Network_t>::GetSettings(my_pin, "chemistry");
  auto const network_settings = Network_t::GetSettings(my_pin, pmy_pack);

  Kokkos::parallel_for(
      "Chemistry_ODE_Solve", policy,
      KOKKOS_LAMBDA(const int& mb_idx, const int& k, const int& j,
                    const int& i) {
        // Create the chemistry object
        Network_t chem_net(network_settings, mb_idx, k, j, i, w0, /* ... */);

        // ------ Load cell values ------
        // Load the chemical species into the `y` array. The loop runs to
        // neqs - 1 because the internal energy occupies the last slot.
        int grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs - 1; s_idx++) {
          chem_net.y(s_idx) = w0(mb_idx, grid_idx, k, j, i);
          grid_idx += 1;
        }
        // Load the internal energy into the last slot
        chem_net.y(Network_t::IIE) = w0(mb_idx, IEN, k, j, i);

        // ------ Solve the ODEs ------
        // The solver aborts via Kokkos::abort if it fails to converge, so there
        // is no failure flag to check here.
        ODE_Solver_t ode_solver(ode_settings, chem_net, t_start, dt);
        ode_solver.SolveODE();

        // ------ Write cell values back out ------
        grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs - 1; s_idx++) {
          w0(mb_idx, grid_idx, k, j, i) = chem_net.y(s_idx);
          grid_idx += 1;
        }
        // Write the internal energy back out
        w0(mb_idx, IEN, k, j, i) = chem_net.y(Network_t::IIE);
      });
}
```
