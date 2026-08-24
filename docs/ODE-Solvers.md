
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

| Option               | Type         | Default | Description                                        |
| -------------------- | ------------ | ------- | -------------------------------------------------- |
| fe_cfl               | Real         | 0.1     | cfl number for subcycling                          |
| fe_n_subcycle_max    | unsigned int | 1e5     | maximum number of substeps                         |
| fe_yfloor            | Real         | 1e-12   | y value floor for calculating subcycling timescale |

*`fe` stands for Forward Euler.*

### Kokkos BDF

The Kokkos BDF solver is a wrapper around the implicit [Backward Differentiation Formula](https://en.wikipedia.org/wiki/Backward_differentiation_formula) solver provided by [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels). BDF methods are implicit, multi-step methods that are well suited to the stiff ODEs common in chemical networks. The solver adapts both its internal step size and order automatically as it integrates across the hydro timestep. Because it is implicit, the ODE system must additionally provide a Jacobian (see the [ODE System API](#ode-system-api) below).

Runtime Parameters:

| Option                     | Type   | Default | Description                                                                                                                                                                              |
| -------------------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kokkos_BDF_first_step_frac | Real   | 0.0     | fraction of the hydro timestep to use for the solver's first internal step, i.e. `dt0 = kokkos_BDF_first_step_frac * dt`. The default value of 0.0 let's the solver decide the time step |

*Note: The Kokkos BDF solver also has a maximum internal step size that is set to the hydro timestep `dt`. As of Kokkos Kernels 4.4 this is not implemented within Kokkos Kernels and so it currently has no effect.*

*Note: The build applies [a patch](../../blob/master/patches/kokkos-kernels-ode-newton-partial-pivoting.patch) to the Kokkos Kernels ODE Newton solver. The upstream solver uses `KokkosBatched::SerialGesv` with static pivoting, whose greedy row-to-column assignment fails spuriously ("`KokkosBatched::gesv: the currently implemented static pivoting failed.`") on well-conditioned sparse Jacobians like those of chemistry networks with an internal energy equation, whose dense energy row can steal a nearly-diagonal species row's only pivot column. The patch replaces that solve with LU with partial pivoting (`SerialGetrf`/`SerialGetrs`), returns from the Newton iteration before applying the update if the linear solve fails, and makes the BDF driver halve its internal step on a linear-solve failure instead of ignoring it.*

### Semi-Implicit Sweep

A synchronous semi-implicit backward-Euler sweep, following the closed-form update used by [Katz 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.512..348K) in RAMSES-RTZ. For a network whose right-hand side has the form `f_i = C_i(y) - y_i * D_i(y)`, each substep applies

```
y_i <- (y_i + C_i * dt) / (1 + D_i * dt)
```

with `C` and `D` frozen at the start of the substep. There is no Jacobian, no linear solve and no Newton iteration, so per-cell scratch is a few `neqs`-sized arrays rather than the `neqs x (23 + 2*neqs + 4)` workspace the BDF solver needs.

The update is unconditionally positive for non-negative `C` and `D` and is asymptotically correct at both ends of the stiffness range: it reduces to forward Euler when `D*dt << 1` and returns the local equilibrium `C_i/D_i` exactly when `D*dt >> 1`. The fast species of a network — in GOW17 these are H2+, H3+, HCO+, CHx and OHx — therefore relax onto their equilibrium rather than forcing a small step. This is the quasi-steady-state approximation applied automatically, per cell and per substep, wherever it is valid, and it is the reason the method is cheap.

That property also dictates the step-size control. Limiting on `min_i |y_i / f_i|` over *all* species, as the forward Euler solver does, is set by the stiffest species — exactly the ones the update already handles exactly — and would reproduce forward Euler's substep count for no gain. The controller therefore skips any species whose destruction rate puts it in the relaxation-dominated regime, and applies a saturating relative-change limit to the rest.

Internal energy is operator-split from the chemistry so the species update stays closed-form, and is advanced once per substep, optionally damped by a numerical `dEdot/dE`.

The method is first order. It is intended for networks whose cost is dominated by stiff species near equilibrium; where the resolved species or the energy equation set the step, a higher-order implicit solver may still win. Compare against the Kokkos BDF solver on the problem of interest before adopting it.

Runtime Parameters:

| Option                | Type         | Default | Description                                                                                                                                              |
| --------------------- | ------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sweep_cfl             | Real         | 0.1     | target fractional change per substep, applied only to the species that are being resolved                                                                |
| sweep_yfloor          | Real         | 1e-12   | abundance floor used when forming the relative-change criterion                                                                                          |
| sweep_n_substep_max   | unsigned int | 1e5     | maximum number of substeps per macro-step; exceeding it is fatal                                                                                         |
| sweep_n_substep_fixed | unsigned int | 0       | if > 0, take exactly this many equal substeps and skip the adaptive controller. Every cell then does identical work, which removes warp divergence on GPU |
| sweep_stiff_threshold | Real         | 1.0     | a species is treated as relaxation-dominated, and excluded from the step-size control, when `D_i * time_remaining` exceeds this                          |
| sweep_n_iter          | unsigned int | 1       | fixed-point iterations of the species update per substep. Values above 1 help where fast species are mutually coupled, as in the CHx -> CO -> HCO+ cycle  |
| sweep_energy_implicit | bool         | true    | damp the internal-energy update with a numerical `dEdot/dE`, costing one extra `Edot` evaluation per substep                                             |
| sweep_renormalize     | bool         | true    | rescale each element back onto its conservation law after every substep                                                                                  |

Beyond the common ODE system API below, this solver requires the network to provide `SetupNextStep`, `CDRates`, `Edot`, and — when `sweep_renormalize` is set — `RenormalizeElements`. Only `GOW17` implements these, so `semi_implicit_sweep` is registered for that network alone.

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
