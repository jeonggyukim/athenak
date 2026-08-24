#ifndef ODE_SOLVERS_KOKKOS_BDF_HPP_
#define ODE_SOLVERS_KOKKOS_BDF_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kokkos_bdf.hpp
//  \brief Wrapper for the Kokkos Kernels BDF ODE solver

#include <KokkosODE_BDF.hpp>
#include <string>  // NOLINT(build/include_order)

#include "athena.hpp"

namespace ode_solvers {

struct KokkosBDFSettings {
  /// Fraction of the integration interval (the hydro timestep) to use for the
  /// solver's first internal step, i.e. dt0 = first_step_frac * dt. A value of
  /// 0 (the default) gives dt0 = 0, which lets the Kokkos Kernels BDF driver
  /// auto-select its first step. This is an opt-in escape hatch: a fixed
  /// fraction is a poor global control because it is too small in the easy
  /// (near-equilibrium) regime, needlessly slowing every step, yet still too
  /// large to cure the ill-conditioned first-cycle solve for stiff networks.
  Real first_step_frac;
};

/*!
 * \brief Kokkos Kernels' BDFSolve, with the internal step count returned.
 *
 * This mirrors KokkosODE::Experimental::BDFSolve (Kokkos Kernels 4.7.04) and
 * adds only a counter around its internal while loop. Upstream returns void and
 * keeps the loop private, so there is no way to ask it how many internal steps
 * one macro-step cost -- and that count is exactly what is needed to compare
 * chemistry cost against hydro cost, since a macro-step that quietly subcycles
 * a thousand times is not comparable to a single hydro update.
 *
 * Upstream also ignores its own max_step argument (`(void)max_step;`) and hard
 * codes atol = 1e-6, rtol = 1e-3, so those are reproduced here rather than made
 * settable; changing them would change the answer, not just the diagnostics.
 *
 * Re-check this against upstream whenever the pinned Kokkos Kernels version in
 * the top level CMakeLists.txt changes.
 *
 * \return The number of internal BDF steps taken.
 */
template <class ode_type, class mat_type, class vec_type, class scalar_type>
KOKKOS_FUNCTION int CountedBDFSolve(const ode_type& ode,
                                    const scalar_type t_start,
                                    const scalar_type t_end,
                                    const scalar_type initial_step,
                                    const vec_type& y0, const vec_type& y_new,
                                    mat_type& temp, mat_type& temp2) {
  using KAT = Kokkos::ArithTraits<scalar_type>;

  auto rhs = Kokkos::subview(temp, Kokkos::ALL(), 0);
  auto update = Kokkos::subview(temp, Kokkos::ALL(), 1);

  int order = 1, num_equal_steps = 0;
  constexpr scalar_type min_factor = 0.2;
  scalar_type dt = initial_step;
  scalar_type t = t_start;

  constexpr int max_newton_iters = 10;
  scalar_type atol = 1.0e-6, rtol = 1.0e-3;

  // Compute rhs = f(t_start, y0)
  ode.evaluate_function(t_start, 0, y0, rhs);

  // Check if we need to compute the initial time step size.
  if (initial_step == KAT::zero()) {
    KokkosODE::Impl::initial_step_size(ode, order, t_start, atol, rtol, y0, rhs,
                                       temp, dt);
  }

  // Initialize D(:, 0) = y0 and D(:, 1) = dt*rhs
  auto D = Kokkos::subview(temp, Kokkos::ALL(), Kokkos::pair<int, int>(2, 10));
  for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
    D(eqIdx, 0) = y0(eqIdx);
    D(eqIdx, 1) = dt * rhs(eqIdx);
    rhs(eqIdx) = 0;
  }

  int n_steps = 0;
  while (t < t_end) {
    KokkosODE::Impl::BDFStep(ode, t, dt, t_end, order, num_equal_steps,
                             max_newton_iters, atol, rtol, min_factor, y0,
                             y_new, rhs, update, temp, temp2);

    for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
      y0(eqIdx) = y_new(eqIdx);
    }
    ++n_steps;
  }
  return n_steps;
}

/*!
 * \brief Solve a system of ODEs using the BDF solver from Kokkos Kernels
 *
 * \tparam T The type of the ODE system to solve
 */
template <typename ode_t>
class KokkosBDF {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION
  KokkosBDF(KokkosBDFSettings const settings, ode_t& ode_system,
            Real const t_start, Real const dt)
      : ode_system(ode_system),
        t_start(t_start),
        dt(dt),
        t_end(t_start + dt),
        dt0(settings.first_step_frac * dt),
        max_step(dt),
        temp_(&temp_buffer_[0][0], ode_t::neqs, 23 + 2 * ode_t::neqs + 4),
        temp2_(&temp2_buffer_[0][0], 6, 7) {}
  KOKKOS_FUNCTION
  ~KokkosBDF() = default;

  // ----- Variables -----
  /// The system of ODEs to solve
  ode_t& ode_system;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;
  /// Time to integrate to
  const Real t_end;
  /// First time step size, if zero then the solver will decide
  const Real dt0;
  /// The maximum time step. Kokkos Kernels discards this argument outright
  /// (`(void)max_step;` in BDFSolve), still true as of 4.7.04, so it does
  /// nothing and there is no input-file control over the internal step size.
  const Real max_step;
  /// Number of internal BDF steps the last SolveODE() call took. Diagnostic
  /// only: per-cell chemistry cost scales with this.
  int n_substeps = 0;

  /*!
   * \brief Get the settings for the  ODE solver from the input file
   *
   * \param pin The ParameterInput object
   * \param module The physics module that this ODE solver is called in. The
   * name should match the block name in the input file for the physics module.
   * \return KokkosBDFSettings The settings for the Kokkos BDF solver
   */
  static KokkosBDFSettings GetSettings(ParameterInput* pin,
                                       std::string module) {
    // Default 0 => dt0 = 0 => the solver auto-selects its first step. A fixed
    // fraction of the macro-step is a poor global control (too small in the
    // easy regime, too large in the stiff first cycle), so it is opt-in only.
    return KokkosBDFSettings{
        pin->GetOrAddReal(module, "kokkos_BDF_first_step_frac", 0.0)};
  }

  KOKKOS_FUNCTION
  void SolveODE() {
    n_substeps = CountedBDFSolve(ode_system, t_start, t_end, dt0, ode_system.y,
                                 ode_system.y_new, temp_, temp2_);
  }

 private:
  // temporary storage for inside the BDF solver
  Real temp_buffer_[ode_t::neqs][23 + 2 * ode_t::neqs + 4];
  Real temp2_buffer_[6][7];
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> temp_;
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> temp2_;
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_KOKKOS_BDF_HPP_
