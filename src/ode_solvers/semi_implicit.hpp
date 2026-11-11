#ifndef ODE_SOLVERS_SEMI_IMPLICIT_HPP_
#define ODE_SOLVERS_SEMI_IMPLICIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file semi_implicit.hpp
//  \brief A semi-implicit Gauss-Seidel solver for chemical networks written as
//  creation and destruction rates

#include <algorithm>
#include <string>
#include <utility>

#include "athena.hpp"

namespace ode_solvers {

struct SemiImplicitSettings {
  /// Fraction of the energy and limiting-species timescales a substep may take
  Real semi_implicit_cfl;
  /// Floor on the abundance used to scale the species timescales
  Real semi_implicit_yfloor;
  /// Maximum number of substeps; the last one takes all the time that is left
  int semi_implicit_nsub_max;
  /// If positive, use this many equal substeps instead of choosing them
  int semi_implicit_nsub_fixed;
  /// Recompute the rate coefficients after the energy update
  bool semi_implicit_refresh_rates;
};

namespace semi_implicit_detail {
/// True if every index in the sequence is distinct and below `n`
template <int... K>
constexpr bool IsValidOrder(std::integer_sequence<int, K...>, const int n) {
  constexpr int idx[] = {K...};
  for (std::size_t a = 0; a < sizeof...(K); a++) {
    if (idx[a] < 0 || idx[a] >= n) return false;
    for (std::size_t b = a + 1; b < sizeof...(K); b++) {
      if (idx[a] == idx[b]) return false;
    }
  }
  return true;
}
}  // namespace semi_implicit_detail

/*!
 * \brief Solve a chemical network with a semi-implicit Gauss-Seidel method.
 *
 * \details The interval is split into substeps of length h, each limited to
 * `semi_implicit_cfl` times the energy timescale e/|Edot| and the timescales
 * y/|f| of the species the network lists in `StepLimitingSpecies`. In each
 * substep the internal energy takes one linearized backward Euler step,
 *   e^{n+1} = e^n + h Edot / (1 - h dEdot/de),
 * and then, with the rate coefficients recomputed at the new temperature unless
 * `semi_implicit_refresh_rates` is false, each species takes one backward Euler
 * step with its creation rate C and destruction rate D,
 *   y_k^{n+1} = (y_k^n + h C_k) / (1 + h D_k),
 * in the order the network lists. C and D are evaluated from the current state,
 * so a species sees the new values of the species updated before it.
 *
 * This solver needs the right-hand side split into creation and destruction
 * rates, f_k = C_k - y_k D_k with C_k, D_k >= 0, and treats the last variable
 * (IIE) as the internal energy. It therefore applies to chemical and
 * ionization networks with an energy equation, not to a general ODE system. It
 * has no error estimate; its accuracy is set by the substep length.
 *
 * The network provides, in addition to `neqs`, `IIE` and `y`:
 *   - `UpdateOrder`, a `std::integer_sequence<int, ...>` of species indices in
 *     the order they are updated. It is a compile-time list, so the update is
 *     unrolled with constant indices and `y` stays in registers.
 *   - `StepLimitingSpecies`, the species whose relative change limits the
 *     substep, in the same form. Species that relax to equilibrium within a
 *     substep should not be listed; backward Euler carries them there anyway.
 *   - `SetupNextStep(y)`, which prepares the rate coefficients for a substep
 *     and returns the ghost species (abundances not stored in `y` but set by
 *     element conservation), or an empty struct if there are none.
 *   - `CDRates(y, ghosts)`, the creation and destruction rates, used as
 *     `f(k) = creation(k) - y(k) * destruction(k)`.
 *   - `Edot(y, ghosts)`, the internal energy equation.
 *
 * \tparam ode_t The type of the chemical network to solve
 */
template <typename ode_t>
class SemiImplicit {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION
  SemiImplicit(SemiImplicitSettings const settings, ode_t& ode_system,
               Real const t_start, Real const dt)
      : ode_system(ode_system),
        cfl(settings.semi_implicit_cfl),
        yfloor(settings.semi_implicit_yfloor),
        nsub_max(settings.semi_implicit_nsub_max),
        nsub_fixed(settings.semi_implicit_nsub_fixed),
        refresh_rates(settings.semi_implicit_refresh_rates),
        t_start(t_start),
        dt(dt) {}
  KOKKOS_FUNCTION
  ~SemiImplicit() = default;

  static_assert(semi_implicit_detail::IsValidOrder(typename ode_t::UpdateOrder{},
                                                   ode_t::IIE),
                "UpdateOrder must list distinct species indices below IIE");
  static_assert(semi_implicit_detail::IsValidOrder(
                    typename ode_t::StepLimitingSpecies{}, ode_t::IIE),
                "StepLimitingSpecies must list distinct species indices below IIE");

  // ----- Variables -----
  /// A small number to keep the timescales finite
  static constexpr Real small = 1e-35;
  /// The system of ODEs to solve
  ode_t& ode_system;
  /// Fraction of the timescales a substep may take
  const Real cfl;
  /// Floor on the abundance used to scale the species timescales
  const Real yfloor;
  /// Maximum number of substeps
  const int nsub_max;
  /// If positive, the number of equal substeps
  const int nsub_fixed;
  /// Recompute the rate coefficients after the energy update
  const bool refresh_rates;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;

  /*!
   * \brief Get the settings for the semi-implicit solver from the input file
   *
   * \param pin The ParameterInput object
   * \param module The physics module that this ODE solver is called in. The
   * name should match the block name in the input file for the physics module.
   * \return SemiImplicitSettings The settings for the semi-implicit solver
   */
  static SemiImplicitSettings GetSettings(ParameterInput* pin,
                                          std::string module) {
    return SemiImplicitSettings{
        pin->GetOrAddReal(module, "semi_implicit_cfl", 0.1),
        pin->GetOrAddReal(module, "semi_implicit_yfloor", 1.0e-12),
        pin->GetOrAddInteger(module, "semi_implicit_nsub_max", 100),
        pin->GetOrAddInteger(module, "semi_implicit_nsub_fixed", 0),
        pin->GetOrAddBoolean(module, "semi_implicit_refresh_rates", false)};
  }

  KOKKOS_FUNCTION
  void SolveODE() {
    auto& y = ode_system.y;
    const Real t_end = t_start + dt;
    Real t_now = t_start;
    for (int n = 0; t_now < t_end; n++) {
      Real y_n[ode_t::neqs];  // NOLINT(runtime/arrays)
      for (int i = 0; i < ode_t::neqs; i++) {
        y_n[i] = Kokkos::fmax(y(i), 0.0);
        y(i) = y_n[i];
      }

      auto ghosts = ode_system.SetupNextStep(y);
      const Real edot = ode_system.Edot(y, ghosts);
      const Real dedot_de = EnergyDerivative_(ghosts, edot);

      // Dividing the time left by the substeps left lands the last substep
      // exactly on t_end, where accumulating a fixed h would leave t_now short
      const Real t_left = t_end - t_now;
      Real h;
      if (nsub_fixed > 0) {
        h = t_left / Kokkos::max(nsub_fixed - n, 1);
      } else if (n >= nsub_max - 1) {
        h = t_left;
      } else {
        h = ChooseStep_(ghosts, edot, t_left);
      }

      UpdateEnergy_(edot, dedot_de, h);
      if (refresh_rates) ghosts = ode_system.SetupNextStep(y);
      Sweep_(typename ode_t::UpdateOrder{}, y_n, ghosts, h);
      t_now = (h == t_left) ? t_end : t_now + h;
    }
  }

 private:
  /// Relative change of the internal energy used to difference Edot
  static constexpr Real de_rel = 1.0e-4;

  /*!
   * \brief dEdot/de by a one-sided difference at fixed species, clamped to
   * <= 0: only a negative slope damps the energy step
   */
  template <class ghost_type>
  KOKKOS_FUNCTION Real EnergyDerivative_(const ghost_type& ghosts,
                                         const Real edot) {
    auto& y = ode_system.y;
    const Real e = y(ode_t::IIE);
    if (edot == 0.0 || e <= 0.0) return 0.0;
    const Real de = de_rel * e;
    y(ode_t::IIE) = e + de;
    const Real edot_p = ode_system.Edot(y, ghosts);
    y(ode_t::IIE) = e;
    return Kokkos::fmin((edot_p - edot) / de, 0.0);
  }

  /*!
   * \brief The substep: `cfl` times the shortest of the energy timescale
   * e/|Edot| and the relative timescales y/|f| of the step-limiting species.
   * The species timescale is relative because the trace species sit at 1e-10
   * to 1e-4 of H and change by orders of magnitude while their absolute change
   * is tiny.
   */
  template <class ghost_type>
  KOKKOS_FUNCTION Real ChooseStep_(const ghost_type& ghosts, const Real edot,
                                   const Real t_left) const {
    const auto& y = ode_system.y;
    Real h = t_left;
    const Real e = y(ode_t::IIE);
    if (e > 0.0) {
      h = Kokkos::fmin(h, cfl * e / (Kokkos::fabs(edot) + small));
    }
    const auto rates = ode_system.CDRates(y, ghosts);
    LimitBySpecies_(typename ode_t::StepLimitingSpecies{}, rates, h);
    return h;
  }

  template <int... K, class rates_type>
  KOKKOS_FUNCTION void LimitBySpecies_(std::integer_sequence<int, K...>,
                                       const rates_type& rates, Real& h) const {
    const auto& y = ode_system.y;
    ((h = Kokkos::fmin(
          h, cfl * Kokkos::fmax(y(K), yfloor) /
                 (Kokkos::fabs(rates.creation(K) - y(K) * rates.destruction(K)) +
                  small))),
     ...);
  }

  KOKKOS_FUNCTION void UpdateEnergy_(const Real edot, const Real dedot_de,
                                     const Real h) {
    auto& y = ode_system.y;
    const Real e = y(ode_t::IIE);
    y(ode_t::IIE) = Kokkos::fmax(e + h * edot / (1.0 - h * dedot_de), 0.0);
  }

  template <int... K, class ghost_type>
  KOKKOS_FUNCTION void Sweep_(std::integer_sequence<int, K...>, const Real* y_n,
                              const ghost_type& ghosts, const Real h) {
    (UpdateOne_<K>(y_n, ghosts, h), ...);
  }

  template <int K, class ghost_type>
  KOKKOS_FUNCTION void UpdateOne_(const Real* y_n, const ghost_type& ghosts,
                                  const Real h) {
    auto& y = ode_system.y;
    const auto rates = ode_system.CDRates(y, ghosts);
    y(K) = (y_n[K] + h * rates.creation(K)) / (1.0 + h * rates.destruction(K));
  }
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_SEMI_IMPLICIT_HPP_
