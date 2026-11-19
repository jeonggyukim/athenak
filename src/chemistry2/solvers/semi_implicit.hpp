#ifndef CHEMISTRY2_SOLVERS_SEMI_IMPLICIT_HPP_
#define CHEMISTRY2_SOLVERS_SEMI_IMPLICIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file semi_implicit.hpp
//! \brief Semi-implicit Gauss-Seidel solver for a chemical network and its energy
//!        equation.
//!
//!   The interval is split into substeps of length h, each limited to dt_sub_frac times
//!   the energy timescale e/|edot| and the timescales y/|f| of the species the network
//!   lists in StepLimitingSpecies; after nsub_max - 1 substeps the last one takes the
//!   time left. In each substep the internal energy takes one linearized backward Euler
//!   step,
//!     e^{n+1} = e^n + h edot / (1 - h dedot/de),
//!   and then, with the rate coefficients recomputed at the new temperature if
//!   refresh_rates, each species in UpdateOrder takes one backward Euler step
//!     y_k^{n+1} = (y_k^n + h c_k) / (1 + h d_k),
//!   with c and d from the current state, so a species sees the new values of those
//!   updated before it. The network's Closures then set the species it does not update.
//!
//!   The network provides, as static members: NSPEC, UpdateOrder, StepLimitingSpecies,
//!   Cell, Substep, Prepare(y, e, cell), Edot(y, e, cell, sub), Rates<K>(y, e, cell, sub)
//!   and Closures(y, y_n, cell).

#include <string>
#include <utility>

#include "athena.hpp"
#include "chemistry2/network/network_config.hpp"
#include "parameter_input.hpp"

namespace chemistry2 {

struct SemiImplicitSettings {
  Real dt_sub_frac;
  Real yfloor;
  int nsub_max;
  bool refresh_rates;

  static SemiImplicitSettings Read(ParameterInput *pin, const std::string &block) {
    return SemiImplicitSettings{
        pin->GetOrAddReal(block, "semi_implicit_dt_sub_frac", 0.1),
        pin->GetOrAddReal(block, "semi_implicit_yfloor", 1.0e-12),
        pin->GetOrAddInteger(block, "semi_implicit_nsub_max", 100000),
        pin->GetOrAddBoolean(block, "semi_implicit_refresh_rates", true)};
  }
};

template <class Network>
class SemiImplicit {
 public:
  static constexpr int NSPEC = Network::NSPEC;
  using Cell = typename Network::Cell;
  using Substep = typename Network::Substep;

  //! Advance y[NSPEC] and e from t_start over dt; returns the number of substeps
  KOKKOS_INLINE_FUNCTION static int Solve(const SemiImplicitSettings &s, const Cell &cell,
                                          const Real t_start, const Real dt, Real *y,
                                          Real &e) {
    const Real t_end = t_start + dt;
    Real t_now = t_start;
    int n = 0;
    for (; t_now < t_end; n++) {
      Real y_n[NSPEC];  // NOLINT(runtime/arrays)
      for (int k = 0; k < NSPEC; k++) {
        y_n[k] = Kokkos::fmax(y[k], 0.0);
        y[k] = y_n[k];
      }
      e = Kokkos::fmax(e, 0.0);

      Substep sub = Network::Prepare(y, e, cell);
      const Real edot = Network::Edot(y, e, cell, sub);
      const Real dedot_de = EnergyDerivative(y, e, cell, sub, edot);
      const Real t_left = t_end - t_now;
      const Real h = (n >= s.nsub_max - 1)
                         ? t_left
                         : ChooseStep(s, y, e, cell, sub, edot, t_left,
                                      typename Network::StepLimitingSpecies{});

      e = Kokkos::fmax(e + h*edot/(1.0 - h*dedot_de), 0.0);
      if (s.refresh_rates) sub = Network::Prepare(y, e, cell);
      UpdateInOrder(y, y_n, e, cell, sub, h, typename Network::UpdateOrder{});
      Network::Closures(y, y_n, cell);

      t_now = (h == t_left) ? t_end : t_now + h;
    }
    return n;
  }

 private:
  static constexpr Real SMALL = 1e-35;
  static constexpr Real DE_REL = 1.0e-4;

  //! dedot/de by a one-sided difference at fixed species, clamped to <= 0: only a
  //! negative slope damps the energy step
  KOKKOS_INLINE_FUNCTION static Real EnergyDerivative(const Real *y, const Real e,
                                                      const Cell &cell,
                                                      const Substep &sub,
                                                      const Real edot) {
    if (edot == 0.0 || e <= 0.0) return 0.0;
    const Real de = DE_REL*e;
    const Real edot_p = Network::Edot(y, e + de, cell, sub);
    return Kokkos::fmin((edot_p - edot)/de, 0.0);
  }

  template <int... K>
  KOKKOS_INLINE_FUNCTION static Real ChooseStep(const SemiImplicitSettings &s,
                                                const Real *y, const Real e,
                                                const Cell &cell, const Substep &sub,
                                                const Real edot, const Real t_left,
                                                std::integer_sequence<int, K...>) {
    Real h = t_left;
    if (e > 0.0) h = Kokkos::fmin(h, s.dt_sub_frac*e/(Kokkos::fabs(edot) + SMALL));
    ((h = Kokkos::fmin(h, LimitBySpecies<K>(s, y, e, cell, sub))), ...);
    return h;
  }

  template <int K>
  KOKKOS_INLINE_FUNCTION static Real LimitBySpecies(const SemiImplicitSettings &s,
                                                    const Real *y, const Real e,
                                                    const Cell &cell,
                                                    const Substep &sub) {
    const CD r = Network::template Rates<K>(y, e, cell, sub);
    return s.dt_sub_frac*Kokkos::fmax(y[K], s.yfloor)/
           (Kokkos::fabs(r.c - y[K]*r.d) + SMALL);
  }

  template <int... K>
  KOKKOS_INLINE_FUNCTION static void UpdateInOrder(Real *y, const Real *y_n, const Real e,
                                                   const Cell &cell, const Substep &sub,
                                                   const Real h,
                                                   std::integer_sequence<int, K...>) {
    (UpdateSpecies<K>(y, y_n, e, cell, sub, h), ...);
  }

  template <int K>
  KOKKOS_INLINE_FUNCTION static void UpdateSpecies(Real *y, const Real *y_n, const Real e,
                                                   const Cell &cell, const Substep &sub,
                                                   const Real h) {
    const CD r = Network::template Rates<K>(y, e, cell, sub);
    y[K] = (y_n[K] + h*r.c)/(1.0 + h*r.d);
  }
};

}  // namespace chemistry2

#endif  // CHEMISTRY2_SOLVERS_SEMI_IMPLICIT_HPP_
