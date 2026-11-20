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
#include <string_view>
#include <type_traits>
#include <utility>

#include "athena.hpp"
#include "chemistry2/network/network_config.hpp"
#include "parameter_input.hpp"

namespace chemistry2 {

struct SemiImplicitSettings {
  Real dt_sub_frac;    // substep as a fraction of the shortest limiting timescale
  Real yfloor;         // abundance floor in the species timescale y/|f|
  int nsub_max;        // the last allowed substep takes the time left
  bool refresh_rates;  // recompute rates at the new temperature before the species

  static SemiImplicitSettings Read(ParameterInput *pin, const std::string &block) {
    return SemiImplicitSettings{
        pin->GetOrAddReal(block, "semi_implicit_dt_sub_frac", 0.1),
        pin->GetOrAddReal(block, "semi_implicit_yfloor", 1.0e-12),
        pin->GetOrAddInteger(block, "semi_implicit_nsub_max", 100000),
        pin->GetOrAddBoolean(block, "semi_implicit_refresh_rates", true)};
  }
};

//! Limits and tolerance of SolveToEquilibrium
struct EquilibriumSettings {
  Real dt_max;   // longest substep [code time]; the total time is not limited
  int nsub_max;  // maximum number of substeps
  Real tol;      // relative change per substep below which the state has converged
  bool fixed_t;  // species only, at the temperature of the initial state
};

template <class Network>
class SemiImplicit {
 public:
  static constexpr std::string_view NAME = "semi_implicit";  // <chemistry2>/ode_solver
  using Settings = SemiImplicitSettings;
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
      const Real t_left = t_end - t_now;
      const Real h = TakeSubstep(s, cell, t_left, n, y, e);
      // land exactly on t_end on the last substep
      t_now = (h == t_left) ? t_end : t_now + h;
    }
    return n;
  }

  //! Substeps from (y, e) until every stored species and e change by less than q.tol
  //! per substep (relative); with q.fixed_t, the species only, e reset after each
  //! substep to keep the initial temperature. The energy test has the same form as the
  //! species test, so it needs no heating/cooling split from the network.
  //! Substeps much longer than the equilibration time can cycle between two states
  //! around the equilibrium (energy and species updated in turn); a change that
  //! reverses the previous one without halving it halves the longest substep allowed.
  //! Returns the substep count, or -q.nsub_max if it did not converge.
  KOKKOS_INLINE_FUNCTION static int SolveToEquilibrium(const SemiImplicitSettings &s,
                                                       const Cell &cell,
                                                       const EquilibriumSettings &q,
                                                       Real *y, Real &e) {
    Real T0 = 0.0;
    if constexpr (HasEnergy<Network>::value) {
      if (q.fixed_t) T0 = Network::Temperature(y, e, cell);
    } else {
      if (q.fixed_t) Kokkos::abort("chemistry2: fixed_t needs Network::Energy");
    }
    Real dt_cap = q.dt_max, f_prev = 0.0;
    Real d_prev[NSPEC + 1] = {};  // NOLINT(runtime/arrays)
    for (int n = 0; n < q.nsub_max; ++n) {
      Real y_prev[NSPEC];  // NOLINT(runtime/arrays)
      for (int k = 0; k < NSPEC; k++) y_prev[k] = y[k];
      const Real e_prev = e;
      // each substep may take dt_cap; the step limiter shortens it (index 0: Solve's
      // rule that the last substep takes all the time left does not apply)
      const Real h = TakeSubstep(s, cell, dt_cap, 0, y, e, q.fixed_t);
      if constexpr (HasEnergy<Network>::value) {
        if (q.fixed_t) e = Network::Energy(y, T0, cell);
      }
      // relative changes of the stored species and e; converged when the largest is
      // below tol, a cycle when they point against the previous ones
      Real d[NSPEC + 1];  // NOLINT(runtime/arrays)
      for (int k = 0; k < NSPEC; k++) {
        d[k] = (y[k] - y_prev[k])/Kokkos::fmax(y[k], s.yfloor);
      }
      d[NSPEC] = q.fixed_t ? 0.0 : (e - e_prev)/Kokkos::fmax(e, SMALL);
      Real f = 0.0, dot = 0.0;
      for (int k = 0; k <= NSPEC; k++) {
        f = Kokkos::fmax(f, Kokkos::fabs(d[k]));
        dot += d[k]*d_prev[k];
        d_prev[k] = d[k];
      }
      if (f < q.tol) return n + 1;
      // a cycle: reversed and not shrinking; otherwise the cap recovers toward dt_max
      if (dot < 0.0 && f > 0.5*f_prev) {
        dt_cap = 0.5*h;
      } else if (dot > 0.0) {
        dt_cap = Kokkos::fmin(2.0*dt_cap, q.dt_max);
      }
      f_prev = f;
    }
    return -q.nsub_max;
  }

 private:
  //! One substep from (y, e), of at most t_left; n is the substep index, and the last
  //! allowed one (n = nsub_max - 1) takes all of t_left. With species_only, e is not
  //! updated and does not limit the step. Returns the length taken.
  KOKKOS_INLINE_FUNCTION static Real TakeSubstep(const SemiImplicitSettings &s,
                                                 const Cell &cell, const Real t_left,
                                                 const int n, Real *y, Real &e,
                                                 const bool species_only = false) {
    // start of substep: clip negative abundances and energy, keep y^n
    Real y_n[NSPEC];  // NOLINT(runtime/arrays)
    for (int k = 0; k < NSPEC; k++) {
      y_n[k] = Kokkos::fmax(y[k], 0.0);
      y[k] = y_n[k];
    }
    e = Kokkos::fmax(e, 0.0);

    // rates at the current state, then edot and its (non-positive) slope d(edot)/de
    Substep sub = Network::Prepare(y, e, cell);
    Real edot = 0.0, dedot_de = 0.0;
    if (!species_only) {
      if constexpr (HasEdotDerivative<Network>::value) {
        edot = Network::EdotWithDerivative(y, e, cell, sub, &dedot_de);
        dedot_de = Kokkos::fmin(dedot_de, 0.0);
      } else {
        edot = Network::Edot(y, e, cell, sub);
        dedot_de = EnergyDerivative(y, e, cell, sub, edot);
      }
    }
    // substep length: the limiting timescales, or all the time left at nsub_max
    const Real h = (n >= s.nsub_max - 1)
                       ? t_left
                       : ChooseStep(s, y, e, cell, sub, edot, t_left,
                                    typename Network::StepLimitingSpecies{});

    // energy first (linearized backward Euler), then the species in order, then the
    // species set by closure (e.g. electrons by charge conservation)
    if (!species_only) {
      e = Kokkos::fmax(e + h*edot/(1.0 - h*dedot_de), 0.0);
      if (s.refresh_rates) sub = Network::Prepare(y, e, cell);
    }
    UpdateInOrder(y, y_n, e, cell, sub, h, typename Network::UpdateOrder{});
    Network::Closures(y, y_n, cell);
    return h;
  }


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

  //! Shortest of: the time left, dt_sub_frac e/|edot|, each step-limiting species'
  //! limit, and the network's own MaxStep
  template <int... K>
  KOKKOS_INLINE_FUNCTION static Real ChooseStep(const SemiImplicitSettings &s,
                                                const Real *y, const Real e,
                                                const Cell &cell, const Substep &sub,
                                                const Real edot, const Real t_left,
                                                std::integer_sequence<int, K...>) {
    Real h = t_left;
    if (e > 0.0) h = Kokkos::fmin(h, s.dt_sub_frac*e/(Kokkos::fabs(edot) + SMALL));
    ((h = Kokkos::fmin(h, LimitBySpecies<K>(s, y, e, cell, sub))), ...);
    if constexpr (HasMaxStep<Network>::value) {
      h = Kokkos::fmin(h, Network::MaxStep(s.dt_sub_frac, y, e, cell, sub));
    }
    return h;
  }

  //! dt_sub_frac times the timescale max(y_K, yfloor)/|c_K - d_K y_K| of species K
  template <int K>
  KOKKOS_INLINE_FUNCTION static Real LimitBySpecies(const SemiImplicitSettings &s,
                                                    const Real *y, const Real e,
                                                    const Cell &cell,
                                                    const Substep &sub) {
    const CD r = Network::template Rates<K>(y, e, cell, sub);
    return s.dt_sub_frac*Kokkos::fmax(y[K], s.yfloor)/
           (Kokkos::fabs(r.c - y[K]*r.d) + SMALL);
  }

  //! Update the species K... one after another (Gauss-Seidel order)
  template <int... K>
  KOKKOS_INLINE_FUNCTION static void UpdateInOrder(Real *y, const Real *y_n, const Real e,
                                                   const Cell &cell, const Substep &sub,
                                                   const Real h,
                                                   std::integer_sequence<int, K...>) {
    (UpdateSpecies<K>(y, y_n, e, cell, sub, h), ...);
  }

  //! One backward Euler step of species K, or the network's own update for a block
  template <int K>
  KOKKOS_INLINE_FUNCTION static void UpdateSpecies(Real *y, const Real *y_n, const Real e,
                                                   const Cell &cell, const Substep &sub,
                                                   const Real h) {
    if constexpr (HasCustomUpdate<Network, K>()) {
      Network::template Update<K>(y, y_n, e, cell, sub, h);
    } else {
      const CD r = Network::template Rates<K>(y, e, cell, sub);
      y[K] = (y_n[K] + h*r.c)/(1.0 + h*r.d);
    }
  }

  //! A network marks an entry of UpdateOrder whose update is its own (a block such as
  //! CO/HCO+) with `template <int K> static constexpr bool CUSTOM_UPDATE`.
  template <class N, int K, class = void>
  struct CustomUpdateTrait : std::false_type {};
  template <class N, int K>
  struct CustomUpdateTrait<N, K, std::void_t<decltype(N::template CUSTOM_UPDATE<K>)>>
      : std::bool_constant<N::template CUSTOM_UPDATE<K>> {};
  template <class N, int K>
  static constexpr bool HasCustomUpdate() { return CustomUpdateTrait<N, K>::value; }

  //! A network that sets `static constexpr bool MAX_STEP = true` supplies
  //! MaxStep(dt_sub_frac, y, e, cell, sub), its own substep limit [code time]
  template <class N, class = void>
  struct HasMaxStep : std::false_type {};
  template <class N>
  struct HasMaxStep<N, std::void_t<decltype(N::MAX_STEP)>>
      : std::bool_constant<N::MAX_STEP> {};

  //! A network that sets `static constexpr bool EDOT_DERIVATIVE = true` supplies
  //! EdotWithDerivative(y, e, cell, sub, &dedot_de): edot, and d(edot)/de at fixed
  //! species, from one evaluation
  template <class N, class = void>
  struct HasEdotDerivative : std::false_type {};
  template <class N>
  struct HasEdotDerivative<N, std::void_t<decltype(N::EDOT_DERIVATIVE)>>
      : std::bool_constant<N::EDOT_DERIVATIVE> {};

  //! A network with Energy(y, T, cell), the inverse of Temperature, supports fixed_t
  template <class N, class = void>
  struct HasEnergy : std::false_type {};
  template <class N>
  struct HasEnergy<N, std::void_t<decltype(N::Energy(std::declval<const Real *>(),
                                                     Real(),
                                                     std::declval<const Cell &>()))>>
      : std::true_type {};
};

}  // namespace chemistry2

#endif  // CHEMISTRY2_SOLVERS_SEMI_IMPLICIT_HPP_
