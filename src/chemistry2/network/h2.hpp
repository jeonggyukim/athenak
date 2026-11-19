#ifndef CHEMISTRY2_NETWORK_H2_HPP_
#define CHEMISTRY2_NETWORK_H2_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file h2.hpp
//! \brief H2 test network: H2 formation on grains and destruction by cosmic rays, with
//!        gas-dust cooling (Gong et al. 2023, Sec. 4.1).
//!
//!   H2 is updated by backward Euler; atomic H follows from hydrogen conservation,
//!   x_H = x_H,tot - 2 x_H2, with x_H,tot taken at the start of the substep.

#include <array>
#include <utility>

#include "athena.hpp"
#include "chemistry2/network/network_config.hpp"
#include "chemistry2/network/species_table.hpp"
#include "units/units.hpp"

namespace chemistry2 {

struct H2 {
  static constexpr std::array<SpeciesRow, 2> rows = {
      Species("H2").H(2),
      Species("H").H(1)};
  static constexpr bool STORES_ELECTRONS = false;
  static constexpr int ROW_EL = -1;
  static constexpr int HYDROGEN_REMAINDER_CHARGE = 0;
  enum : int {IH2, IH};
  static constexpr int NSPEC = NumSpecies<H2>();

  using UpdateOrder = std::integer_sequence<int, IH2>;
  using StepLimitingSpecies = std::integer_sequence<int, IH2>;

  static constexpr Real K_GR = 3.0e-17;      // H2 formation on grains [cm^3 s^-1]
  static constexpr Real K_CR = 3.0*2.0e-16;  // H2 destruction by cosmic rays [s^-1]
  static constexpr Real X_HE = 0.1;
  static constexpr Real ALPHA_GD = 3.2e-34;  // gas-dust coupling, DESPOTIC
  static constexpr Real T_FLOOR = 1.0;       // no cooling below [K]

  //! Constants of one cell over the hydro step
  struct Cell {
    Real n_h;       // [cm^-3]
    Real gamma;
    Real u_time;    // time unit [s]
    Real u_energy;  // energy density unit [erg cm^-3]
    bool const_cv;  // heat capacity of atomic gas, whatever x_H2
  };

  //! Rate coefficients of one substep; the H2 network has none that depend on T
  struct Substep {};

  KOKKOS_INLINE_FUNCTION static Substep Prepare(const Real *, Real, const Cell &) {
    return Substep{};
  }

  KOKKOS_INLINE_FUNCTION static Real Temperature(const Real *y, Real e, const Cell &c) {
    const Real x_h2 = c.const_cv ? 0.0 : y[IH2];
    const Real x_h2_0 = Kokkos::fmin(Kokkos::fmax(x_h2, 0.0), 0.5);
    const Real cv = units::Units::k_boltzmann_cgs*((1. - 2.*x_h2_0) + x_h2_0 + X_HE)/
                    (c.gamma - 1.0);
    return e*c.u_energy/c.n_h/cv;
  }

  //! de/dt in code units
  KOKKOS_INLINE_FUNCTION static Real Edot(const Real *y, Real e, const Cell &c,
                                          const Substep &) {
    const Real t = Temperature(y, e, c);
    if (t < T_FLOOR) return 0.0;
    const Real dedt = 0.0 - ALPHA_GD*c.n_h*Kokkos::sqrt(t)*t;
    return c.u_time*(dedt*c.n_h/c.u_energy);
  }

  template <int K>
  KOKKOS_INLINE_FUNCTION static CD Rates(const Real *y, Real, const Cell &c,
                                         const Substep &) {
    if constexpr (K == IH2) {
      return CD{K_GR*c.n_h*y[IH]*c.u_time, K_CR*c.u_time};
    } else {
      static_assert(K == IH, "H2 network has species IH2 and IH");
      return CD{2*K_CR*y[IH2]*c.u_time, 2*(K_GR*c.n_h)*c.u_time};
    }
  }

  //! Atomic H from hydrogen conservation
  KOKKOS_INLINE_FUNCTION static void Closures(Real *y, const Real *y_n, const Cell &) {
    const Real x_h_tot = 2.0*y_n[IH2] + y_n[IH];
    y[IH] = Kokkos::fmax(x_h_tot - 2.0*y[IH2], 0.0);
  }
};

}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_H2_HPP_
