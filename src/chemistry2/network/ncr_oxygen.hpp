#ifndef CHEMISTRY2_NETWORK_NCR_OXYGEN_HPP_
#define CHEMISTRY2_NETWORK_NCR_OXYGEN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ncr_oxygen.hpp
//! \brief NCR with an oxygen ion ladder: the NCR species (H2, H+, e) first, then O+ to
//!        O^Q+, O I the oxygen remainder. NCR's own functions act on the first three
//!        species; the ladder (ions/ion_ladder.hpp, rates from ions/ion_table.hpp)
//!        advances oxygen after NCR's H2/H+ block, with collisional ionization,
//!        recombination and charge transfer with H and H+ at the substep's T, n_e,
//!        x_H+ and x_H I. The electrons replace NCR's O+ term (x_H+ x_O) by the charge
//!        of the stored oxygen ions. Ion line cooling and the higher ions are not in
//!        yet (P6d).

#include <array>
#include <string>
#include <string_view>
#include <utility>

#include "athena.hpp"
#include "chemistry2/ions/ion_ladder.hpp"
#include "chemistry2/ions/ion_table.hpp"
#include "chemistry2/network/ncr.hpp"

namespace chemistry2 {

namespace ncr_oxygen {
constexpr std::string_view ION_NAMES[8] = {"O+", "O++", "O3+", "O4+", "O5+", "O6+",
                                           "O7+", "O8+"};
template <int Q>
constexpr std::array<SpeciesRow, 3 + Q> Rows() {
  std::array<SpeciesRow, 3 + Q> r = {};
  for (int i = 0; i < 3; ++i) r[i] = NCR::rows[i];
  for (int q = 1; q <= Q; ++q) r[2 + q] = Species(ION_NAMES[q - 1]).Charge(q).O(1);
  return r;
}
}  // namespace ncr_oxygen

//! Q: the top tracked oxygen stage (O^Q+), 1 <= Q <= 8; no higher ions above it
template <int Q>
struct NCROxygen {
  static_assert(Q >= 1 && Q <= 8, "oxygen has stages O+ to O8+");
  using Base = NCR;
  static constexpr std::string_view NAME = "NCR_oxygen";  // <chemistry2>/network
  static constexpr std::array<SpeciesRow, 3 + Q> rows = ncr_oxygen::Rows<Q>();
  static constexpr bool STORES_ELECTRONS = true;
  static constexpr int ROW_EL = Base::IE;
  static constexpr int HYDROGEN_REMAINDER_CHARGE = 0;
  enum : int {IH2 = Base::IH2, IH_PLUS = Base::IH_PLUS, IE = Base::IE, IO1 = 3};
  static constexpr int NSPEC = 3 + Q;
  using UpdateOrder = std::integer_sequence<int, IH2, IO1>;  // H2/H+ block, then oxygen
  using StepLimitingSpecies = std::integer_sequence<int>;
  static constexpr bool MAX_STEP = true;
  static constexpr bool EDOT_DERIVATIVE = true;

  struct Settings {
    Base::Settings base;
    RowTable ions;  // oxygen links, on the NCR table's grid
  };

  static Settings ReadSettings(ParameterInput *pin, const std::string &b) {
    Settings s{Base::ReadSettings(pin, b), RowTable()};
    // the ion data are read only when this chemical network is the one selected
    if (pin->GetOrAddString(b, "network", "") != std::string(NAME)) return s;
    const std::string dir = pin->GetOrAddString(b, "ion_data_dir",
                                                "inputs/chemistry2/ions/");
    s.ions = ions::BuildIonTable(ions::IonRateData(dir), s.base.table.grid, 8, Q);
    return s;
  }

  struct Cell {
    Base::Cell base;
    RowTable ions;
    Real x_o;  // oxygen per H
  };

  KOKKOS_INLINE_FUNCTION static Cell MakeCell(const Real rho, const Real gamma,
                                              const Settings &s, const UnitFactors &u,
                                              const CellInputs &in) {
    return Cell{Base::MakeCell(rho, gamma, s.base, u, in), s.ions,
                ncr::X_O_STD*s.base.zg};
  }

  using Substep = Base::Substep;

  KOKKOS_INLINE_FUNCTION static Real Temperature(const Real *y, const Real e,
                                                 const Cell &c) {
    return Base::Temperature(y, e, c.base);
  }

  //! NCR's substep, with O I (for its cooling) from the stored oxygen ions
  KOKKOS_INLINE_FUNCTION static Substep Prepare(const Real *y, const Real e,
                                                const Cell &c) {
    Substep sub = Base::Prepare(y, e, c.base);
    Real ions = 0.0;
    for (int q = 1; q <= Q; ++q) ions += y[2 + q];
    sub.x_oi = Kokkos::fmax(c.x_o - ions - sub.x_co, ncr::TINY);
    return sub;
  }

  KOKKOS_INLINE_FUNCTION static Real IonCharge(const Real *y) {
    Real q_sum = 0.0;
    for (int q = 1; q <= Q; ++q) q_sum += q*y[2 + q];
    return q_sum;
  }

  template <int K>
  KOKKOS_INLINE_FUNCTION static CD Rates(const Real *y, Real e, const Cell &cl,
                                         const Substep &sub) {
    if constexpr (K < 3) {
      return Base::template Rates<K>(y, e, cl.base, sub);
    } else {
      return CD{0.0, 0.0};  // oxygen ions follow the ladder in Update<IO1>
    }
  }

  template <int K>
  static constexpr bool CUSTOM_UPDATE = (K == IH2 || K == IO1);

  template <int K>
  KOKKOS_INLINE_FUNCTION static void Update(Real *y, const Real *y_n, const Real e,
                                            const Cell &cl, const Substep &sub,
                                            const Real h) {
    if constexpr (K == IH2) {
      Base::template Update<IH2>(y, y_n, e, cl.base, sub, h);
      // NCR counts O+ as x_H+ x_O; the stored oxygen ions replace that term
      y[IE] += IonCharge(y_n) - y[IH_PLUS]*cl.x_o;
    } else {
      static_assert(K == IO1, "NCR_oxygen's custom updates: H2/H+ block, oxygen");
      constexpr int N = Q + 1;
      ions::IonCoefficients k[N];
      ions::LookupIon<N>(cl.ions, sub.slot, k);
      const Real x_hi = Kokkos::fmax(1.0 - 2.0*y[IH2] - y[IH_PLUS], ncr::TINY);
      const ions::ChainCell cc{cl.base.n_h, y[IE]*cl.base.n_h, y[IH_PLUS], x_hi,
                               cl.base.u_time};
      const Real xi_ph[N] = {};
      Real up[N], dn[N];
      ions::ChainRates<N, false>(k, xi_ph, 0.0, false, false, 0.0, cc, up, dn);
      Real xn[N], x[N];
      Real ions_n = 0.0;
      for (int q = 1; q <= Q; ++q) {
        xn[q] = y_n[2 + q];
        ions_n += xn[q];
      }
      xn[0] = Kokkos::fmax(cl.x_o - ions_n, 0.0);
      const Real src[N] = {}, loss[N] = {};
      ions::LadderStep<N>(xn, up, dn, src, loss, h, cl.x_o, x);
      const Real q_old = IonCharge(y);
      for (int q = 1; q <= Q; ++q) y[2 + q] = x[q];
      y[IE] += IonCharge(y) - q_old;
    }
  }

  KOKKOS_INLINE_FUNCTION static Real MaxStep(const Real frac, const Real *y, Real e,
                                             const Cell &cl, const Substep &sub) {
    return Base::MaxStep(frac, y, e, cl.base, sub);
  }

  KOKKOS_INLINE_FUNCTION static Real Edot(const Real *y, const Real e, const Cell &c,
                                          const Substep &sub) {
    return Base::Edot(y, e, c.base, sub);
  }

  KOKKOS_INLINE_FUNCTION static Real EdotWithDerivative(const Real *y, const Real e,
                                                        const Cell &c,
                                                        const Substep &sub,
                                                        Real *dedot_de) {
    return Base::EdotWithDerivative(y, e, c.base, sub, dedot_de);
  }

  KOKKOS_INLINE_FUNCTION static void Closures(Real *, const Real *, const Cell &) {}
};

}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_NCR_OXYGEN_HPP_
