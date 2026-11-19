#ifndef CHEMISTRY2_IONS_ION_LADDER_HPP_
#define CHEMISTRY2_IONS_ION_LADDER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion_ladder.hpp
//! \brief The ionization stages of one element as a chain, ported from Tigris photchem
//!        (src/photchem/ions/ion_ladder.hpp): one backward Euler step of N stages, and
//!        the rates of a chain whose stages are ionized by electrons, charge transfer
//!        with H+ and photons and recombine with electrons and by charge transfer with
//!        H I.
//!
//!   For stage q, without molecular terms,
//!     dx_q/dt = I_{q-1} x_{q-1} - (I_q + R_q) x_q + R_{q+1} x_{q+1},
//!   with I the ionization and R the recombination rate. N is a template argument, so
//!   the arrays are sized at compile time and the loops are unrolled.

#include "athena.hpp"

namespace chemistry2 {
namespace ions {

//! One backward Euler step for the N stages of one element, rates held fixed over the
//! step. Stage q ionizes at up[q] and recombines at dn[q]; src[q] and loss[q] are its
//! formation from and destruction to molecules. (I + hM) x = x^n + h src is solved by the
//! Thomas algorithm: M is diagonally dominant with non-positive off-diagonal elements,
//! so every pivot is positive and each stage follows from its own balance. total is the
//! element abundance; the round-off of element conservation goes into the most abundant
//! stage, and nothing is changed if that is the neutral (kept by element conservation
//! in the host).
template <int N>
KOKKOS_INLINE_FUNCTION void LadderStep(const Real *xn, const Real *up, const Real *dn,
                                       const Real *src, const Real *loss, const Real h,
                                       const Real total, Real *x) {
  static_assert(N >= 2, "a chain has at least two stages");
  Real cp[N], dp[N];  // NOLINT(runtime/arrays)
  for (int q = 0; q < N; ++q) {
    const Real sub = (q > 0) ? -h*up[q - 1] : 0.0;
    const Real sup = (q < N - 1) ? -h*dn[q + 1] : 0.0;
    const Real diag = 1.0 + h*(up[q] + dn[q] + loss[q]);
    const Real rhs = xn[q] + h*src[q];
    const Real inv_den = 1.0/((q > 0) ? diag - sub*cp[q - 1] : diag);
    cp[q] = sup*inv_den;
    dp[q] = ((q > 0) ? rhs - sub*dp[q - 1] : rhs)*inv_den;
  }
  x[N - 1] = dp[N - 1];
  for (int q = N - 2; q >= 0; --q) x[q] = dp[q] - cp[q]*x[q + 1];
  int qmax = 0;
  for (int q = 1; q < N; ++q) {
    if (x[q] > x[qmax]) qmax = q;
  }
  if (qmax > 0) {
    Real rest = total;
    for (int q = 0; q < N; ++q) {
      rest += h*(src[q] - loss[q]*x[q]);
      if (q != qmax) rest -= x[q];
    }
    x[qmax] = Kokkos::fmax(rest, 0.0);
  }
}

//! Rate coefficients of one ion at the cell temperature [cm^3 s^-1]: collisional
//! ionization, charge-transfer ionization by H+, radiative plus dielectronic
//! recombination, charge-transfer recombination by H I
struct IonCoefficients {
  Real ci, ct_ion, rec, ct_rec;
};

//! The cell state a chain reads: n_H and n_e [cm^-3], x_H+ and x_H I per H, and u, the
//! factor that converts a rate [s^-1] to the host's time unit
struct ChainCell {
  Real n_h, n_e, x_hp, x_h0, u;
};

//! Ionization and recombination rates of a chain of NT tracked stages, neutral first,
//! plus, when HIGH, the higher ions of the element above the top tracked stage:
//! up[q] = u (k_ci n_e + k_ct,ion n_H x_H+ + xi_ph,q) below the top (up0_extra added to
//! the neutral when has_up0_extra); with HIGH the top ionizes into the higher ions and
//! they return at u high_return, both only when high_on. dn[q] = u (k_rec n_e +
//! k_ct,rec n_H x_H I) for q >= 1. N = NT + 1 with HIGH, NT without.
template <int NT, bool HIGH>
KOKKOS_INLINE_FUNCTION void ChainRates(const IonCoefficients *k, const Real *xi_ph,
                                       const Real up0_extra, const bool has_up0_extra,
                                       const bool high_on, const Real high_return,
                                       const ChainCell &c, Real *up, Real *dn) {
  constexpr int N = HIGH ? NT + 1 : NT;
  for (int q = 0; q < NT - 1; ++q) {
    Real a = k[q].ci*c.n_e + k[q].ct_ion*c.n_h*c.x_hp + xi_ph[q];
    if (q == 0 && has_up0_extra) a += up0_extra;
    up[q] = c.u*a;
  }
  up[NT - 1] = 0.0;
  if constexpr (HIGH) {
    if (high_on) {
      up[NT - 1] = c.u*(k[NT - 1].ci*c.n_e + k[NT - 1].ct_ion*c.n_h*c.x_hp +
                        xi_ph[NT - 1]);
    }
    up[N - 1] = 0.0;
  }
  dn[0] = 0.0;
  for (int q = 1; q < NT; ++q) {
    dn[q] = c.u*(k[q].rec*c.n_e + k[q].ct_rec*c.n_h*c.x_h0);
  }
  if constexpr (HIGH) dn[N - 1] = high_on ? c.u*high_return : 0.0;
}

}  // namespace ions
}  // namespace chemistry2

#endif  // CHEMISTRY2_IONS_ION_LADDER_HPP_
