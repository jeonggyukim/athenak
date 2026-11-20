#ifndef CHEMISTRY2_NETWORK_GOW17_TABLE_HPP_
#define CHEMISTRY2_NETWORK_GOW17_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gow17_table.hpp
//! \brief The temperature-only coefficients of GOW17 in one row table: the rate
//!        coefficients of GOW17TemperatureRates and the coefficients of the heating and
//!        cooling terms, interpolated linearly.

#include "athena.hpp"
#include "chemistry2/ions/rates/coll_ion_rate.hpp"
#include "chemistry2/network/gow17_rates.hpp"
#include "chemistry2/rates/row_table.hpp"
#include "chemistry2/thermo/hot_gas.hpp"
#include "chemistry2/thermo/ionized.hpp"
#include "chemistry2/thermo/thermo.hpp"

namespace chemistry2 {
namespace gow17 {

enum : int {
  C_K2 = 0,                              // N_2BODY two-body columns
  C_LOG10_K9L = C_K2 + N_2BODY,
  C_LOG10_K9H, C_LOG10_K10L, C_LOG10_K10H,
  C_NCR_H, C_NCR_H2, C_KGR_H2,
  C_GR_TFAC, C_GR_TEXP = C_GR_TFAC + 4,
  C_CII = C_GR_TEXP + 4,                 // k10e, k10HI, k10H2
  C_CII_BOLTZ = C_CII + 3,
  C_CI = C_CII_BOLTZ + 1,                // 9 rate coefficients
  C_CI_BOLTZ = C_CI + 9,                 // b10, b20, b21
  C_OI = C_CI_BOLTZ + 3,
  C_OI_BOLTZ = C_OI + 9,
  C_H2GEFF = C_OI_BOLTZ + 3,             // geff_H, geff_H2
  C_H2COOL = C_H2GEFF + 2,               // LHI, LH2, LHe, LHplus, Le, LTE
  C_LYA = C_H2COOL + 6,                  // fac, k01
  C_CIE = C_LYA + 2,                     // hot::NCOL CIE columns (Lambda_H,
                                         // Lambda_He, Lambda_metal, xe_He, xe_metal);
                                         // zero without a CIE table
  C_HII_FF = C_CIE + hot::NCOL,          // free-free factor g_ff sqrt(T4) 1.422e-25
  C_HII_REC,                             // recombination energy times alpha_B
                                         // [erg cm^3 s^-1]
  N_COLS
};

// Every column holds the value and is interpolated linearly. At 64 rows per octave the
// worst case is an Arrhenius or Boltzmann factor exp(-x) at the bottom of an octave,
// relative error (x^2 - 2x)/(8 N^2): 0.24% at x = 10, 2.2% at x = 27.6; logarithmic or
// nqt1 storage of the values was measured and was no better at the same cost
// (design/chemistry2/p2-gow17.md, review with Fable, 2026-11-20).
constexpr int ROWS_PER_OCTAVE = 64;

//! Every value of one row at temperature T (analytic)
inline void FillRow(const Real T, const bool kgr_h2_const, Real *row) {
  Real v[N_COLS];
  const GOW17TemperatureRates tr(T, T, kgr_h2_const);
  for (int i = 0; i < N_2BODY; ++i) v[C_K2 + i] = tr.k2[i];
  // the collisional limits are defined at every T; the network applies T_coll > 700 K
  const GOW17TemperatureRates hot(Kokkos::fmax(T, 701.0), Kokkos::fmax(T, 701.0),
                                  kgr_h2_const);
  v[C_LOG10_K9L] = hot.log10_k9l;
  v[C_LOG10_K9H] = hot.log10_k9h;
  v[C_LOG10_K10L] = hot.log10_k10l;
  v[C_LOG10_K10H] = hot.log10_k10h;
  v[C_NCR_H] = hot.ncr_h;
  v[C_NCR_H2] = hot.ncr_h2;
  v[C_K2 + I2_H_E] = hot.k2[I2_H_E];
  v[C_KGR_H2] = tr.kgr_h2;
  for (int s = 0; s < 4; ++s) {
    v[C_GR_TFAC + s] = tr.gr_tfac[s];
    v[C_GR_TEXP + s] = tr.gr_texp[s];
  }
  const Thermo::CIIRateCoefficients cii(T);
  v[C_CII] = cii.k10e;
  v[C_CII + 1] = cii.k10HI;
  v[C_CII + 2] = cii.k10H2;
  v[C_CII_BOLTZ] = Kokkos::exp(-Thermo::E10CII()/(k_boltzmann_cgs*T));
  // C I cooling is cut off above 1e6 K, and its electron fits overflow above ~1e7 K
  const Thermo::CIRateCoefficients ci(Kokkos::fmin(T, 1.0e6));
  const Real ci_k[9] = {ci.k10HI, ci.k20HI, ci.k21HI, ci.k10H2, ci.k20H2, ci.k21H2,
                        ci.k10e, ci.k20e, ci.k21e};
  const Thermo::OIRateCoefficients oi(T);
  const Real oi_k[9] = {oi.k10HI, oi.k20HI, oi.k21HI, oi.k10H2, oi.k20H2, oi.k21H2,
                        oi.k10e, oi.k20e, oi.k21e};
  for (int i = 0; i < 9; ++i) {
    v[C_CI + i] = ci_k[i];
    v[C_OI + i] = oi_k[i];
  }
  const Real kbT = k_boltzmann_cgs*T;
  v[C_CI_BOLTZ] = Kokkos::exp(-Thermo::E10CI()/kbT);
  v[C_CI_BOLTZ + 1] = Kokkos::exp(-Thermo::E20CI()/kbT);
  v[C_CI_BOLTZ + 2] = Kokkos::exp(-Thermo::E21CI()/kbT);
  v[C_OI_BOLTZ] = Kokkos::exp(-Thermo::E10OI()/kbT);
  v[C_OI_BOLTZ + 1] = Kokkos::exp(-Thermo::E20OI()/kbT);
  v[C_OI_BOLTZ + 2] = Kokkos::exp(-Thermo::E21OI()/kbT);
  const Thermo::H2GeffRateCoefficients geff(T);
  v[C_H2GEFF] = geff.geff_H;
  v[C_H2GEFF + 1] = geff.geff_H2;
  const Thermo::H2CoolingCoefficients h2c(T);
  const Real h2[6] = {h2c.LHI, h2c.LH2, h2c.LHe, h2c.LHplus, h2c.Le, h2c.LTE};
  for (int i = 0; i < 6; ++i) v[C_H2COOL + i] = h2[i];
  const Real T4 = T/1.0e4;
  const Real fac = 5.31e-8*Kokkos::pow(T4, 0.15)/(1. + Kokkos::pow(T4/5., 0.65));
  v[C_LYA] = fac;
  v[C_LYA + 1] = fac*Kokkos::exp(-11.84/T4);
  for (int k = 0; k < hot::NCOL; ++k) v[C_CIE + k] = 0.0;  // FillCIE
  v[C_HII_FF] = ionized::HIIFreeFreeFactor(T);
  v[C_HII_REC] = ionized::HIIRecombinationEnergy(T)*tr.k2[I2_HP_E];
  for (int c = 0; c < N_COLS; ++c) row[c] = v[c];
}

//! The CIE columns of one row at T from cie (left at zero without it)
inline void FillCIE(const Real T, const hot::CIESource *cie, Real *row) {
  if (cie == nullptr) return;
  for (int k = 0; k < hot::NCOL; ++k) row[C_CIE + k] = cie->Eval(k, T);
}

//! The GOW17 table, rows_per_octave rows per factor of 2 in T from 1 K to 2^30 K; the
//! He I collisional ionization column from ci_he (Voronov 1997) and the CIE columns
//! from cie, each zero without its source
inline RowTable BuildGOW17Table(const bool kgr_h2_const,
                                const int rows_per_octave = ROWS_PER_OCTAVE,
                                const Rates::CollIonFit *ci_he = nullptr,
                                const hot::CIESource *cie = nullptr) {
  const RowGrid grid = RowGrid::Make(1.0, 1073741824.0, 30*rows_per_octave + 1);
  return BuildRowTable(grid, N_COLS, [&](Real T, Real *row) {
    FillRow(T, kgr_h2_const, row);
    row[C_K2 + I2_HE_E] = ci_he ? ci_he->Rate(T) : 0.0;
    FillCIE(T, cie, row);
  });
}

//! The temperature-only rate coefficients from the table: slot at T, and the
//! collisional columns at T_coll (equal to T unless the rate temperature is capped)
KOKKOS_INLINE_FUNCTION GOW17TemperatureRates LookupRates(const RowTable &t,
                                                         const Slot &s, const Real T,
                                                         const Real T_coll) {
  GOW17TemperatureRates r;
  r.T = T;
  r.T_coll = T_coll;
  r.sqrt_T = Kokkos::sqrt(T);
  r.ln_T = 0.0;  // not used after the table
  for (int i = 0; i < N_2BODY; ++i) r.k2[i] = t.At(s, C_K2 + i);
  r.k2[I2_H2_H] = 0.0;
  r.k2[I2_H2_H2] = 0.0;
  r.coll_active = T_coll > 7.0e2;
  const Slot sc = (T_coll == T) ? s : t.grid.Locate(T_coll);
  r.log10_k9l = t.At(sc, C_LOG10_K9L);
  r.log10_k9h = t.At(sc, C_LOG10_K9H);
  r.log10_k10l = t.At(sc, C_LOG10_K10L);
  r.log10_k10h = t.At(sc, C_LOG10_K10H);
  r.ncr_h = t.At(sc, C_NCR_H);
  r.ncr_h2 = t.At(sc, C_NCR_H2);
  r.k2[I2_H_E] = r.coll_active ? t.At(sc, C_K2 + I2_H_E) : 0.0;
  r.kgr_h2 = t.At(s, C_KGR_H2);
  for (int i = 0; i < 4; ++i) {
    r.gr_tfac[i] = t.At(s, C_GR_TFAC + i);
    r.gr_texp[i] = t.At(s, C_GR_TEXP + i);
  }
  return r;
}

//! Column c at slot s as a Real, or as a dual::Dual whose derivative is the row
//! difference times dw_dT; each row pair is read once
template <class S>
KOKKOS_INLINE_FUNCTION S RowCoef(const RowTable &t, const Slot &s, const Real dw_dT,
                                 const int c) {
  const Real v0 = t.data(s.i0, c);
  const Real dv = t.data(s.i1, c) - v0;
  if constexpr (std::is_same_v<S, dual::Dual>) {
    return dual::Dual(v0 + s.w*dv, dv*dw_dT);
  } else {
    return v0 + s.w*dv;
  }
}

//! The temperature-only coefficients of the cooling terms, at T_cap
template <class S>
struct GOW17ThermoCoefficients {
  Thermo::CIIRateCoefficientsT<S> cii;
  S cii_boltz;
  Thermo::CIRateCoefficientsT<S> ci;
  S ci_boltz[3];
  Thermo::OIRateCoefficientsT<S> oi;
  S oi_boltz[3];
  Thermo::H2CoolingCoefficientsT<S> h2cool;
};

template <class S>
KOKKOS_INLINE_FUNCTION GOW17ThermoCoefficients<S> LookupThermo(const RowTable &t,
                                                               const Slot &s,
                                                               const Real dw_dT) {
  auto at = [&](const int c) { return RowCoef<S>(t, s, dw_dT, c); };
  GOW17ThermoCoefficients<S> c;
  c.cii.k10e = at(C_CII);
  c.cii.k10HI = at(C_CII + 1);
  c.cii.k10H2 = at(C_CII + 2);
  c.cii_boltz = at(C_CII_BOLTZ);
  S *ci[9] = {&c.ci.k10HI, &c.ci.k20HI, &c.ci.k21HI, &c.ci.k10H2, &c.ci.k20H2,
              &c.ci.k21H2, &c.ci.k10e, &c.ci.k20e, &c.ci.k21e};
  S *oi[9] = {&c.oi.k10HI, &c.oi.k20HI, &c.oi.k21HI, &c.oi.k10H2, &c.oi.k20H2,
              &c.oi.k21H2, &c.oi.k10e, &c.oi.k20e, &c.oi.k21e};
  for (int i = 0; i < 9; ++i) {
    *ci[i] = at(C_CI + i);
    *oi[i] = at(C_OI + i);
  }
  for (int i = 0; i < 3; ++i) {
    c.ci_boltz[i] = at(C_CI_BOLTZ + i);
    c.oi_boltz[i] = at(C_OI_BOLTZ + i);
  }
  c.h2cool.LHI = at(C_H2COOL);
  c.h2cool.LH2 = at(C_H2COOL + 1);
  c.h2cool.LHe = at(C_H2COOL + 2);
  c.h2cool.LHplus = at(C_H2COOL + 3);
  c.h2cool.Le = at(C_H2COOL + 4);
  c.h2cool.LTE = at(C_H2COOL + 5);
  return c;
}

}  // namespace gow17
}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_GOW17_TABLE_HPP_
