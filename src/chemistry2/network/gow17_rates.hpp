#ifndef CHEMISTRY2_NETWORK_GOW17_RATES_HPP_
#define CHEMISTRY2_NETWORK_GOW17_RATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gow17_rates.hpp
//! \brief Rate coefficients of the GOW17 network (Gong, Ostriker & Wolfire 2017), in
//!        two parts: those that depend on the temperature alone (GOW17TemperatureRates,
//!        later read from a table) and the substep's coefficients built from them and
//!        the state (GOW17Rates). From src/chemistry/network/gow17.hpp (UpdateRates_).

#include "athena.hpp"

namespace chemistry2 {
namespace gow17 {

// Two-body reactions (rate coefficients in cm^3 s^-1; times n_H in GOW17Rates)
enum : int {
  I2_H3P_C,       // H3+ + *C -> CH + H2
  I2_H3P_O,       // H3+ + *O -> OH + H2
  I2_H3P_CO,      // H3+ + CO -> HCO+ + H2
  I2_HEP_H2,      // He+ + H2 -> H+ + *He + *H
  I2_HEP_CO,      // He+ + CO -> C+ + *O + *He
  I2_CP_H2,       // C+ + H2 -> CH + *H
  I2_CP_OH,       // C+ + OH -> HCO+
  I2_CH_O,        // CH + *O -> CO + *H
  I2_OH_C,        // OH + *C -> CO + *H
  I2_HEP_E,       // He+ + *e -> *He
  I2_H3P_E,       // H3+ + *e -> H2 + *H
  I2_CP_E,        // C+ + *e -> *C
  I2_HCOP_E,      // HCO+ + *e -> CO + *H
  I2_H2P_H2,      // H2+ + H2 -> H3+ + *H
  I2_HP_E,        // H+ + *e -> *H
  I2_H2_H,        // H2 + *H -> 3 *H
  I2_H2_H2,       // H2 + H2 -> H2 + 2 *H
  I2_H_E,         // *H + *e -> H+ + 2 *e
  I2_H3P_E_3H,    // H3+ + *e -> 3 *H
  I2_HEP_H2_H2P,  // He+ + H2 -> H2+ + *He
  I2_CH_H,        // CH + *H -> H2 + *C
  I2_OH_O,        // OH + *O -> *O + *O + *H
  I2_CP_H2_E,     // C+ + H2 + *e -> *C + *H + *H
  I2_SIP_E,       // Si+ + *e -> *Si
  I2_H3P_O_H2,    // H3+ + *O + *e -> H2 + *O + *H
  I2_HEP_OH,      // He+ + OH -> O+ + *He + *H
  I2_H2P_H,       // H2+ + *H -> H+ + H2
  I2_HP_O,        // H+ + *O -> O+ + *H
  I2_OP_H,        // O+ + *H -> H+ + *O
  I2_OP_H2_OH,    // O+ + H2 -> OH + *H
  I2_OP_H2,       // O+ + H2 -> *O + *H + *H
  N_2BODY
};
// Cosmic-ray reactions (s^-1, times xi_CR / 2e-16 s^-1 in units of the primary rate)
enum : int {ICR_H2, ICR_HE, ICR_H, ICR_C, ICR_CO_PHOTO, ICR_CO, ICR_SI, N_CR};
// Grain-assisted recombination (times n_H Z_d)
enum : int {IGR_H2, IGR_HP, IGR_CP, IGR_HEP, IGR_SIP, N_GR};
// Photo reactions (s^-1, times the field in Draine units)
enum : int {IPH_C, IPH_CHX, IPH_CO, IPH_OHX, IPH_H2, IPH_SI, N_PH};

constexpr Real SMALL_REAL = 1e-50;

//! Coefficients that depend on the temperature alone
struct GOW17TemperatureRates {
  Real T, T_coll;   // rate temperature and collisional temperature [K]
  Real sqrt_T, ln_T;
  Real k2[N_2BODY];  // two-body rate coefficients without n_H and the H2O+ branching;
                     // H2_H, H2_H2 unused (set from the limits below)
  bool coll_active;  // collisional dissociation and ionization on (T_coll > 700 K)
  Real log10_k9l, log10_k9h, log10_k10l, log10_k10h;  // H2 dissociation limits
  Real ncr_h, ncr_h2;                                  // their critical densities
  Real kgr_h2;               // H2 formation on grains [cm^3 s^-1]
  Real gr_tfac[4], gr_texp[4];  // grain recombination: c3 T^c4 and -c5 - c6 ln T

  GOW17TemperatureRates() = default;
  KOKKOS_INLINE_FUNCTION GOW17TemperatureRates(const Real T_rates, const Real T_coll_in,
                                               const bool kgr_h2_const);
};

KOKKOS_INLINE_FUNCTION Real CIIRecombination(const Real T) {
  constexpr Real A = 2.995e-9;
  constexpr Real B = 0.7849;
  constexpr Real T0 = 6.670e-3;
  constexpr Real T1 = 1.943e6;
  constexpr Real C = 0.1597;
  constexpr Real T2 = 4.955e4;
  const Real BN = B + C*Kokkos::exp(-T2/T);
  const Real term1 = Kokkos::sqrt(T/T0);
  const Real term2 = Kokkos::sqrt(T/T1);
  const Real alpha_rr = A/(term1*Kokkos::pow(1.0 + term1, 1.0 - BN)*
                           Kokkos::pow(1.0 + term2, 1.0 + BN));
  // T^(-3/2) = T0^(-3/2) / term1^3
  constexpr Real T0_INV15 = 1.83574032975375144e3;
  const Real alpha_dr = (T0_INV15/(term1*term1*term1))*
                        (6.346e-9*Kokkos::exp(-1.217e1/T) +
                         9.793e-09*Kokkos::exp(-7.38e1/T) +
                         1.634e-06*Kokkos::exp(-1.523e+04/T));
  return alpha_rr + alpha_dr;
}

//! H2 formation on dust grains (TIGRESS-NCR, Kim et al. 2023)
KOKKOS_INLINE_FUNCTION Real H2FormationOnGrains(const Real T, const bool constant) {
  const Real kgr0 = 3.0e-17;
  if (constant) return kgr0;
  // Hollenbach & McKee (1979), Tgr2 = 0, renormalized to ~kgr0 for 50 < T < 200 K
  const Real T2 = T*1e-2;
  return kgr0*Kokkos::sqrt(T2)*2.0/(1 + 0.4*Kokkos::sqrt(T2) + 0.2*T2 + 0.08*T2*T2);
}

KOKKOS_INLINE_FUNCTION GOW17TemperatureRates::GOW17TemperatureRates(
    const Real T_rates, const Real T_coll_in, const bool kgr_h2_const)
    : T(T_rates), T_coll(T_coll_in) {
  sqrt_T = Kokkos::sqrt(T);
  ln_T = Kokkos::log(T);
  const Real logT = Kokkos::log10(T);
  constexpr Real LN10 = 2.30258509299404590e0;

  constexpr Real base[N_2BODY] = {
      1.00,    1.99e-9,  1.7e-9,    1.26e-13, 1.6e-9,        3.3e-13 * 0.7,
      1.00,    7.0e-11,  7.95e-10,  1.0e-11,  4.54e-7,       1.00,
      1.06e-5, 1.76e-9,  2.753e-14, 1.00,     1.00,          1.00,
      8.46e-7, 7.20e-15, 2.81e-11,  3.5e-11,  3.3e-13 * 0.3, 1.46e-10,
      1.99e-9, 1.00,     6.4e-10,   1.00,     1.00,          1.6e-9,
      1.6e-9};
  for (int i = 0; i < N_2BODY; ++i) k2[i] = base[i];

  const Real tp_m0190 = Kokkos::exp(-0.190*ln_T);
  const Real tp_m13 = Kokkos::exp(-1.3*ln_T);
  const Real tp_m0339 = Kokkos::exp(-0.339*ln_T);
  const Real tp_m05 = Kokkos::exp(-0.5*ln_T);
  const Real tp_m052 = Kokkos::exp(-0.52*ln_T);
  const Real tp_m064 = Kokkos::exp(-0.64*ln_T);
  const Real tp_0042 = Kokkos::exp(0.042*ln_T);
  const Real tp_026 = Kokkos::exp(0.26*ln_T);
  const Real tp_m062 = Kokkos::exp(-0.62*ln_T);
  k2[I2_H3P_O] *= tp_m0190;
  k2[I2_H3P_O_H2] *= tp_m0190;
  k2[I2_CP_H2] *= tp_m13;
  k2[I2_CP_H2_E] *= tp_m13;
  k2[I2_OH_C] *= tp_m0339;
  k2[I2_HEP_E] *= tp_m05;
  k2[I2_H3P_E] *= tp_m052;
  k2[I2_H3P_E_3H] *= tp_m052;
  k2[I2_HCOP_E] *= tp_m064;
  k2[I2_H2P_H2] *= tp_0042;
  k2[I2_CH_H] *= tp_026;
  k2[I2_SIP_E] *= tp_m062;

  // H3+ + C -> CH+, CH2+ (Vissapragada et al. 2016)
  constexpr Real LN300 = 5.70378247465620092e0;
  constexpr Real LN315614 = 1.26622752269192862e1;
  constexpr Real LN115188 = 1.16543208551515374e1;
  constexpr Real c_chx[4] = {3.4e-8, 6.97e-9, 1.31e-7, 1.51e-4};
  constexpr Real ti_chx[4] = {7.62, 1.38, 2.66e1, 8.11e3};
  const Real t1_chx = 1.04e-9*Kokkos::exp(2.31e-3*(LN300 - ln_T));
  const Real t2_chx = c_chx[0]*Kokkos::exp(-ti_chx[0]/T) +
                      c_chx[1]*Kokkos::exp(-ti_chx[1]/T) +
                      c_chx[2]*Kokkos::exp(-ti_chx[2]/T) +
                      c_chx[3]*Kokkos::exp(-ti_chx[3]/T);
  k2[I2_H3P_C] *= t1_chx + Kokkos::exp(-1.5*ln_T)*t2_chx;
  k2[I2_HEP_H2] *= Kokkos::exp(-22.5/T);
  const Real exp_m23_T = Kokkos::exp(-23./T);
  k2[I2_CP_H2] *= exp_m23_T;
  k2[I2_CP_H2_E] *= exp_m23_T;
  const Real kida = 0.62 + 45.41/sqrt_T;  // KIDA ion-dipole factor, per unit n_H
  k2[I2_CP_OH] = 9.15e-10*kida;
  k2[I2_OH_C] *= Kokkos::exp(0.108/T);
  k2[I2_HEP_E] *= 11.19 + (-1.676 + (-0.2852 + 0.04433*logT)*logT)*logT;
  k2[I2_CP_E] = CIIRecombination(T);
  k2[I2_H2P_H2] *= Kokkos::exp(-T/46600.);
  k2[I2_HP_E] *= Kokkos::exp(1.5*(LN315614 - ln_T))*
                 Kokkos::pow(1.0 + Kokkos::exp(0.407*(LN115188 - ln_T)), -2.242);
  k2[I2_HEP_OH] = 1.35e-9*kida;
  k2[I2_HP_O] *= (1.1e-11*Kokkos::exp(0.517*ln_T) + 4.0e-10*Kokkos::exp(6.69e-3*ln_T))*
                 Kokkos::exp(-227./T);
  k2[I2_OP_H] *= 4.99e-11*Kokkos::exp(0.405*ln_T) + 7.5e-10*Kokkos::exp(-0.458*ln_T);

  // Collisional dissociation of H2 and ionization of H (Glover & Mac Low 2007)
  constexpr Real TEMP_COLL = 7.0e2;
  coll_active = T_coll > TEMP_COLL;
  if (coll_active) {
    const Real ln_tcoll = Kokkos::log(T_coll);
    const Real log_t4 = Kokkos::log10(T_coll/1.0e4);
    const Real ln_te = Kokkos::log(T_coll*8.6173e-5);
    log10_k9l = Kokkos::log10(6.67e-12*Kokkos::sqrt(T_coll)*
                              Kokkos::exp(-(1. + 63590./T_coll)));
    log10_k9h = Kokkos::log10(3.52e-9*Kokkos::exp(-43900.0/T_coll));
    log10_k10l = Kokkos::log10(5.996e-30*Kokkos::exp(4.1881*ln_tcoll)/
                               Kokkos::pow((1.0 + 6.761e-6*T_coll), 5.6881)*
                               Kokkos::exp(-54657.4/T_coll));
    log10_k10h = Kokkos::log10(1.3e-9*Kokkos::exp(-53300.0/T_coll));
    ncr_h = Kokkos::exp(LN10*(3.0 - 0.416*log_t4 - 0.327*log_t4*log_t4));
    ncr_h2 = Kokkos::exp(LN10*(4.845 - 1.3*log_t4 + 1.62*log_t4*log_t4));
    k2[I2_H_E] *= Kokkos::exp(
        -3.271396786e1 +
        (1.35365560e1 +
         (-5.73932875 +
          (1.56315498 +
           (-2.877056e-1 +
            (3.48255977e-2 +
             (-2.63197617e-3 + (1.11954395e-4 + (-2.03914985e-6)*ln_te)*ln_te)*ln_te)*
                ln_te)*ln_te)*ln_te)*ln_te)*ln_te);
  } else {
    log10_k9l = log10_k9h = log10_k10l = log10_k10h = 0.0;
    ncr_h = ncr_h2 = 0.0;
    k2[I2_H_E] = 0.0;
  }
  k2[I2_H2_H] = 0.0;
  k2[I2_H2_H2] = 0.0;

  kgr_h2 = H2FormationOnGrains(T, kgr_h2_const);

  // Grain-assisted recombination (Weingartner & Draine 2001): the T-only factors
  constexpr Real c[4][7] = {
      {12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5},    // H+
      {45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4},    // C+
      {5.572, 3.185e-7, 1.512, 5.115e3, 3.903e-7, 0.4956, 5.494e-7},    // He+
      {2.166, 5.678e-8, 1.874, 4.375e4, 1.635e-6, 0.8964, 7.538e-5}};   // Si+
  for (int s = 0; s < 4; ++s) {
    gr_tfac[s] = c[s][3]*Kokkos::exp(c[s][4]*ln_T);
    gr_texp[s] = -c[s][5] - c[s][6]*ln_T;
  }
}

//! Grain recombination coefficient c0, c1, c2 of species s (H+, C+, He+, Si+)
KOKKOS_INLINE_FUNCTION void GrainRecCoefficients(const int s, Real &c0, Real &c1,
                                                 Real &c2) {
  constexpr Real c[4][3] = {
      {12.25, 8.074e-6, 1.378},
      {45.58, 6.089e-3, 1.128},
      {5.572, 3.185e-7, 1.512},
      {2.166, 5.678e-8, 1.874}
  };
  c0 = c[s][0];
  c1 = c[s][1];
  c2 = c[s][2];
}

//! The substep's rate coefficients [s^-1 per unit abundance of the partner]
struct GOW17Rates {
  Real k2[N_2BODY];
  Real kcr[N_CR];
  Real kgr[N_GR];
};

//! From the T-only part and the state: x_H2, the ghosts x_H and x_e, n_H, Z_d, the
//! cosmic-ray ionization rate xi_cr [s^-1] and the photoelectric field g_pe [Draine]
KOKKOS_INLINE_FUNCTION GOW17Rates MakeGOW17Rates(const GOW17TemperatureRates &t,
                                                 const Real x_h2, const Real x_h,
                                                 const Real x_e, const Real n_h,
                                                 const Real zd, const Real xi_cr,
                                                 const Real g_pe) {
  GOW17Rates r;
  constexpr Real LN10 = 2.30258509299404590e0;

  constexpr Real kcr_base[N_CR] = {2.0, 1.1, 1.0, 520., 92., 6.52, 8400.};
  for (int i = 0; i < N_CR; ++i) r.kcr[i] = kcr_base[i]*xi_cr;
  const Real kcr_h_fac = 1.15*2*x_h2 + 1.5*x_h;
  r.kcr[ICR_H2] *= kcr_h_fac;
  r.kcr[ICR_H] *= kcr_h_fac;
  r.kcr[ICR_C] *= (2*x_h2 + 3.85/kcr_base[ICR_C]);
  r.kcr[ICR_CO_PHOTO] *= 2*x_h2;
  r.kcr[ICR_SI] *= 2*x_h2;

  for (int i = 0; i < N_2BODY; ++i) r.k2[i] = t.k2[i]*n_h;

  // H2O+ + e branching, for H3+ + O and O+ + H2
  const Real h2oplus_ratio = (x_e < SMALL_REAL)
                                 ? 1.0e10
                                 : 6e-10*x_h2/(5.3e-6/t.sqrt_T*x_e);
  const Real fac_h2 = h2oplus_ratio/(h2oplus_ratio + 1.);
  const Real fac_e = 1./(h2oplus_ratio + 1.);
  r.k2[I2_H3P_O] *= fac_h2;
  r.k2[I2_H3P_O_H2] *= fac_e;
  r.k2[I2_OP_H2_OH] *= fac_h2;
  r.k2[I2_OP_H2] *= fac_e;

  // Collisional dissociation: density blend of the low- and high-density limits
  if (t.coll_active && n_h > SMALL_REAL) {
    const Real div_ncr = x_h/(t.ncr_h + SMALL_REAL) + x_h2/(t.ncr_h2 + SMALL_REAL);
    const Real ncr = (div_ncr < SMALL_REAL) ? 1./SMALL_REAL : 1./div_ncr;
    const Real n2ncr = n_h/ncr;
    r.k2[I2_H2_H] = Kokkos::exp(LN10*(t.log10_k9h*n2ncr/(1. + n2ncr) +
                                      t.log10_k9l/(1. + n2ncr)))*n_h;
    r.k2[I2_H2_H2] = Kokkos::exp(LN10*(t.log10_k10h*n2ncr/(1. + n2ncr) +
                                       t.log10_k10l/(1. + n2ncr)))*n_h;
  } else {
    r.k2[I2_H2_H] = 0.0;
    r.k2[I2_H2_H2] = 0.0;
    r.k2[I2_H_E] = 0.0;
  }

  // Grain-assisted recombination
  r.kgr[IGR_H2] = t.kgr_h2*n_h*zd;
  if (x_e > SMALL_REAL) {
    constexpr Real GPE_LIMIT = 1.0e-10;  // keeps psi finite in the dark
    const Real psi = 1.7*Kokkos::fmax(g_pe, GPE_LIMIT)*t.sqrt_T/n_h/x_e;
    const Real ln_psi = Kokkos::log(psi);
    for (int s = 0; s < 4; ++s) {
      Real c0, c1, c2;
      GrainRecCoefficients(s, c0, c1, c2);
      const Real inner = 1.0 + t.gr_tfac[s]*Kokkos::exp(t.gr_texp[s]*ln_psi);
      r.kgr[IGR_HP + s] = 1.0e-14*c0/(1.0 + c1*Kokkos::exp(c2*ln_psi)*inner)*n_h*zd;
    }
  } else {
    for (int s = 0; s < 4; ++s) r.kgr[IGR_HP + s] = 0.0;
  }
  return r;
}

}  // namespace gow17
}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_GOW17_RATES_HPP_
