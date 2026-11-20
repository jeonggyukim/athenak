#ifndef CHEMISTRY2_THERMO_H2_COOLING_MOSELEY_HPP_
#define CHEMISTRY2_THERMO_H2_COOLING_MOSELEY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file h2_cooling_moseley.hpp
//! \brief H2 rovibrational cooling of Moseley et al. (2021, MNRAS 500, 3290), eqs. 17-21,
//!        collisions with H I and H2, fitted for T <= 5000 K. The four temperature
//!        factors are a separate function so that rate tables and direct evaluation use
//!        the same expressions.

#include "athena.hpp"

namespace chemistry2 {
namespace h2_moseley {

//! The four temperature factors of eq. 17 [erg s^-1]
KOKKOS_INLINE_FUNCTION void TFactors(const Real T, Real *f) {
  const Real T3 = T*1e-3, inv_T3 = 1.0/T3;
  const Real s3 = Kokkos::sqrt(T3), t32 = s3*T3;
  const Real eb = Kokkos::exp(-inv_T3);
  f[0] = 1.1e-25*s3*Kokkos::exp(-0.51*inv_T3);
  f[1] = 2.0e-25*T3*eb;
  f[2] = 2.4e-24*t32*eb*eb;
  f[3] = 1.7e-23*t32*(eb*eb)*(eb*eb);
}

//! The four density factors of eqs. 17-21 [cm^-3] at n(H I) and n(H2) [cm^-3]
KOKKOS_INLINE_FUNCTION void DensityFactors(const Real n_hi, const Real n_h2, Real *d) {
  constexpr Real N1 = 50.0, N2 = 450.0, N3 = 25.0, N4 = 900.0;
  const Real x1 = n_hi + 5.0*n_h2, x2 = n_hi + 4.5*n_h2;
  const Real x3 = n_hi + 0.75*n_h2, x4 = n_hi + 0.05*n_h2;
  d[0] = 0.7*x1/(1.0 + x1/N1) + 0.3*x1/(1.0 + x1/(10.0*N1));
  d[1] = 0.35*x2/(1.0 + x2/N2) + 0.65*x2/(1.0 + x2/(10.0*N2));
  d[2] = x3/(1.0 + x3/N3);
  d[3] = 0.45*x4/(1.0 + x4/N4) + 0.55*x4/(1.0 + x4/(10.0*N4));
}

//! H2 rovibrational cooling [erg s^-1 per H]; x_h2 = n(H2)/n_H
KOKKOS_INLINE_FUNCTION Real Cooling(const Real x_h2, const Real x_hi, const Real n_h,
                                    const Real T) {
  Real f[4], d[4];
  TFactors(T, f);
  DensityFactors(x_hi*n_h, x_h2*n_h, d);
  return x_h2*(f[0]*d[0] + f[1]*d[1] + f[2]*d[2] + f[3]*d[3]);
}

}  // namespace h2_moseley
}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_H2_COOLING_MOSELEY_HPP_
