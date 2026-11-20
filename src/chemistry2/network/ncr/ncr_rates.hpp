#ifndef CHEMISTRY2_NETWORK_NCR_NCR_RATES_HPP_
#define CHEMISTRY2_NETWORK_NCR_NCR_RATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ncr_rates.hpp
//! \brief Constants and rate coefficients of the NCR chemical network, from Tigris
//!        photchem (src/photchem/ncr_rates.hpp).

#include "athena.hpp"

namespace chemistry2 {
namespace ncr {

constexpr Real X_C_STD = 1.6e-4;              // C per H at Z_g = 1
constexpr Real X_O_STD = 3.2e-4;              // O per H at Z_g = 1
constexpr Real X_EL_MAX = 1.2006199779862501;  // largest free-electron fraction
constexpr Real TEMP_HOT0 = 2.0e4;             // [K], grain recombination blend
constexpr Real TEMP_HOT1 = 3.5e4;             // [K]
constexpr Real TINY = 1.0e-20;

//! Case B recombination coefficient of H+ [cm^3 s^-1] (Hui & Gnedin 1997)
KOKKOS_INLINE_FUNCTION Real AlphaBHII(const Real T) {
  const Real bb = 315614.0/T, cc = 115188.0/T;
  return 2.753e-14*Kokkos::pow(bb, 1.5)*Kokkos::pow(1.0 + Kokkos::pow(cc, 0.407), -2.242);
}

//! Case A recombination coefficient of H+ [cm^3 s^-1] (Hui & Gnedin 1997)
KOKKOS_INLINE_FUNCTION Real AlphaAHII(const Real T) {
  const Real lam = 315614.0/T;
  return 1.269e-13*Kokkos::pow(lam, 1.503)/
         Kokkos::pow(1.0 + Kokkos::pow(lam/0.522, 0.470), 1.923);
}

//! Collisional ionization of H I by electrons [cm^3 s^-1], zero below 3000 K
KOKKOS_INLINE_FUNCTION Real CollIonHI(const Real T) {
  if (T <= 3.0e3) return 0.0;
  const Real l = Kokkos::log(T*8.6173e-5);
  return Kokkos::exp(-3.271396786e1 + (1.35365560e1 + (-5.73932875 + (1.56315498 +
      (-2.877056e-1 + (3.48255977e-2 + (-2.63197617e-3 + (1.11954395e-4 +
      (-2.03914985e-6)*l)*l)*l)*l)*l)*l)*l)*l);
}

//! Radiative plus dielectronic recombination of C+ [cm^3 s^-1] (iCII_rec_rate = 2)
KOKKOS_INLINE_FUNCTION Real AlphaCII(const Real T) {
  const Real bn = 0.7849 + 0.1597*Kokkos::exp(-4.955e4/T);
  const Real t1 = Kokkos::sqrt(T/6.670e-3), t2 = Kokkos::sqrt(T/1.943e6);
  const Real rr = 2.995e-9/(t1*Kokkos::pow(1.0 + t1, 1.0 - bn)*
                            Kokkos::pow(1.0 + t2, 1.0 + bn));
  const Real dr = Kokkos::pow(T, -1.5)*(6.346e-9*Kokkos::exp(-1.217e1/T) +
                                        9.793e-9*Kokkos::exp(-7.38e1/T) +
                                        1.634e-6*Kokkos::exp(-1.523e4/T));
  return rr + dr;
}

//! Grain-assisted recombination (Weingartner & Draine 2001 form), coefficients c
KOKKOS_INLINE_FUNCTION Real GrainRec(const Real *c, const Real T, const Real lnT,
                                     const Real psi) {
  const Real inner = 1.0 + c[3]*Kokkos::pow(T, c[4])*Kokkos::pow(psi, -c[5] - c[6]*lnT);
  return 1.0e-14*c[0]/(1.0 + c[1]*Kokkos::pow(psi, c[2])*inner);
}

}  // namespace ncr
}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_NCR_NCR_RATES_HPP_
