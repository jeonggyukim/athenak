#ifndef CHEMISTRY2_THERMO_NCR_TERMS_HPP_
#define CHEMISTRY2_THERMO_NCR_TERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ncr_terms.hpp
//! \brief Cooling terms of the NCR chemical network that GOW17 does not have, from
//!        Tigris photchem (src/photchem/ncr_rates.hpp), each in erg s^-1 per H:
//!        free-free and H+ recombination cooling, and H2 rovibrational cooling of
//!        Moseley et al. (2021) (GOW17 uses Glover & Abel 2008 / Glover 2015).

#include "athena.hpp"
#include "units/units.hpp"

namespace chemistry2 {
namespace ncr_terms {

//! Free-free (Draine 2011 eq. 10.12) and case B recombination cooling of H+;
//! alpha_b [cm^3 s^-1] is the recombination coefficient at T
KOKKOS_INLINE_FUNCTION Real CoolingHII(const Real x_e, const Real x_hii, const Real n_h,
                                       const Real T, const Real alpha_b) {
  const Real a = Kokkos::log(3.9810717e-06*T);
  const Real g_ff = 1.0 + 0.44/(1.0 + 0.058*a*a);
  const Real ff = 1.422e-25*g_ff*Kokkos::sqrt(T*1e-4);
  const Real e_rec = (0.684 - 0.0208*Kokkos::log(T*1e-4))*
                     units::Units::k_boltzmann_cgs*T;
  return (ff + e_rec*alpha_b)*n_h*x_e*x_hii;
}

//! H2 rovibrational cooling, Moseley et al. (2021), collisions with H I and H2
KOKKOS_INLINE_FUNCTION Real CoolingH2Moseley(const Real x_h2, const Real x_hi,
                                             const Real n_h, const Real T) {
  constexpr Real N1 = 50.0, N2 = 450.0, N3 = 25.0, N4 = 900.0;
  const Real T3 = T*1e-3, inv_T3 = 1.0/T3;
  const Real n_h2 = x_h2*n_h, n_hi = x_hi*n_h;
  const Real x1 = n_hi + 5.0*n_h2, x2 = n_hi + 4.5*n_h2;
  const Real x3 = n_hi + 0.75*n_h2, x4 = n_hi + 0.05*n_h2;
  const Real s3 = Kokkos::sqrt(T3), t32 = s3*T3;
  const Real eb = Kokkos::exp(-inv_T3);
  const Real f1 = 1.1e-25*s3*Kokkos::exp(-0.51*inv_T3)*
                  (0.7*x1/(1.0 + x1/N1) + 0.3*x1/(1.0 + x1/(10.0*N1)));
  const Real f2 = 2.0e-25*T3*eb*(0.35*x2/(1.0 + x2/N2) + 0.65*x2/(1.0 + x2/(10.0*N2)));
  const Real f3 = 2.4e-24*t32*eb*eb*(x3/(1.0 + x3/N3));
  const Real f4 = 1.7e-23*t32*(eb*eb)*(eb*eb)*
                  (0.45*x4/(1.0 + x4/N4) + 0.55*x4/(1.0 + x4/(10.0*N4)));
  return x_h2*(f1 + f2 + f3 + f4);
}

//! Nebular metal-line cooling of photoionized gas (Kim et al. 2023), scaled by Z_g
KOKKOS_INLINE_FUNCTION Real CoolingNebular(const Real x_e, const Real x_hii,
                                           const Real n_h, const Real T, const Real zg) {
  constexpr Real A[7] = {-0.0050817, 0.00765822, 0.11832144, -0.50515842, 0.81569592,
                         -0.58648172, 0.69170381};
  const Real l = Kokkos::log(T*1e-4);
  const Real poly = Kokkos::pow(10.0, (((((A[0]*l + A[1])*l + A[2])*l + A[3])*l +
                                        A[4])*l + A[5])*l + A[6]);
  const Real f_red = 1.0/(1.0 + 0.12*Kokkos::pow(x_e*n_h*1e-2, 0.38 - 0.12*l));
  return 3.677602203699553e-21*zg*x_hii*x_e*n_h/Kokkos::sqrt(T)*
         Kokkos::exp(-38585.52/T)*poly*f_red;
}

}  // namespace ncr_terms
}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_NCR_TERMS_HPP_
