#ifndef CHEMISTRY2_THERMO_IONIZED_HPP_
#define CHEMISTRY2_THERMO_IONIZED_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionized.hpp
//! \brief Cooling of warm photoionized gas, from Tigris photchem (src/photchem/
//!        ncr_rates.hpp): H+ free-free and case B recombination, and the nebular
//!        metal-line fit. The temperature-only factors are separate functions so that
//!        rate tables and direct evaluation use the same expressions.

#include "athena.hpp"
#include "units/units.hpp"

namespace chemistry2 {
namespace ionized {

//! Free-free cooling of H+ per n_e n_H+ [erg cm^3 s^-1] (Draine 2011 eq. 10.12)
KOKKOS_INLINE_FUNCTION Real HIIFreeFreeFactor(const Real T) {
  const Real a = Kokkos::log(3.9810717e-06*T);
  return 1.422e-25*(1.0 + 0.44/(1.0 + 0.058*a*a))*Kokkos::sqrt(T*1e-4);
}

//! Mean kinetic energy of the electrons recombining to H in case B [erg] (Draine 2011
//! eq. 27.23, errata)
KOKKOS_INLINE_FUNCTION Real HIIRecombinationEnergy(const Real T) {
  return (0.684 - 0.0208*Kokkos::log(T*1e-4))*units::Units::k_boltzmann_cgs*T;
}

//! Free-free and case B recombination cooling of H+ [erg s^-1 per H]; alpha_b
//! [cm^3 s^-1] is the recombination coefficient at T
KOKKOS_INLINE_FUNCTION Real CoolingHII(const Real x_e, const Real x_hii, const Real n_h,
                                       const Real T, const Real alpha_b) {
  return (HIIFreeFreeFactor(T) + HIIRecombinationEnergy(T)*alpha_b)*n_h*x_e*x_hii;
}

//! Temperature part of the nebular fit: exp(-38585.52/T) poly(ln T4)/sqrt(T)
KOKKOS_INLINE_FUNCTION Real NebularTFactor(const Real T) {
  constexpr Real A[7] = {-0.0050817, 0.00765822, 0.11832144, -0.50515842, 0.81569592,
                         -0.58648172, 0.69170381};
  const Real l = Kokkos::log(T*1e-4);
  const Real poly = Kokkos::pow(10.0, (((((A[0]*l + A[1])*l + A[2])*l + A[3])*l +
                                        A[4])*l + A[5])*l + A[6]);
  return Kokkos::exp(-38585.52/T)*poly/Kokkos::sqrt(T);
}

//! Nebular metal-line cooling of photoionized gas (Kim et al. 2023), scaled by Z_g
//! [erg s^-1 per H]
KOKKOS_INLINE_FUNCTION Real CoolingNebular(const Real x_e, const Real x_hii,
                                           const Real n_h, const Real T, const Real zg) {
  const Real l = Kokkos::log(T*1e-4);
  const Real f_red = 1.0/(1.0 + 0.12*Kokkos::pow(x_e*n_h*1e-2, 0.38 - 0.12*l));
  return 3.677602203699553e-21*zg*x_hii*x_e*n_h*NebularTFactor(T)*f_red;
}

}  // namespace ionized
}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_IONIZED_HPP_
