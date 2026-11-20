#ifndef CHEMISTRY2_THERMO_DUST_HPP_
#define CHEMISTRY2_THERMO_DUST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust.hpp
//! \brief Collisional energy exchange between gas and dust, from Tigris photchem
//!        (radiative_properties/dust_temperature.hpp): per H, Z_d alpha_gd n_H T^(1/2)
//!        (T - T_d), negative where the dust is the warmer.

#include "athena.hpp"
#include "chemistry2/util/dual.hpp"

namespace chemistry2 {
namespace dust {

//! alpha_gd(T) [erg cm^3 s^-1 K^-3/2] of Hollenbach & McKee (1989) eq. 2.15, an MRN
//! distribution down to 100 Angstrom
template <class S>
KOKKOS_INLINE_FUNCTION S AlphaGasDustHM89(const S T) {
  return 3.8e-33*(1.0 - 0.8*dual::exp(-75.0/T));
}

//! Energy the gas gives the dust per H [erg s^-1 H^-1]; alpha_gd <= 0 selects
//! AlphaGasDustHM89
template <class S>
KOKKOS_INLINE_FUNCTION S GasToDust(const Real n_h, const S T, const Real T_d,
                                   const Real z_d, const Real alpha_gd) {
  const S a = (alpha_gd > 0.0) ? S(alpha_gd) : AlphaGasDustHM89(T);
  return (z_d*n_h)*a*dual::sqrt(T)*(T - T_d);
}

}  // namespace dust
}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_DUST_HPP_
