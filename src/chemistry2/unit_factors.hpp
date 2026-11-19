#ifndef CHEMISTRY2_UNIT_FACTORS_HPP_
#define CHEMISTRY2_UNIT_FACTORS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file unit_factors.hpp
//! \brief Code-unit factors the chemistry kernel needs, computed once on the host.
//!
//!   Every conversion in the kernel is one multiplication by a factor or by its inverse;
//!   no unit getter is called and no constant divides on the device.

#include "athena.hpp"
#include "units/units.hpp"

namespace chemistry2 {

struct UnitFactors {
  Real nh_per_density;  // n_H [cm^-3] per code density: density_cgs / (mu_H m_H)
  Real density_per_nh;  // code density per n_H [cm^-3]
  Real time_cgs;        // code time [s]
  Real energy_cgs;      // code energy density [erg cm^-3]
  Real energy_per_cgs;  // code energy density per erg cm^-3
};

//! From the code units in cgs and the mean mass per H in units of m_H
KOKKOS_INLINE_FUNCTION UnitFactors MakeUnitFactors(const Real density_cgs,
                                                   const Real time_cgs,
                                                   const Real energy_cgs,
                                                   const Real mu_h) {
  const Real nh_per_density = density_cgs/(mu_h*units::Units::hydrogen_mass_cgs);
  return UnitFactors{nh_per_density, 1.0/nh_per_density, time_cgs, energy_cgs,
                     1.0/energy_cgs};
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_UNIT_FACTORS_HPP_
