#ifndef CHEMISTRY2_NETWORK_NCR_TABLE_HPP_
#define CHEMISTRY2_NETWORK_NCR_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ncr_table.hpp
//! \brief The NCR rate table: GOW17's columns (the shared fine-structure and Ly-alpha
//!        coefficients are read through gow17::LookupThermo), then NCR's temperature-only
//!        rate coefficients and cooling factors. Same grid as the GOW17 table.

#include "athena.hpp"
#include "chemistry2/network/gow17_table.hpp"
#include "chemistry2/network/ncr_rates.hpp"
#include "units/units.hpp"
#include "chemistry2/rates/row_table.hpp"
#include "chemistry2/thermo/hot_gas.hpp"

namespace chemistry2 {
namespace ncr {

enum : int {
  C_ALPHA_B = gow17::N_COLS,  // H+ case B recombination [cm^3 s^-1]
  C_K_COLL_HI,                 // H I collisional ionization [cm^3 s^-1]
  C_ALPHA_CII,                 // C+ radiative + dielectronic recombination
  C_K_CH2P,                    // CH2+ channel 3.3e-13 T^-1.3 exp(-23/T)
  C_KGR_FAC,                   // HM79 factor of H2 formation on grains
  C_L9L, C_L9H, C_L10L, C_L10H,  // log10 of the H2 collisional dissociation limits
  C_NCR_H2, C_NCR_HI,          // their critical densities [cm^-3]
  C_K1619,                     // molecular-ion recombination for x_e,mol
  C_MOS,                       // 4 T-only factors of the Moseley et al. H2 cooling
  C_HII_FF = C_MOS + 4,        // free-free factor g_ff sqrt(T4) 1.422e-25
  C_HII_REC,                   // recombination energy times alpha_B [erg cm^3 s^-1]
  C_NEB,                       // nebular T-part exp(-38585.52/T) poly / sqrt(T)
  C_ALPHA_A,                   // H+ case A recombination [cm^3 s^-1]
  C_CIE,                       // hot::NCOL CIE columns (Lambda_He, Lambda_metal,
                               // xe_He, xe_metal); zero without a CIE table
  N_NCR_COLS = C_CIE + hot::NCOL
};

//! Fill one row at T: GOW17's columns, then NCR's
inline void FillNCRRow(const Real T, Real *row) {
  gow17::FillRow(T, false, row);
  row[C_ALPHA_B] = AlphaBHII(T);
  row[C_K_COLL_HI] = CollIonHI(T);
  row[C_ALPHA_CII] = AlphaCII(T);
  row[C_K_CH2P] = 3.3e-13*Kokkos::pow(T, -1.3)*Kokkos::exp(-23.0/T);
  const Real T2 = T*1e-2;
  row[C_KGR_FAC] = Kokkos::sqrt(T2)*2.0/(1.0 + 0.4*Kokkos::sqrt(T2) + 0.2*T2 +
                                         0.08*T2*T2);
  const Real Tc = Kokkos::fmax(T, 7.0e2);  // the limits are used above 700 K only
  const Real inv_T = 1.0/Tc;
  const Real lt4 = Kokkos::log10(Tc*1e-4);
  row[C_L9L] = Kokkos::log10(6.67e-12*Kokkos::sqrt(Tc)*
                             Kokkos::exp(-(1.0 + 63590.0*inv_T)));
  row[C_L9H] = Kokkos::log10(3.52e-9*Kokkos::exp(-43900.0*inv_T));
  row[C_L10L] = Kokkos::log10(5.996e-30*Kokkos::pow(Tc, 4.1881)/
                              Kokkos::pow(1.0 + 6.761e-6*Tc, 5.6881)*
                              Kokkos::exp(-54657.4*inv_T));
  row[C_L10H] = Kokkos::log10(1.3e-9*Kokkos::exp(-53300.0*inv_T));
  row[C_NCR_H2] = Kokkos::pow(10.0, 4.845 - 1.3*lt4 + 1.62*lt4*lt4);
  row[C_NCR_HI] = Kokkos::pow(10.0, 3.0 - 0.416*lt4 - 0.327*lt4*lt4);
  row[C_K1619] = 1.0e-7*Kokkos::pow(T*1e-2, -0.5);
  const Real T3 = T*1e-3, inv_T3 = 1.0/T3, s3 = Kokkos::sqrt(T3);
  const Real eb = Kokkos::exp(-inv_T3);
  row[C_MOS] = 1.1e-25*s3*Kokkos::exp(-0.51*inv_T3);
  row[C_MOS + 1] = 2.0e-25*T3*eb;
  row[C_MOS + 2] = 2.4e-24*s3*T3*eb*eb;
  row[C_MOS + 3] = 1.7e-23*s3*T3*(eb*eb)*(eb*eb);
  const Real a = Kokkos::log(3.9810717e-06*T);
  row[C_HII_FF] = 1.422e-25*(1.0 + 0.44/(1.0 + 0.058*a*a))*Kokkos::sqrt(T*1e-4);
  row[C_HII_REC] = (0.684 - 0.0208*Kokkos::log(T*1e-4))*units::Units::k_boltzmann_cgs*T*
                   AlphaBHII(T);
  constexpr Real A[7] = {-0.0050817, 0.00765822, 0.11832144, -0.50515842, 0.81569592,
                         -0.58648172, 0.69170381};
  const Real l = Kokkos::log(T*1e-4);
  const Real poly = Kokkos::pow(10.0, (((((A[0]*l + A[1])*l + A[2])*l + A[3])*l +
                                        A[4])*l + A[5])*l + A[6]);
  row[C_NEB] = Kokkos::exp(-38585.52/T)*poly/Kokkos::sqrt(T);
  row[C_ALPHA_A] = AlphaAHII(T);
}

//! The NCR table; with cie, its columns at each row temperature (else zero)
inline RowTable BuildNCRTable(const int rows_per_octave = gow17::ROWS_PER_OCTAVE,
                              const hot::CIESource *cie = nullptr) {
  const RowGrid grid = RowGrid::Make(1.0, 1073741824.0, 30*rows_per_octave + 1);
  return BuildRowTable(grid, N_NCR_COLS, [&](Real T, Real *row) {
    FillNCRRow(T, row);
    for (int k = 0; k < hot::NCOL; ++k) row[C_CIE + k] = cie ? cie->Eval(k, T) : 0.0;
  });
}

}  // namespace ncr
}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_NCR_TABLE_HPP_
