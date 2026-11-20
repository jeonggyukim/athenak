#ifndef CHEMISTRY2_IONS_ION_TABLE_HPP_
#define CHEMISTRY2_IONS_ION_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion_table.hpp
//! \brief The rate coefficients of an ion ladder on the row-table temperature grid: per
//!        link q -> q+1 of an element, {CI_q, CT_ion_q, rec_{q+1}, CT_rec_{q+1}}
//!        [cm^3 s^-1], contiguous per element, built on the host from the Tigris
//!        readers (Voronov 1997, Badnell 2023, Kingdon & Ferland 1996).

#include <string>

#include "athena.hpp"
#include "chemistry2/ions/ion_ladder.hpp"
#include "chemistry2/ions/rates/coll_ion_rate.hpp"
#include "chemistry2/ions/rates/ct_rate.hpp"
#include "chemistry2/ions/rates/ion_threshold.hpp"
#include "chemistry2/ions/rates/recomb_rate.hpp"
#include "chemistry2/util/row_table.hpp"

namespace chemistry2 {
namespace ions {

constexpr int LINK_COLS = 4;  // CI_q, CT_ion_q, rec_{q+1}, CT_rec_{q+1}
enum : int { L_CI, L_CT_ION, L_REC, L_CT_REC };

//! The rate readers, loaded from inputs/chemistry2/ions
struct IonRateData {
  Rates::RecombRate rec;
  Rates::CollIonRate ci;
  Rates::ChargeTransferRate ct;
  IonThreshold th;  // ionization energies (Verner et al. 1996)

  explicit IonRateData(const std::string &dir) {
    rec.Load(dir + "badnell_rr_2023.dat", dir + "badnell_dr_C_2023.dat",
             dir + "badnell_dr_E_2023.dat");
    ci.Load(dir + "voronov97_coll_ion.dat");
    ct.LoadRecomb(dir + "kingdon_ferland96_ct_rec.dat");
    ct.LoadIon(dir + "kingdon_ferland96_ct_ion.dat");
    th.Load(dir + "verner96_photx.dat");
  }
};

//! Table of the n_links links of element Z (stages 0 .. n_links), on grid
inline RowTable BuildIonTable(const IonRateData &d, const RowGrid &grid, const int z,
                              const int n_links) {
  return BuildRowTable(grid, n_links*LINK_COLS, [&](Real T, Real *row) {
    for (int q = 0; q < n_links; ++q) {
      Real *r = row + q*LINK_COLS;
      r[L_CI] = d.ci.Rate(z, z - q, T);
      r[L_CT_ION] = d.ct.CtIon(z, z - q, T);        // X^q + H+ -> X^(q+1) + H
      r[L_REC] = d.rec.Rate(z, z - q - 1, T);       // X^(q+1) + e -> X^q
      r[L_CT_REC] = d.ct.CtRec(z, z - q - 1, T);    // X^(q+1) + H -> X^q + H+
    }
  });
}

//! The IonCoefficients of NT stages (NT - 1 links) of one element at slot s, in the
//! layout ChainRates reads: k[q].ci, ct_ion of stage q, k[q].rec, ct_rec of stage q
template <int NT>
KOKKOS_INLINE_FUNCTION void LookupIon(const RowTable &t, const Slot &s,
                                      IonCoefficients *k) {
  k[0].rec = 0.0;
  k[0].ct_rec = 0.0;
  for (int q = 0; q < NT - 1; ++q) {
    const int c = q*LINK_COLS;
    k[q].ci = t.At(s, c + L_CI);
    k[q].ct_ion = t.At(s, c + L_CT_ION);
    k[q + 1].rec = t.At(s, c + L_REC);
    k[q + 1].ct_rec = t.At(s, c + L_CT_REC);
  }
  k[NT - 1].ci = 0.0;
  k[NT - 1].ct_ion = 0.0;
}

}  // namespace ions
}  // namespace chemistry2

#endif  // CHEMISTRY2_IONS_ION_TABLE_HPP_
