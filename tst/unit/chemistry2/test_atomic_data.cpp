//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_atomic_data.cpp
//! \brief The atomic data read by the chemical networks: the He I collisional
//!        ionization fit (Voronov 1997) and ionization energies (Verner et al. 1996)
//!        against the file rows, and the GOW17 table column filled from the fit.

#include <cmath>
#include <cstdio>
#include <string>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/ions/rates/coll_ion_rate.hpp"
#include "chemistry2/ions/rates/ion_threshold.hpp"
#include "chemistry2/network/gow17/gow17_table.hpp"

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  std::string dir(__FILE__);
  dir = dir.substr(0, dir.find_last_of('/')) + "/../../../inputs/chemistry2/ions/";

  // the He I row of voronov97_coll_ion.dat: 1 1 24.6 0 1.75e-08 0.18 0.35
  Rates::CollIonRate ci;
  ci.Load(dir + "voronov97_coll_ion.dat");
  const Rates::CollIonFit he = ci.Get(2, 2);
  report.Expect(he.de_ev == 24.6 && he.p == 0.0 && he.a == 1.75e-8 && he.x == 0.18 &&
                he.k == 0.35, "He I Voronov row read as in the file");
  // H I row: 0 0 13.6 0 2.91e-08 0.232 0.39
  const Rates::CollIonFit h = ci.Get(1, 1);
  report.Expect(h.de_ev == 13.6 && h.a == 2.91e-8 && h.x == 0.232 && h.k == 0.39,
                "H I Voronov row read as in the file");

  // ionization energies: verner96_photx.dat rows 1 1 1.360E+01 and 2 2 2.459E+01
  chemistry2::ions::IonThreshold th;
  th.Load(dir + "verner96_photx.dat");
  report.Expect(th.Ev(1, 1) == 13.60 && th.Ev(2, 2) == 24.59 && th.Ev(2, 1) == 54.42,
                "Verner thresholds H I, He I, He II");
  report.ExpectNear(th.Erg(2, 2), 24.59*1.602176634e-12, 1e-15, 0.0,
                    "He I threshold in erg");

  // the fit at a few temperatures against the formula written out here
  {
    double worst = 0.0;
    for (const double T : {1.0e4, 3.0e4, 1.0e5, 1.0e6, 1.0e8}) {
      const double u = 24.6*1.602176634e-12/(1.3806488e-16*T);
      const double want = 1.75e-8/(0.18 + u)*std::pow(u, 0.35)*std::exp(-u);
      worst = std::fmax(worst, std::fabs(he.Rate(T)/want - 1.0));
    }
    report.ExpectNear(worst, 0.0, 0.0, 1e-13, "He I rate = Voronov formula");
  }

  // the GOW17 table column equals the fit at the row temperatures, and is zero without
  // the fit
  {
    namespace g = chemistry2::gow17;
    const chemistry2::RowTable t = g::BuildGOW17Table(false, g::ROWS_PER_OCTAVE, &he);
    const chemistry2::RowTable t0 = g::BuildGOW17Table(false);
    auto h_t = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.data);
    auto h_0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t0.data);
    double worst = 0.0, zero = 0.0;
    for (int i = 0; i < t.grid.n_rows; i += 97) {
      const double T = t.grid.RowTemperature(i);
      const double want = he.Rate(T);
      const double got = h_t(i, g::C_K2 + g::I2_HE_E);
      if (want > 0.0) worst = std::fmax(worst, std::fabs(got/want - 1.0));
      zero = std::fmax(zero, std::fabs(h_0(i, g::C_K2 + g::I2_HE_E)));
    }
    report.ExpectNear(worst, 0.0, 0.0, 1e-14, "GOW17 He I column = Voronov fit");
    report.Expect(zero == 0.0, "GOW17 He I column is zero without the fit");
  }
  return report.ExitCode();
}
