//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ion_rates.cpp
//! \brief The ion rate data (inputs/chemistry2/ions: Badnell 2023 radiative and
//!        dielectronic recombination, Voronov 1997 collisional ionization) read by the
//!        readers ported from Tigris, and the collisional-ionization equilibrium of
//!        oxygen they give: x_{q+1}/x_q = CI_q/rec_{q+1}, all nine stages. Checks: the
//!        fractions sum to 1 and O VI peaks near 3e5 K at 0.1-0.3 (Gnat & Sternberg
//!        2007, Bryans et al. 2009). With an output path as argument, writes the
//!        fractions for the comparison with Cloudy.

#include <cmath>
#include <cstdio>
#include <string>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/ions/rates/coll_ion_rate.hpp"
#include "chemistry2/ions/rates/recomb_rate.hpp"
#include "chemistry2/ions/ion_table.hpp"

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  std::string dir(__FILE__);
  dir = dir.substr(0, dir.find_last_of('/')) + "/../../../inputs/chemistry2/ions/";
  Rates::RecombRate rec;
  rec.Load(dir + "badnell_rr_2023.dat", dir + "badnell_dr_C_2023.dat",
           dir + "badnell_dr_E_2023.dat");
  Rates::CollIonRate ci;
  ci.Load(dir + "voronov97_coll_ion.dat");
  constexpr int Z = 8, NS = Z + 1;  // O I to O IX
  FILE *out = (argc > 1) ? std::fopen(argv[1], "w") : nullptr;
  if (out) std::fprintf(out, "# T [K], x(O I) ... x(O IX)\n");
  double worst_sum = 0.0, o6_peak = 0.0, t_peak = 0.0;
  for (int i = 0; i <= 80; ++i) {
    const Real T = std::pow(10.0, 4.0 + 4.0*i/80.0);
    double x[NS];
    x[0] = 1.0;
    double sum = 1.0;
    for (int q = 0; q < NS - 1; ++q) {
      // stage q has Z - q electrons; CI of that ion, recombination of the next
      const double r = ci.Rate(Z, Z - q, T)/rec.Rate(Z, Z - q - 1, T);
      x[q + 1] = x[q]*r;
      sum += x[q + 1];
    }
    double s = 0.0;
    for (double &v : x) {
      v /= sum;
      s += v;
    }
    worst_sum = std::fmax(worst_sum, std::fabs(s - 1.0));
    if (x[5] > o6_peak) {
      o6_peak = x[5];
      t_peak = T;
    }
    if (out) {
      std::fprintf(out, "%.6e", T);
      for (double v : x) std::fprintf(out, " %.6e", v);
      std::fprintf(out, "\n");
    }
  }
  if (out) std::fclose(out);
  std::printf("  O VI peak %.3f at T %.3g K\n", o6_peak, t_peak);
  report.ExpectNear(worst_sum, 0.0, 0.0, 1e-12, "fractions sum to 1");
  report.Expect(o6_peak > 0.1 && o6_peak < 0.3 && t_peak > 2e5 && t_peak < 4e5,
                "O VI CIE peak 0.1-0.3 at 2e5-4e5 K");
  // the ion RowTable (64 rows per octave) against the readers, O I-O IV links, where a
  // rate is above 1e-30 cm^3 s^-1, T 1e2-1e7 K
  {
    const chemistry2::ions::IonRateData data(dir);
    const chemistry2::RowGrid grid = chemistry2::RowGrid::Make(1.0, 1073741824.0,
                                                               30*64 + 1);
    const chemistry2::RowTable tab = chemistry2::ions::BuildIonTable(data, grid, Z, 3);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tab.data);
    double worst_tab = 0.0;
    for (int i = 0; i < 400; ++i) {
      const Real T = std::pow(10.0, 2.0 + 5.0*(i + 0.37)/400.0);
      const chemistry2::Slot s = grid.Locate(T);
      for (int q = 0; q < 3; ++q) {
        const Real want[4] = {data.ci.Rate(Z, Z - q, T), data.ct.CtIon(Z, Z - q, T),
                              data.rec.Rate(Z, Z - q - 1, T),
                              data.ct.CtRec(Z, Z - q - 1, T)};
        for (int k = 0; k < 4; ++k) {
          const int c = q*chemistry2::ions::LINK_COLS + k;
          const double got = h(s.i0, c) + s.w*(h(s.i1, c) - h(s.i0, c));
          // above 1e-20 cm^3 s^-1: deeper in its Boltzmann tail (below ~1e4 K, 1e-30 to
          // 1e-24) collisional ionization is up to 7% off at 64 rows per octave, and
          // negligible against recombination (~1e-12)
          const double e = (want[k] > 1e-20) ? std::fabs(got/want[k] - 1.0) : 0.0;
          worst_tab = std::fmax(worst_tab, e);
        }
      }
    }
    std::printf("  ion table vs readers: worst relative %.2e\n", worst_tab);
    report.ExpectNear(worst_tab, 0.0, 0.0, 3e-2, "ion table vs readers");
  }
  return report.ExitCode();
}
