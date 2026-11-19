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
  return report.ExitCode();
}
