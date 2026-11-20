//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ion_ladder_evolution.cpp
//! \brief Oxygen ionizing from neutral at fixed T = 6.1e4 K and n_H = 1e-5 cm^-3 (the
//!        conditions of Katz 2022 Fig. 2), collisional ionization and recombination
//!        only, n_e = 1.2 n_H (H and He ionized): LadderStep at 20 steps per decade of
//!        time against 2000 steps per decade, and the state at 1e4 Myr against the CIE
//!        balance x_{q+1}/x_q = CI_q/rec_{q+1}. With an output path as argument, writes
//!        the time evolution of the coarse run.

#include <cmath>
#include <cstdio>
#include <string>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/ions/ion_ladder.hpp"
#include "chemistry2/ions/rates/coll_ion_rate.hpp"
#include "chemistry2/ions/rates/recomb_rate.hpp"

namespace {
constexpr int Z = 8, NS = Z + 1;
constexpr double MYR = 3.15576e13;

//! Evolve from neutral to each output time with n_per_decade log-spaced steps per decade
//! from 1e-3 Myr; returns the fractions at the output times
void Evolve(const Real *up, const Real *dn, int n_per_decade, const double *t_out,
            int n_out, double out[][NS]) {
  Real x[NS] = {1.0}, xn[NS];
  const Real src[NS] = {}, loss[NS] = {};
  double t = 0.0;
  int k = 0;
  const double t0 = 1e-3*MYR;
  for (int i = 0; k < n_out; ++i) {
    const double t_reg = t0*std::pow(10.0, static_cast<double>(i)/n_per_decade);
    const double t_next = std::fmin(t_reg, t_out[k]);
    for (int q = 0; q < NS; ++q) xn[q] = x[q];
    chemistry2::ions::LadderStep<NS>(xn, up, dn, src, loss, t_next - t, 1.0, x);
    t = t_next;
    if (t >= t_out[k]) {
      for (int q = 0; q < NS; ++q) out[k][q] = x[q];
      ++k;
      --i;  // the next regular step is still ahead
    }
  }
}
}  // namespace

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
  const Real T = 6.1e4, n_h = 1.0e-5, n_e = 1.2*n_h;
  Real up[NS] = {}, dn[NS] = {};
  for (int q = 0; q < NS - 1; ++q) up[q] = ci.Rate(Z, Z - q, T)*n_e;
  for (int q = 1; q < NS; ++q) dn[q] = rec.Rate(Z, Z - q, T)*n_e;
  constexpr int N_OUT = 7;
  const double t_out[N_OUT] = {1e-2*MYR, 1e-1*MYR, 1*MYR, 10*MYR, 1e2*MYR, 1e3*MYR,
                               1e4*MYR};
  double coarse[N_OUT][NS], fine[N_OUT][NS];
  Evolve(up, dn, 20, t_out, N_OUT, coarse);
  Evolve(up, dn, 2000, t_out, N_OUT, fine);
  double worst = 0.0;
  for (int k = 0; k < N_OUT; ++k) {
    for (int q = 0; q < NS; ++q) {
      worst = std::fmax(worst, std::fabs(coarse[k][q] - fine[k][q]));
    }
  }
  // CIE balance at the end
  double x_eq[NS], s = 1.0;
  x_eq[0] = 1.0;
  for (int q = 0; q < NS - 1; ++q) {
    x_eq[q + 1] = x_eq[q]*up[q]/dn[q + 1];
    s += x_eq[q + 1];
  }
  double worst_eq = 0.0;
  for (int q = 0; q < NS; ++q) {
    worst_eq = std::fmax(worst_eq, std::fabs(fine[N_OUT - 1][q] - x_eq[q]/s));
  }
  std::printf("  20 vs 2000 steps per decade: worst absolute %.2e; at 1e4 Myr vs CIE "
              "%.2e\n", worst, worst_eq);
  if (argc > 1) {
    FILE *f = std::fopen(argv[1], "w");
    std::fprintf(f, "# t [Myr], x(O I) ... x(O IX), 20 steps per decade\n");
    for (int k = 0; k < N_OUT; ++k) {
      std::fprintf(f, "%.6e", t_out[k]/MYR);
      for (int q = 0; q < NS; ++q) std::fprintf(f, " %.6e", coarse[k][q]);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
  }
  // measured 5.3e-3, 2026-11-20
  report.ExpectNear(worst, 0.0, 0.0, 2e-2, "20 steps per decade within 2e-2 absolute");
  report.ExpectNear(worst_eq, 0.0, 0.0, 1e-6, "equilibrium at 1e4 Myr is CIE");
  return report.ExitCode();
}
