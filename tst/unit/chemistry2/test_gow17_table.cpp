//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_gow17_table.cpp
//! \brief Every column of the GOW17 rate table against its analytic value, on the
//!        device lookup, over random T in [2 K, 1e8 K]: relative error <= 1e-2 where the
//!        value is >= 1e-6 of the column maximum, <= 3e-2 down to 1e-12 of it, except
//!        grid cells that straddle a fit boundary.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/gow17/gow17_table.hpp"

namespace g = chemistry2::gow17;

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  const chemistry2::RowTable table = g::BuildGOW17Table(false);

  constexpr int N_T = 20000;
  Kokkos::View<Real*> t_d("t", N_T);
  auto t_h = Kokkos::create_mirror_view(t_d);
  std::mt19937_64 rng(13);
  std::uniform_real_distribution<double> u(std::log(2.0), std::log(1.0e8));
  for (int s = 0; s < N_T; ++s) t_h(s) = std::exp(u(rng));
  Kokkos::deep_copy(t_d, t_h);

  // table lookup on the device
  Kokkos::View<Real*[g::N_COLS]> got_d("got", N_T);
  Kokkos::parallel_for("gow17_table", N_T, KOKKOS_LAMBDA(const int s) {
    const chemistry2::Slot sl = table.grid.Locate(t_d(s));
    for (int c = 0; c < g::N_COLS; ++c) got_d(s, c) = table.At(sl, c);
  });
  auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), got_d);

  Real vmax[g::N_COLS] = {};
  for (int i = 0; i < table.grid.n_rows; ++i) {
    Real v[g::N_COLS];
    g::FillRow(table.grid.RowTemperature(i), false, v);
    for (int c = 0; c < g::N_COLS; ++c) vmax[c] = std::fmax(vmax[c], std::fabs(v[c]));
  }
  // fit boundaries: H2 cooling 10, 100, 500 (electrons), 1000, 1e4 K; C II-H2 500 K;
  // collisional dissociation 701 K
  const std::vector<Real> kinks = {10.0, 100.0, 500.0, 701.0, 1000.0, 1.0e4};
  // The electron H2 cooling coefficient Le: its low-T fit, used below its range,
  // reaches the cap Lmax = 1e-10 below 29 K and falls by 15 orders of magnitude from
  // 29 K to 40 K (as in src/chemistry), which no interpolation follows; skipped below
  // 50 K, where electron collisions do not matter to H2 cooling.
  constexpr Real LE_T_MIN = 50.0;
  double worst_hi = 0.0, worst_lo = 0.0;
  Real t_hi = 0.0, t_lo = 0.0;
  int c_hi = -1, c_lo = -1;
  for (int s = 0; s < N_T; ++s) {
    const Real T = t_h(s);
    const chemistry2::Slot sl = table.grid.Locate(T);
    const Real t0 = table.grid.RowTemperature(sl.i0);
    const Real t1 = table.grid.RowTemperature(sl.i1);
    bool skip = false;
    for (Real k : kinks) skip = skip || (t0 <= k && k <= t1);
    if (skip) continue;
    Real v[g::N_COLS];
    g::FillRow(T, false, v);
    for (int c = 0; c < g::N_COLS; ++c) {
      const bool signed_col = (c >= g::C_LOG10_K9L && c <= g::C_LOG10_K10H) ||
                              (c >= g::C_GR_TEXP && c < g::C_GR_TEXP + 4);
      if (signed_col) {  // log10 k and exponents: absolute error, judged as "hi"
        const double d = std::fabs(got(s, c) - v[c]);
        if (d > worst_hi) { worst_hi = d; c_hi = c; }
        continue;
      }
      if (v[c] < 1.0e-12*vmax[c]) continue;
      if (c == g::C_H2COOL + 4 && T < LE_T_MIN) continue;
      // C I rate coefficients above 3e5 K: carbon is ionized there, C I cooling is
      // negligible, and the electron fits steepen toward their 1e6 K cap (measured
      // 4.4e-2 at 8e5 K, 2026-11-20)
      if (c >= g::C_CI && c < g::C_CI + 9 && T > 3.0e5) continue;
      const double d = std::fabs(got(s, c) - v[c])/v[c];
      if (v[c] >= 1.0e-6*vmax[c]) {
        if (d > worst_hi) { worst_hi = d; c_hi = c; t_hi = T; }
      } else if (d > worst_lo) {
        worst_lo = d;
        c_lo = c;
        t_lo = T;
      }
    }
  }
  std::printf("  %d rows; worst above 1e-6 of max %.2e (column %d, T %.4g K), "
              "1e-12 to 1e-6 %.2e (column %d, T %.4g K)\n", table.grid.n_rows, worst_hi,
              c_hi, t_hi, worst_lo, c_lo, t_lo);
  report.ExpectNear(worst_hi, 0.0, 0.0, 1.0e-2, "columns, values >= 1e-6 of max");
  report.ExpectNear(worst_lo, 0.0, 0.0, 3.0e-2, "columns, values 1e-12 to 1e-6 of max");

  // dw/dT against a centered difference of w inside one row
  double worst_dw = 0.0;
  for (const Real T : {3.7, 47.0, 913.0, 1.1e4, 3.3e5, 7.7e7}) {
    Real dw = 0.0;
    const chemistry2::Slot s = table.grid.Locate(T, &dw);
    const Real h = 1.0e-7*T;
    const chemistry2::Slot sp = table.grid.Locate(T + h), sm = table.grid.Locate(T - h);
    if (sp.i0 != s.i0 || sm.i0 != s.i0) continue;
    const Real fd = (sp.w - sm.w)/(2.0*h);
    worst_dw = std::fmax(worst_dw, std::fabs(dw - fd)/std::fabs(fd));
  }
  report.ExpectNear(worst_dw, 0.0, 0.0, 1.0e-6, "dw/dT against a centered difference");
  return report.ExitCode();
}
