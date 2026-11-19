//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bench_gow17_table_error.cpp
//! \brief Interpolation error of every GOW17 table column, plain and NqtLog-encoded,
//!        against the analytic value, for several rows per octave. Prints, per column,
//!        the worst relative error over random T in [2 K, 1e8 K] (absolute for
//!        columns that change sign), skipping T within one row of a known kink.

#include <cmath>
#include <cstdio>
#include <random>

#include "athena.hpp"
#include "chemistry2/network/gow17_table.hpp"

using chemistry2::NqtExp;
using chemistry2::NqtLog;
namespace g = chemistry2::gow17;

namespace {
// Analytic values of every column at T (decoded)
void Values(const Real T, Real *v) {
  Real row[g::N_COLS];
  g::FillRow(T, false, row);
  for (int c = 0; c < g::N_COLS; ++c) v[c] = row[c];
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  // fit boundaries: H2 cooling 10, 100, 1000, 1e4 K; CII-H2 500 K; collisional 701 K
  const Real kinks[] = {10.0, 100.0, 500.0, 701.0, 1000.0, 1.0e4, 46600.0};
  for (int rpo : {16, 32, 64}) {
    const chemistry2::RowGrid grid = chemistry2::RowGrid::Make(1.0, 1073741824.0,
                                                               30*rpo + 1);
    double worst_plain[g::N_COLS] = {}, worst_nqt[g::N_COLS] = {},
           worst_log[g::N_COLS] = {};
    // each column's largest value over the range, for the significance threshold
    Real vmax[g::N_COLS] = {};
    for (int i = 0; i < grid.n_rows; ++i) {
      Real v[g::N_COLS];
      Values(grid.RowTemperature(i), v);
      for (int c = 0; c < g::N_COLS; ++c) vmax[c] = std::fmax(vmax[c], std::fabs(v[c]));
    }
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(std::log(2.0), std::log(1.0e8));
    for (int s = 0; s < 20000; ++s) {
      const Real T = std::exp(u(rng));
      const chemistry2::Slot sl = grid.Locate(T);
      const Real t0 = grid.RowTemperature(sl.i0), t1 = grid.RowTemperature(sl.i1);
      bool skip = false;
      for (Real k : kinks) skip = skip || (t0 <= k && k <= t1);
      if (skip) continue;
      Real v[g::N_COLS], v0[g::N_COLS], v1[g::N_COLS];
      Values(T, v);
      Values(t0, v0);
      Values(t1, v1);
      for (int c = 0; c < g::N_COLS; ++c) {
        const Real plain = v0[c] + sl.w*(v1[c] - v0[c]);
        const bool pos = v[c] > 0 && v0[c] > 0 && v1[c] > 0;
        const Real q0 = pos ? NqtLog(v0[c]) : 0.0, q1 = pos ? NqtLog(v1[c]) : 0.0;
        const Real nqt = pos ? NqtExp(q0 + sl.w*(q1 - q0)) : plain;
        const Real lg = pos ? std::exp(std::log(v0[c]) +
                                       sl.w*(std::log(v1[c]) - std::log(v0[c])))
                            : plain;
        const Real scale = pos ? std::fabs(v[c]) : 1.0;  // absolute where sign changes
        if (v[c] == 0.0 && v0[c] == 0.0 && v1[c] == 0.0) continue;
        if (pos && v[c] < 1.0e-12*vmax[c]) continue;  // too small to matter
        worst_plain[c] = std::fmax(worst_plain[c], std::fabs(plain - v[c])/scale);
        worst_nqt[c] = std::fmax(worst_nqt[c], std::fabs(nqt - v[c])/scale);
        worst_log[c] = std::fmax(worst_log[c], std::fabs(lg - v[c])/scale);
      }
    }
    double wp = 0, wn = 0, wc = 0;
    int c_wp = 0, c_wn = 0, c_wc = 0;
    for (int c = 0; c < g::N_COLS; ++c) {
      if (worst_plain[c] > wp) { wp = worst_plain[c]; c_wp = c; }
      if (worst_nqt[c] > wn) { wn = worst_nqt[c]; c_wn = c; }
      const double chosen = worst_plain[c];
      if (chosen > wc) { wc = chosen; c_wc = c; }
    }
    std::printf("rows/octave %2d (%d rows): worst plain %.2e (col %d), "
                "nqt %.2e (col %d), "
                "chosen encoding %.2e (col %d)\n", rpo, grid.n_rows, wp, c_wp, wn, c_wn,
                wc, c_wc);
    // best of plain and log per column
    double wbest = 0;
    int n_log = 0;
    for (int c = 0; c < g::N_COLS; ++c) {
      const bool use_log = worst_log[c] < worst_plain[c];
      n_log += use_log;
      wbest = std::fmax(wbest, use_log ? worst_log[c] : worst_plain[c]);
    }
    std::printf("  best of plain/log per column: worst %.2e, %d log columns\n", wbest,
                n_log);
    for (int c = 0; c < g::N_COLS; ++c) {
      std::printf("  rpo %d col %2d plain %.1e nqt %.1e log %.1e\n", rpo, c,
                  worst_plain[c], worst_nqt[c], worst_log[c]);
    }
  }
  return 0;
}
