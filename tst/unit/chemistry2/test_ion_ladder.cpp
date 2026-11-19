//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ion_ladder.cpp
//! \brief LadderStep (after Tigris tst/unit/photchem/test_ion_ladder.cpp), 2-10 stages,
//!        rates 1e-20-1e-8 s^-1, steps 1e6 s to 1 Gyr: against dense Gaussian
//!        elimination (1e-8 relative); a stage balance x_{q+1}/x_q = up[q]/dn[q+1] kept
//!        by one step (1e-8: round-off of 1 + h rate, h rate up to 3e8); with molecular
//!        source and loss terms, the element conserved (1e-9) and no stage negative.
//!        ChainRates against the formulas.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/ions/ion_ladder.hpp"

namespace {
constexpr int NMAX = 10;

void DenseSolve(int n, double a[NMAX][NMAX], double *b, double *x) {
  for (int c = 0; c < n; ++c) {
    int p = c;
    for (int r = c + 1; r < n; ++r) {
      if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
    }
    for (int k = 0; k < n; ++k) std::swap(a[c][k], a[p][k]);
    std::swap(b[c], b[p]);
    for (int r = c + 1; r < n; ++r) {
      const double f = a[r][c]/a[c][c];
      for (int k = c; k < n; ++k) a[r][k] -= f*a[c][k];
      b[r] -= f*b[c];
    }
  }
  for (int r = n - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < n; ++k) s -= a[r][k]*x[k];
    x[r] = s/a[r][r];
  }
}

std::mt19937_64 rng(7);
double LogU(double lo, double hi) {
  std::uniform_real_distribution<double> u(std::log10(lo), std::log10(hi));
  return std::pow(10.0, u(rng));
}

template <int N>
void CheckN(unit::Report *report) {
  double worst_dense = 0.0, worst_bal = 0.0, worst_cons = 0.0;
  bool negative = false;
  for (int trial = 0; trial < 200; ++trial) {
    Real up[N], dn[N], src[N] = {}, loss[N] = {}, xn[N], x[N];
    double total = 0.0;
    // ionization and recombination rates [s^-1] of ISM gas, and steps of 1e6 s to 1 Gyr,
    // as in the Tigris test
    for (int q = 0; q < N; ++q) {
      up[q] = (q < N - 1) ? LogU(1e-20, 1e-8) : 0.0;
      dn[q] = (q > 0) ? LogU(1e-15, 1e-9) : 0.0;
      xn[q] = LogU(1e-10, 1.0);
      total += xn[q];
    }
    constexpr Real H_SET[4] = {1e6, 3.15e10, 3.15e13, 3.15e16};
    const Real h = H_SET[trial % 4];
    chemistry2::ions::LadderStep<N>(xn, up, dn, src, loss, h, total, x);
    double a[NMAX][NMAX] = {}, b[NMAX], xd[NMAX];
    for (int q = 0; q < N; ++q) {
      a[q][q] = 1.0 + h*(up[q] + dn[q]);
      if (q > 0) a[q][q - 1] = -h*up[q - 1];
      if (q < N - 1) a[q][q + 1] = -h*dn[q + 1];
      b[q] = xn[q];
    }
    DenseSolve(N, a, b, xd);
    for (int q = 0; q < N; ++q) {
      const double d = std::fabs(x[q] - xd[q])/std::fmax(std::fabs(xd[q]), 1e-24*total);
      worst_dense = std::fmax(worst_dense, d);
    }
    // fixed point: the balance x_{q+1}/x_q = up[q]/dn[q+1] is kept by a step of any h
    Real xb[N], xi[N];
    xb[0] = 1.0;
    double sb = 1.0;
    for (int q = 0; q < N - 1; ++q) {
      xb[q + 1] = xb[q]*up[q]/dn[q + 1];
      sb += xb[q + 1];
    }
    for (int q = 0; q < N; ++q) xb[q] /= sb;
    chemistry2::ions::LadderStep<N>(xb, up, dn, src, loss, h, 1.0, xi);
    for (int q = 0; q < N - 1; ++q) {
      if (xb[q] > 1e-200 && xb[q + 1] > 1e-200) {
        const double r = (xi[q + 1]/xi[q])/(up[q]/dn[q + 1]);
        worst_bal = std::fmax(worst_bal, std::fabs(r - 1.0));
      }
    }
    // with molecular terms (h src <= 1e-2 total, h loss <= 1):
    // sum x = total + h sum(src - loss x)
    for (int q = 0; q < N; ++q) {
      src[q] = LogU(1e-6, 1e-2)*total/h;
      loss[q] = LogU(1e-3, 1.0)/h;
    }
    chemistry2::ions::LadderStep<N>(xn, up, dn, src, loss, h, total, x);
    double sum = 0.0, budget = total;
    for (int q = 0; q < N; ++q) {
      sum += x[q];
      budget += h*(src[q] - loss[q]*x[q]);
      negative = negative || x[q] < 0.0;
    }
    int qmax = 0;
    for (int q = 1; q < N; ++q) {
      if (x[q] > x[qmax]) qmax = q;
    }
    if (qmax > 0) {
      worst_cons = std::fmax(worst_cons, std::fabs(sum - budget)/
                             std::fmax(std::fabs(budget), total));
    }
  }
  const std::string n = std::to_string(N) + " stages";
  report->ExpectNear(worst_dense, 0.0, 0.0, 1e-8, n + ": against dense elimination");
  // measured up to 2.7e-9 (balance) and 2.1e-10 (conservation), 2026-11-20
  report->ExpectNear(worst_bal, 0.0, 0.0, 1e-8, n + ": balance kept by one step");
  report->ExpectNear(worst_cons, 0.0, 0.0, 1e-9, n + ": element conservation");
  report->Expect(!negative, n + ": no negative stage");
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  CheckN<2>(&report);
  CheckN<3>(&report);
  CheckN<4>(&report);
  CheckN<6>(&report);
  CheckN<10>(&report);

  // ChainRates: 3 tracked stages and the higher ions
  const chemistry2::ions::IonCoefficients k[3] = {
      {1e-10, 2e-11, 0.0, 0.0}, {3e-12, 4e-13, 5e-12, 6e-13},
      {7e-14, 8e-15, 9e-12, 1e-12}};
  const Real xi_ph[3] = {1e-9, 2e-10, 3e-11};
  const chemistry2::ions::ChainCell cell{100.0, 1.0, 1e-2, 0.9, 3.15e13};
  Real up[4], dn[4];
  chemistry2::ions::ChainRates<3, true>(k, xi_ph, 5e-16, true, true, 7e-12, cell, up, dn);
  const double want_up0 = cell.u*(1e-10*1.0 + 2e-11*100.0*1e-2 + 1e-9 + 5e-16);
  const double want_up2 = cell.u*(7e-14*1.0 + 8e-15*100.0*1e-2 + 3e-11);
  const double want_dn1 = cell.u*(5e-12*1.0 + 6e-13*100.0*0.9);
  report.ExpectNear(up[0], want_up0, 1e-14, 0.0, "ChainRates up[0]");
  report.ExpectNear(up[2], want_up2, 1e-14, 0.0, "ChainRates up[top] into higher ions");
  report.ExpectNear(dn[1], want_dn1, 1e-14, 0.0, "ChainRates dn[1]");
  report.ExpectNear(dn[3], cell.u*7e-12, 1e-14, 0.0, "ChainRates higher-ion return");
  report.Expect(up[3] == 0.0 && dn[0] == 0.0, "ChainRates end rates zero");
  return report.ExitCode();
}
