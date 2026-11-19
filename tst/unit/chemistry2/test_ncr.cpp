//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ncr.cpp
//! \brief NCR chemical network: the H2/H+ block satisfies the backward Euler equations
//!        of both species and conserves hydrogen; a long semi-implicit solve stays finite
//!        and in bounds with a moderate substep count.

#include <cmath>
#include <cstdio>
#include <random>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/ncr.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {
using Net = chemistry2::NCRAnalytic;
constexpr int NSPEC = Net::NSPEC;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;

Net::Settings Defaults() {
  return Net::Settings{1.0, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20,
                       true, false, 10.0, 0.0};
}

Net::Cell MakeCell(Real n_h, Real chi, Real xi) {
  const chemistry2::UnitFactors u = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  return Net::MakeCell(1.0, GAMMA, Defaults(), u, chemistry2::CellInputs{chi, xi, 0.0});
}

Real Energy(const Real *y, Real n_h, Real T) {
  return n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, y[Net::IE], GAMMA)*T/U_ENERGY;
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  std::mt19937_64 rng(31);
  std::uniform_real_distribution<double> u(0.0, 1.0);

  // The block: residual of backward Euler in both species, hydrogen sum
  double worst_res = 0.0, worst_sum = 0.0;
  int checked = 0;
  for (int n = 0; n < 2000; ++n) {
    const Real n_h = std::pow(10.0, -1.0 + 5.0*u(rng));
    const Real chi = std::pow(10.0, -3.0 + 4.0*u(rng));
    const Real xi = std::pow(10.0, -17.0 + 2.0*u(rng));
    const Real T = std::pow(10.0, 1.0 + 3.5*u(rng));
    Real y[NSPEC];
    y[Net::IH2] = 0.49*u(rng);
    y[Net::IH_PLUS] = (1.0 - 2.0*y[Net::IH2])*u(rng)*u(rng);
    y[Net::IE] = y[Net::IH_PLUS] + 1.6e-4*u(rng);
    const Net::Cell c = MakeCell(n_h, chi, xi);
    const Real e = Energy(y, n_h, T);
    const Net::Substep sub = Net::Prepare(y, e, c);
    const Real h = std::pow(10.0, -6.0 + 6.0*u(rng));  // code time (Myr-ish)
    Real yn[NSPEC];
    for (int k = 0; k < NSPEC; ++k) yn[k] = y[k];
    Net::Update<Net::IH2>(y, yn, e, c, sub, h);
    const Real x_hi = 1.0 - 2.0*y[Net::IH2] - y[Net::IH_PLUS];
    worst_sum = std::fmax(worst_sum, std::fabs(x_hi + 2.0*y[Net::IH2] + y[Net::IH_PLUS]
                                               - 1.0));
    // where no clamp acted, x^{n+1} = x^n + dt (C x_HI^{n+1} - D x^{n+1})
    const bool interior = y[Net::IH2] > 1e-15 && y[Net::IH_PLUS] > 1e-15 &&
                          x_hi > 1e-12 && y[Net::IH2] < 0.25 && sub.T < 3.5e4;
    if (interior) {
      const Real dt = h*c.u_time;
      const Real r1 = y[Net::IH2] - yn[Net::IH2] -
                      dt*(sub.c_h2*x_hi - sub.d_h2*y[Net::IH2]);
      const Real r2 = y[Net::IH_PLUS] - yn[Net::IH_PLUS] -
                      dt*(sub.c_hii*x_hi - sub.d_hii*y[Net::IH_PLUS]);
      const Real s1 = std::fabs(yn[Net::IH2]) + dt*(sub.c_h2*x_hi + sub.d_h2*y[Net::IH2]);
      const Real s2 = std::fabs(yn[Net::IH_PLUS]) +
                      dt*(sub.c_hii*x_hi + sub.d_hii*y[Net::IH_PLUS]);
      worst_res = std::fmax(worst_res, std::fmax(std::fabs(r1)/s1, std::fabs(r2)/s2));
      ++checked;
    }
  }
  std::printf("  block: %d interior states, worst residual %.2e, hydrogen sum %.2e\n",
              checked, worst_res, worst_sum);
  report.Expect(checked > 500, "enough interior states");
  report.ExpectNear(worst_res, 0.0, 0.0, 1e-12, "backward Euler residual of the block");
  report.ExpectNear(worst_sum, 0.0, 0.0, 1e-14, "hydrogen conserved");

  // Long solves: atomic warm gas at n_H 1-1e3, chi 1, xi 2e-16, to 10 Myr
  for (const Real n_h : {1.0, 30.0, 1.0e3}) {
    Real y[NSPEC] = {1e-6, 1e-4, 1e-4 + 1.6e-4};
    Real e = Energy(y, n_h, 8.0e3);
    const Net::Cell c = MakeCell(n_h, 1.0, 2.0e-16);
    const int nsub = chemistry2::SemiImplicit<Net>::Solve(
        chemistry2::SemiImplicitSettings{0.1, 1.0e-12, 100000, true}, c, 0.0, 10.0, y, e);
    const Real T = Net::Temperature(y, e, c);
    std::printf("  n_H %g: %d substeps, T %.3g K, x_H2 %.3g, x_H+ %.3g, x_e %.3g\n", n_h,
                nsub, T, y[Net::IH2], y[Net::IH_PLUS], y[Net::IE]);
    const bool ok = std::isfinite(T) && T > 0 && y[Net::IH2] >= 0 && y[Net::IH2] <= 0.5 &&
                    y[Net::IH_PLUS] >= 0 && y[Net::IH_PLUS] <= 1 && nsub < 100000;
    report.Expect(ok, "long solve finite and in bounds, n_H " + std::to_string(n_h));
  }
  return report.ExitCode();
}
