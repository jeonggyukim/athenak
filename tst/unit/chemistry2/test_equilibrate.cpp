//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_equilibrate.cpp
//! \brief SemiImplicit::SolveToEquilibrium on one NCR cell: the full solve against a
//!        long Solve, the fixed-temperature solve, and the not-converged return.

#include <cmath>
#include <cstdio>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/ncr/ncr.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {
using Net = chemistry2::NCRAnalytic;
using Solver = chemistry2::SemiImplicit<Net>;
constexpr int NSPEC = Net::NSPEC;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;

Net::Cell MakeCell(Real n_h, Real chi, Real xi) {
  const Net::Settings s{1.0, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20,
                        true, false, 10.0, 0.0};
  const chemistry2::UnitFactors u = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  return Net::MakeCell(1.0, GAMMA, s, u, chemistry2::CellInputs{chi, xi, 0.0});
}

// atomic gas, 1% ionized, at T
void Start(const Net::Cell &c, Real T, Real *y, Real &e) {
  y[Net::IH2] = 0.0;
  y[Net::IH_PLUS] = 0.01;
  y[Net::IE] = 0.01;
  e = Net::Energy(y, T, c);
}

Real MaxRelDiff(const Real *a, const Real *b) {
  Real d = 0.0;
  for (int k = 0; k < NSPEC; ++k) {
    d = std::fmax(d, std::fabs(a[k] - b[k])/std::fmax(std::fabs(b[k]), 1e-12));
  }
  return d;
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  const chemistry2::SemiImplicitSettings si{0.1, 1.0e-12, 100000, true};

  // full solve at n_H = 1, 10, 100 cm^-3 (code time 1 Myr), substeps up to 1e7 Myr:
  // against Solve over 1e7 Myr
  for (const Real n_h : {1.0, 10.0, 100.0}) {
    const Net::Cell c = MakeCell(n_h, 1.0, 2.0e-16);
    Real y[NSPEC], e, y_ref[NSPEC], e_ref;
    Start(c, 1.0e3, y, e);
    Start(c, 1.0e3, y_ref, e_ref);
    const int n = Solver::SolveToEquilibrium(si, c, {1.0e7, 100000, 1.0e-6, false}, y, e);
    const int n_ref = Solver::Solve(si, c, 0.0, 1.0e7, y_ref, e_ref);
    const Real dy = MaxRelDiff(y, y_ref);
    const Real de = std::fabs(e/e_ref - 1.0);
    std::printf("  n_H %5.0f: %d substeps (Solve %d), T %.4g K, x_H2 %.4e, x_H+ %.4e; "
                "species %.1e, e %.1e from Solve\n", n_h, n, n_ref,
                Net::Temperature(y, e, c), y[Net::IH2], y[Net::IH_PLUS], dy, de);
    report.Expect(n > 0 && n < n_ref, "full solve converges, in fewer substeps");
    report.ExpectNear(dy, 0.0, 0.0, 1.0e-4, "full solve: species as a long Solve");
    report.ExpectNear(de, 0.0, 0.0, 1.0e-4, "full solve: energy as a long Solve");
  }

  // fixed T, no radiation, n_H = 1: T held, the result as with a tighter tolerance
  for (const Real lt : {4.0, 4.2, 4.4}) {
    const Net::Cell c = MakeCell(1.0, 0.0, 1.0e-30);
    const Real T0 = std::pow(10.0, lt);
    Real y[NSPEC], e, y_t[NSPEC], e_t;
    Start(c, T0, y, e);
    Start(c, T0, y_t, e_t);
    const int n = Solver::SolveToEquilibrium(si, c, {1.0e5, 100000, 1.0e-6, true}, y, e);
    Solver::SolveToEquilibrium(si, c, {1.0e5, 100000, 1.0e-10, true}, y_t, e_t);
    const Real dT = std::fabs(Net::Temperature(y, e, c)/T0 - 1.0);
    const Real dy = MaxRelDiff(y, y_t);
    std::printf("  fixed log T %.1f: %d substeps, x_H+ %.4e; T %.1e, species %.1e from "
                "tol 1e-10\n", lt, n, y[Net::IH_PLUS], dT, dy);
    report.Expect(n > 0, "fixed-T solve converges");
    report.ExpectNear(dT, 0.0, 0.0, 1.0e-12, "fixed-T solve keeps T");
    report.ExpectNear(dy, 0.0, 0.0, 1.0e-4, "fixed-T solve: species as tol 1e-10");
  }

  // three substeps cannot converge from 1e3 K atomic gas: negative return
  {
    const Net::Cell c = MakeCell(1.0, 1.0, 2.0e-16);
    Real y[NSPEC], e;
    Start(c, 1.0e3, y, e);
    const int n = Solver::SolveToEquilibrium(si, c, {1.0e5, 3, 1.0e-6, false}, y, e);
    report.Expect(n == -3, "not converged in nsub_max = 3 returns -3");
  }
  return report.ExitCode();
}
