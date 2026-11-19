//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_h2_semi_implicit.cpp
//! \brief One-cell solves of the chemistry2 H2 network with the chemistry2 semi-implicit
//!        solver, against src/chemistry (H2Network with ode_solvers::SemiImplicit), and
//!        on device against host.
//!
//!   On the host both evaluate the same arithmetic in the same order and agree bit for
//!   bit; the check allows 1e-14 relative. The device result may differ by rounding.

#include <cmath>
#include <cstdio>

#include "athena.hpp"
#include "check.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"
#include "chemistry/network/H2.hpp"
#include "ode_solvers/semi_implicit.hpp"
#include "chemistry2/network/h2.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {

constexpr Real MU_H = 1.4;
constexpr Real GAMMA = 5.0/3.0;
constexpr Real U_TIME = 3.15576e13;   // 1 Myr [s]
constexpr Real U_ENERGY = 1.0e-12;    // [erg cm^-3]

struct Case {
  Real n_h, x_h2, t, dt;
  bool const_cv;
};

struct Result {
  Real y[2];
  Real e;
};

// Density unit such that w0(IDN) = 1 holds n_H
Real DensityCgs(Real n_h) { return n_h*MU_H*units::Units::hydrogen_mass_cgs; }

Real InitialEnergy(const Case &c) {
  const Real x0 = Kokkos::fmin(Kokkos::fmax(c.const_cv ? 0.0 : c.x_h2, 0.0), 0.5);
  const Real cv = units::Units::k_boltzmann_cgs*((1. - 2.*x0) + x0 + 0.1)/(GAMMA - 1.0);
  return c.n_h*cv*c.t/U_ENERGY;
}

Result SolveOld(const Case &c) {
  DvceArray5D<Real> w0("w0", 1, 1, 1, 1, 1);
  Kokkos::deep_copy(w0, 1.0);
  DualArray1D<RegionSize> sizes("sizes", 1);
  DvceArray1D<Real> ir("ir", 1);
  chemistry::H2Network net(chemistry::H2Settings{c.const_cv, false}, 0, 0, 0, 0, w0,
                           sizes, ir, DensityCgs(c.n_h), MU_H, GAMMA,
                           units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY);
  net.y(chemistry::H2Network::IH2) = c.x_h2;
  net.y(chemistry::H2Network::IH) = 1.0 - 2.0*c.x_h2;
  net.y(chemistry::H2Network::IIE) = InitialEnergy(c);
  ode_solvers::SemiImplicit<chemistry::H2Network> solver(
      ode_solvers::SemiImplicitSettings{0.1, 1.0e-12, 100000, true}, net, 0.0, c.dt);
  solver.SolveODE();
  return Result{{net.y(chemistry::H2Network::IH2), net.y(chemistry::H2Network::IH)},
                net.y(chemistry::H2Network::IIE)};
}

KOKKOS_INLINE_FUNCTION chemistry2::H2::Cell MakeCell(Real n_h, bool const_cv) {
  return chemistry2::H2::Cell{n_h, GAMMA, U_TIME, U_ENERGY, const_cv};
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  using Solver = chemistry2::SemiImplicit<chemistry2::H2>;
  const chemistry2::SemiImplicitSettings s{0.1, 1.0e-12, 100000, true};

  constexpr int n_cases = 162;  // 3 n_H x 3 x_H2 x 3 T x 3 dt x 2 const_cv
  Case cases[n_cases];  // NOLINT(runtime/arrays)
  int n = 0;
  for (Real n_h : {1.0, 1.0e2, 1.0e4}) {
    for (Real x_h2 : {0.0, 0.1, 0.45}) {
      for (Real t : {10.0, 1.0e2, 1.0e4}) {
        for (Real dt : {1.0e-3, 1.0, 1.0e2}) {
          for (bool const_cv : {false, true}) {
            cases[n++] = Case{n_h, x_h2, t, dt, const_cv};
          }
        }
      }
    }
  }

  // Host: chemistry2 against src/chemistry
  Kokkos::View<Real*[3], Kokkos::HostSpace> host("host", n_cases);
  double worst = 0.0;
  int n_differ = 0;
  for (int i = 0; i < n_cases; ++i) {
    const Case &c = cases[i];
    // n_H as the old network computes it from w0(IDN) = 1
    const Real n_h = 1.0*DensityCgs(c.n_h)/(MU_H*units::Units::hydrogen_mass_cgs);
    Real y[2] = {c.x_h2, 1.0 - 2.0*c.x_h2};
    Real e = InitialEnergy(c);
    Solver::Solve(s, MakeCell(n_h, c.const_cv), 0.0, c.dt, y, e);
    host(i, 0) = y[0];
    host(i, 1) = y[1];
    host(i, 2) = e;
    const Result old = SolveOld(c);
    const Real got[3] = {y[0], y[1], e};
    const Real want[3] = {old.y[0], old.y[1], old.e};
    for (int q = 0; q < 3; ++q) {
      const double d = std::fabs(got[q] - want[q]);
      if (d > 0.0) ++n_differ;
      if (d > 1.0e-14*std::fabs(want[q]) + 1.0e-300) {
        report.Expect(false, "chemistry2 equals src/chemistry to 1e-14 relative");
        std::printf("    case %d, quantity %d: got %.17e want %.17e\n", i, q, got[q],
                    want[q]);
      }
      if (want[q] != 0.0) worst = std::fmax(worst, d/std::fabs(want[q]));
    }
  }
  std::printf("  %d cases; %d of %d values differ from src/chemistry; worst relative "
              "difference %.2e\n", n_cases, n_differ, 3*n_cases, worst);

  // Device: the same cases, against the host
  Kokkos::View<Case*> cases_d("cases", n_cases);
  auto cases_h = Kokkos::create_mirror_view(cases_d);
  for (int i = 0; i < n_cases; ++i) cases_h(i) = cases[i];
  Kokkos::deep_copy(cases_d, cases_h);
  Kokkos::View<Real*[3]> out_d("out", n_cases);
  Kokkos::View<Real*> e0_d("e0", n_cases);
  auto e0_h = Kokkos::create_mirror_view(e0_d);
  Kokkos::View<Real*> nh_d("nh", n_cases);
  auto nh_h = Kokkos::create_mirror_view(nh_d);
  for (int i = 0; i < n_cases; ++i) {
    e0_h(i) = InitialEnergy(cases[i]);
    nh_h(i) = 1.0*DensityCgs(cases[i].n_h)/(MU_H*units::Units::hydrogen_mass_cgs);
  }
  Kokkos::deep_copy(e0_d, e0_h);
  Kokkos::deep_copy(nh_d, nh_h);
  Kokkos::parallel_for("h2_semi_implicit", n_cases, KOKKOS_LAMBDA(const int i) {
    const Case c = cases_d(i);
    Real y[2] = {c.x_h2, 1.0 - 2.0*c.x_h2};
    Real e = e0_d(i);
    Solver::Solve(s, MakeCell(nh_d(i), c.const_cv), 0.0, c.dt, y, e);
    out_d(i, 0) = y[0];
    out_d(i, 1) = y[1];
    out_d(i, 2) = e;
  });
  auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out_d);
  // Unit tests are compiled without fused multiply-adds (tst/unit/CMakeLists.txt), so
  // the kernel repeats the host arithmetic. With them, rounding changes the step sizes
  // and long solves (dt = 100 Myr at n_H = 1e4) differ by 1e-6 relative.
  double worst_dev = 0.0;
  for (int i = 0; i < n_cases; ++i) {
    for (int q = 0; q < 3; ++q) {
      if (host(i, q) != 0.0) {
        const double d = std::fabs(out_h(i, q) - host(i, q))/std::fabs(host(i, q));
        if (d > 1.0e-14) {
          std::printf("    case %d (n_H %g, x_H2 %g, T %g, dt %g, const_cv %d), "
                      "quantity %d: device %.17e host %.17e\n", i, cases[i].n_h,
                      cases[i].x_h2, cases[i].t, cases[i].dt, cases[i].const_cv, q,
                      out_h(i, q), host(i, q));
        }
        worst_dev = std::fmax(worst_dev, d);
      }
    }
  }
  report.ExpectNear(worst_dev, 0.0, 0.0, 1.0e-14,
                    "worst relative difference, device against host");

  return report.ExitCode();
}
