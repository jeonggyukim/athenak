//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bench_h2_kernel.cpp
//! \brief Time per cell of the H2 semi-implicit update, src/chemistry against
//!        chemistry2, on the same random cells in one kernel each.
//!
//!   usage: bench_h2_kernel [n_cells] [dt_myr] [repeats]
//!   Prints the best of `repeats` timings for each module and the substep count, so the
//!   cost per cell substep follows.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "athena.hpp"
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
constexpr Real U_TIME = 3.15576e13;
constexpr Real U_ENERGY = 1.0e-12;
constexpr Real U_DENSITY = MU_H*units::Units::hydrogen_mass_cgs;  // rho = 1 is n_H = 1
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
  const Real dt = (argc > 2) ? std::atof(argv[2]) : 0.01;
  const int repeats = (argc > 3) ? std::atoi(argv[3]) : 5;

  // Random cells: n_H 10-1e4 cm^-3 (log-uniform), x_H2 0-0.5, T 10-1e4 K (log-uniform)
  DvceArray5D<Real> w0("w0", 1, 1, 1, 1, n);  // w0(0, IDN, 0, 0, i): density, n_H units
  Kokkos::View<Real*[3]> init("init", n);     // x_H2, x_H, e
  {
    auto w0_h = Kokkos::create_mirror_view(w0);
    auto init_h = Kokkos::create_mirror_view(init);
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
      const Real n_h = std::pow(10.0, 1.0 + 3.0*u(rng));
      const Real x_h2 = 0.5*u(rng);
      const Real t = std::pow(10.0, 1.0 + 3.0*u(rng));
      const Real cv = units::Units::k_boltzmann_cgs*((1. - 2.*x_h2) + x_h2 + 0.1)/
                      (GAMMA - 1.0);
      w0_h(0, IDN, 0, 0, i) = n_h;
      init_h(i, 0) = x_h2;
      init_h(i, 1) = 1.0 - 2.0*x_h2;
      init_h(i, 2) = n_h*cv*t/U_ENERGY;
    }
    Kokkos::deep_copy(w0, w0_h);
    Kokkos::deep_copy(init, init_h);
  }
  Kokkos::View<Real*[3]> out("out", n);
  Kokkos::View<int*> nsub("nsub", n);
  DualArray1D<RegionSize> sizes("sizes", 1);
  DvceArray1D<Real> ir("ir", 1);

  const ode_solvers::SemiImplicitSettings s_old{0.1, 1.0e-12, 100000, true};
  const chemistry2::SemiImplicitSettings s_new{0.1, 1.0e-12, 100000, true};
  const chemistry::H2Settings h2_old{false, false};
  const chemistry2::H2::Settings h2_new{false};
  const chemistry2::UnitFactors units =
      chemistry2::MakeUnitFactors(U_DENSITY, U_TIME, U_ENERGY, MU_H);

  auto time_it = [&](auto kernel) {
    double best = 1e30;
    for (int r = 0; r < repeats; ++r) {
      Kokkos::fence();
      Kokkos::Timer timer;
      Kokkos::parallel_for("bench", n, kernel);
      Kokkos::fence();
      best = std::min(best, timer.seconds());
    }
    return best;
  };

  const double t_old = time_it(KOKKOS_LAMBDA(const int i) {
    chemistry::H2Network net(h2_old, 0, 0, 0, i, w0, sizes, ir, U_DENSITY, MU_H, GAMMA,
                             units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY);
    net.y(0) = init(i, 0);
    net.y(1) = init(i, 1);
    net.y(2) = init(i, 2);
    ode_solvers::SemiImplicit<chemistry::H2Network> solver(s_old, net, 0.0, dt);
    solver.SolveODE();
    out(i, 0) = net.y(0);
    out(i, 1) = net.y(1);
    out(i, 2) = net.y(2);
  });
  auto out_old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);

  const double t_new = time_it(KOKKOS_LAMBDA(const int i) {
    Real y[2] = {init(i, 0), init(i, 1)};
    Real e = init(i, 2);
    const chemistry2::H2::Cell c = chemistry2::H2::MakeCell(
        w0(0, IDN, 0, 0, i), GAMMA, h2_new, units, chemistry2::CellInputs{});
    nsub(i) = chemistry2::SemiImplicit<chemistry2::H2>::Solve(s_new, c, 0.0, dt, y, e);
    out(i, 0) = y[0];
    out(i, 1) = y[1];
    out(i, 2) = e;
  });
  auto out_new = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  auto nsub_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nsub);

  double total_sub = 0.0, worst = 0.0;
  for (int i = 0; i < n; ++i) {
    total_sub += nsub_h(i);
    for (int q = 0; q < 3; ++q) {
      if (out_old(i, q) != 0.0) {
        worst = std::max(worst, std::fabs(out_new(i, q) - out_old(i, q))/
                                std::fabs(out_old(i, q)));
      }
    }
  }
  std::printf("cells %d, dt %g Myr, mean substeps %.3f, worst relative difference %.2e\n",
              n, dt, total_sub/n, worst);
  std::printf("src/chemistry: %.4e s, %.4e s per cell substep\n", t_old,
              t_old/total_sub);
  std::printf("chemistry2:    %.4e s, %.4e s per cell substep, ratio new/old %.3f\n",
              t_new, t_new/total_sub, t_new/t_old);
  return 0;
}
