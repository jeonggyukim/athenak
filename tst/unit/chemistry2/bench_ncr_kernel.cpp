//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bench_ncr_kernel.cpp
//! \brief Time per cell and substeps per cell of the chemistry2 semi-implicit update,
//!        NCR against GOW17 (rate table), on cells with the same n_H, T and H2 and H+
//!        fractions.
//!
//!   usage: bench_ncr_kernel [n_cells] [dt_myr] [repeats] [n_h] [g0]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "athena.hpp"
#include "units/units.hpp"
#include "chemistry2/network/gow17.hpp"
#include "chemistry2/network/ncr.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 15);
  const Real dt = (argc > 2) ? std::atof(argv[2]) : 0.01;
  const int repeats = (argc > 3) ? std::atoi(argv[3]) : 3;
  const Real n_h = (argc > 4) ? std::atof(argv[4]) : 100.0;
  using G = chemistry2::GOW17;
  using N = chemistry2::NCR;
  constexpr int NG = G::NSPEC, NN = N::NSPEC;
  Kokkos::View<Real*[4]> state("state", n);  // T, x_H2, x_H+, x_C+
  {
    auto h = Kokkos::create_mirror_view(state);
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int c = 0; c < n; ++c) {
      h(c, 0) = std::pow(10.0, 1.0 + 3.0*u(rng));
      h(c, 1) = 0.5*u(rng);
      h(c, 2) = (1.0 - 2*h(c, 1))*u(rng)*u(rng);
      h(c, 3) = 1.6e-4*u(rng);
    }
    Kokkos::deep_copy(state, h);
  }
  const Real inf = Kokkos::Experimental::infinity_v<Real>;
  G::Settings sg{1.0, 0.1, 1.6e-4, 3.2e-4, 1.7e-6, 1.0, inf, inf, 1.0, 1.0e9, 3.0e20,
                 true, false};
  sg.table = chemistry2::gow17::BuildGOW17Table(false);
  const N::Settings sn{1.0, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20,
                       true, false, 10.0, 0.0};
  const chemistry2::SemiImplicitSettings si{0.1, 1.0e-12, 100000, true};
  const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  const Real g0 = (argc > 5) ? std::atof(argv[5]) : 1.0;
  const chemistry2::CellInputs in{g0, 2.0e-16, 0.0};
  Kokkos::View<int*> nsub("nsub", n);
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
  auto mean_nsub = [&]() {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nsub);
    double s = 0.0;
    for (int c = 0; c < n; ++c) s += h(c);
    return s/n;
  };
  const double t_g = time_it(KOKKOS_LAMBDA(const int c) {
    Real y[NG] = {};
    y[G::IH2] = state(c, 1);
    y[G::IH_PLUS] = state(c, 2);
    y[G::IC_PLUS] = state(c, 3);
    y[G::ICO] = 1.6e-4 - state(c, 3);
    Real x_e = 0.0;
    for (int s = 0; s < NG; ++s) x_e += G::rows[s].charge*y[s];
    Real e = n_h*chemistry2::Thermo::CvCold(y[G::IH2], 0.1, x_e, GAMMA)*state(c, 0)/
             U_ENERGY;
    const G::Cell cell = G::MakeCell(1.0, GAMMA, sg, uf, in);
    nsub(c) = chemistry2::SemiImplicit<G>::Solve(si, cell, 0.0, dt, y, e);
  });
  const double ns_g = mean_nsub();
  const double t_n = time_it(KOKKOS_LAMBDA(const int c) {
    Real y[NN];
    y[N::IH2] = state(c, 1);
    y[N::IH_PLUS] = state(c, 2);
    y[N::IE] = state(c, 2) + state(c, 3);
    Real e = n_h*chemistry2::Thermo::CvCold(y[N::IH2], 0.1, y[N::IE], GAMMA)*
             state(c, 0)/U_ENERGY;
    const N::Cell cell = N::MakeCell(1.0, GAMMA, sn, uf, in);
    nsub(c) = chemistry2::SemiImplicit<N>::Solve(si, cell, 0.0, dt, y, e);
  });
  const double ns_n = mean_nsub();
  std::printf("cells %d, n_H %g, dt %g Myr\n", n, n_h, dt);
  std::printf("GOW17: %.4e s per cell, %.2f substeps, %.4e s per substep\n", t_g/n, ns_g,
              t_g/n/ns_g);
  std::printf("NCR:   %.4e s per cell, %.2f substeps, %.4e s per substep\n", t_n/n, ns_n,
              t_n/n/ns_n);
  std::printf("NCR/GOW17: per cell %.3f, per substep %.3f\n", t_n/t_g,
              (t_n/ns_n)/(t_g/ns_g));
  return 0;
}
