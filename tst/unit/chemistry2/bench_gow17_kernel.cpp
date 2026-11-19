//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bench_gow17_kernel.cpp
//! \brief Time per cell of the GOW17 semi-implicit update, src/chemistry against
//!        chemistry2, on the same random cells, one kernel each.
//!
//!   usage: bench_gow17_kernel [n_cells] [dt_myr] [repeats] [nsub_max]
//!   nsub_max = 1 fixes one substep per cell, so both codes do the same work.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <Kokkos_ArithTraits.hpp>  // src/chemistry's headers need it first
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"
#include "chemistry/chemistry_utils.hpp"
#include "chemistry/network/gow17.hpp"
#include "ode_solvers/semi_implicit.hpp"
#include "chemistry2/network/gow17.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {
using New = chemistry2::GOW17;
using Old = chemistry::GOW17Network;
constexpr int NSPEC = New::NSPEC;
constexpr Real MU_H = 1.4;
constexpr Real GAMMA = 5.0/3.0;
constexpr Real U_TIME = 3.15576e13;
constexpr Real U_ENERGY = 1.0e-12;
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 15);
  const Real dt = (argc > 2) ? std::atof(argv[2]) : 0.01;
  const int repeats = (argc > 3) ? std::atoi(argv[3]) : 3;
  const int nsub_max = (argc > 4) ? std::atoi(argv[4]) : 100000;
  const Real n_h = 100.0, g0 = 1.0, xi_cr = 2.0e-16;
  const Real density_cgs = n_h*MU_H*units::Units::hydrogen_mass_cgs;

  // Random cells at n_H = 100: T 10-1e4 K, species split at random
  DvceArray5D<Real> w0("w0", n, 5, 3, 3, 3);  // old network reads neighbours
  Kokkos::deep_copy(w0, 0.0);
  Kokkos::View<Real*[NSPEC + 1]> init("init", n);
  {
    auto w0_h = Kokkos::create_mirror_view(w0);
    auto init_h = Kokkos::create_mirror_view(init);
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int c = 0; c < n; ++c) {
      for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 3; ++i) w0_h(c, IDN, k, j, i) = 1.0;
      Real y[NSPEC] = {};
      y[New::IHE_PLUS] = 0.1*u(rng)*u(rng);
      y[New::IC_PLUS] = 1.6e-4*u(rng);
      y[New::ICO] = (1.6e-4 - y[New::IC_PLUS])*u(rng);
      y[New::IH2] = 0.5*u(rng);
      y[New::IH_PLUS] = (1.0 - 2*y[New::IH2])*u(rng)*u(rng);
      y[New::IO_PLUS] = 1e-6*u(rng);
      y[New::ISI_PLUS] = 1.7e-6*u(rng);
      Real x_e = 0.0;
      for (int s = 0; s < NSPEC; ++s) x_e += New::rows[s].charge*y[s];
      const Real T = std::pow(10.0, 1.0 + 3.0*u(rng));
      const Real cv = chemistry2::Thermo::CvCold(y[New::IH2], 0.1, x_e, GAMMA);
      for (int s = 0; s < NSPEC; ++s) init_h(c, s) = y[s];
      init_h(c, NSPEC) = n_h*cv*T/U_ENERGY;
    }
    Kokkos::deep_copy(w0, w0_h);
    Kokkos::deep_copy(init, init_h);
  }
  DualArray1D<RegionSize> sizes("sizes", n);
  for (int c = 0; c < n; ++c) {
    sizes.h_view(c).dx1 = sizes.h_view(c).dx2 = sizes.h_view(c).dx3 = 1.0;
  }
  sizes.template modify<HostMemSpace>();
  sizes.template sync<DevExeSpace>();
  DvceArray1D<Real> ir("ir", Old::n_freq);
  {
    auto ir_h = Kokkos::create_mirror_view(ir);
    for (int f = 0; f < Old::n_freq; ++f) ir_h(f) = g0;
    ir_h(Old::n_freq - 1) = xi_cr;
    Kokkos::deep_copy(ir, ir_h);
  }
  Kokkos::View<Real*[NSPEC + 1]> out("out", n);
  Kokkos::View<int*> nsub("nsub", n);

  chemistry::GOW17Settings so{};
  so.zd = 1.0; so.xHe = 0.1; so.xC = 1.6e-4; so.xO = 3.2e-4; so.xSi = 1.7e-6;
  so.temperature_min_rates = 1.0;
  so.temperature_max_rates = Kokkos::Experimental::infinity_v<Real>;
  so.temperature_max_heating = Kokkos::Experimental::infinity_v<Real>;
  so.temperature_min_cooling = 1.0;
  so.temperature_max_cooling_nm = 1.0e9;
  so.Leff_CO_max = 3.0e20;
  so.H2_rovib_cooling = true;
  so.velocity_cgs = 1.0; so.length_cgs = 1.0; so.three_d = true; so.multi_d = true;
  const Real inf = Kokkos::Experimental::infinity_v<Real>;
  const New::Settings sn{1.0, 0.1, 1.6e-4, 3.2e-4, 1.7e-6, 1.0, inf, inf, 1.0, 1.0e9,
                         3.0e20, true, false};
  const ode_solvers::SemiImplicitSettings s_old{0.1, 1.0e-12, nsub_max, true};
  const chemistry2::SemiImplicitSettings s_new{0.1, 1.0e-12, nsub_max, true};
  const chemistry2::UnitFactors units =
      chemistry2::MakeUnitFactors(density_cgs, U_TIME, U_ENERGY, MU_H);

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

  const double t_old = time_it(KOKKOS_LAMBDA(const int c) {
    Old net(so, c, 1, 1, 1, w0, sizes, ir, density_cgs, MU_H, GAMMA,
            units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY);
    for (int s = 0; s <= NSPEC; ++s) net.y(s) = init(c, s);
    ode_solvers::SemiImplicit<Old> solver(s_old, net, 0.0, dt);
    solver.SolveODE();
    for (int s = 0; s <= NSPEC; ++s) out(c, s) = net.y(s);
  });
  const double t_new = time_it(KOKKOS_LAMBDA(const int c) {
    Real y[NSPEC];
    for (int s = 0; s < NSPEC; ++s) y[s] = init(c, s);
    Real e = init(c, NSPEC);
    const New::Cell cell = New::MakeCell(1.0, GAMMA, sn, units,
                                         chemistry2::CellInputs{g0, xi_cr, 0.0});
    nsub(c) = chemistry2::SemiImplicit<New>::Solve(s_new, cell, 0.0, dt, y, e);
    for (int s = 0; s < NSPEC; ++s) out(c, s) = y[s];
    out(c, NSPEC) = e;
  });
  auto nsub_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nsub);
  double total = 0.0;
  for (int c = 0; c < n; ++c) total += nsub_h(c);
  std::printf("cells %d, n_H %g, dt %g Myr, nsub_max %d, mean substeps %.2f\n", n, n_h,
              dt, nsub_max, total/n);
  std::printf("src/chemistry: %.4e s, %.4e s per cell\n", t_old, t_old/n);
  std::printf("chemistry2:    %.4e s, %.4e s per cell, ratio new/old %.3f\n", t_new,
              t_new/n, t_new/t_old);
  return 0;
}
