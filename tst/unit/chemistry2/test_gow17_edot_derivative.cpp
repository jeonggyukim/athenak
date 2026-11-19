//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_gow17_edot_derivative.cpp
//! \brief GOW17 with the rate table: d(edot)/de from dual numbers against a centered
//!        difference over 1e-7 e (Prepare called again at each e), on random states
//!        away from table rows; also checks that the dual pass returns the same edot.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/gow17.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {
using Net = chemistry2::GOW17;
constexpr int NSPEC = Net::NSPEC;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;

std::mt19937_64 rng(23);
Real LogU(Real lo, Real hi) {
  std::uniform_real_distribution<double> u(std::log(lo), std::log(hi));
  return std::exp(u(rng));
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  const Real inf = Kokkos::Experimental::infinity_v<Real>;
  Net::Settings st{1.0, 0.1, 1.6e-4, 3.2e-4, 1.7e-6, 1.0, inf, inf, 1.0, 1.0e9,
                   3.0e20, true, false};
  st.table = chemistry2::gow17::BuildGOW17Table(false);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> err;
  double worst_edot = 0.0;
  int skipped = 0;
  for (int n = 0; n < 2000; ++n) {
    const Real n_h = LogU(1.0, 1.0e4), g0 = LogU(1.0e-3, 1.0e2);
    const Real xi = LogU(1.0e-17, 1.0e-15), T = LogU(10.0, 1.0e6);
    Real y[NSPEC] = {};
    y[Net::IHE_PLUS] = 0.1*u(rng)*u(rng);
    y[Net::IC_PLUS] = 1.6e-4*u(rng);
    y[Net::ICO] = (1.6e-4 - y[Net::IC_PLUS])*u(rng);
    y[Net::IHCO_PLUS] = 1e-9*u(rng);
    y[Net::ICHX] = 1e-8*u(rng);
    y[Net::IOHX] = 1e-7*u(rng);
    y[Net::IH2] = 0.5*u(rng);
    y[Net::IH_PLUS] = (1.0 - 2*y[Net::IH2])*u(rng)*u(rng);
    y[Net::IO_PLUS] = 1e-6*u(rng);
    y[Net::ISI_PLUS] = 1.7e-6*u(rng);
    Real x_e = 0.0;
    for (int s = 0; s < NSPEC; ++s) x_e += Net::rows[s].charge*y[s];
    const Real e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, x_e, GAMMA)*T/
                   U_ENERGY;
    const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
        n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
    st.dust_cooling = (n % 2 == 1);  // half the states with gas-dust exchange
    const Net::Cell c = Net::MakeCell(1.0, GAMMA, st, uf,
                                      chemistry2::CellInputs{g0, xi, 0.0});
    const Real h = 1.0e-7;
    const chemistry2::Slot s0 = st.table.grid.Locate(T);
    const chemistry2::Slot sp = st.table.grid.Locate(T*(1 + h));
    const chemistry2::Slot sm = st.table.grid.Locate(T*(1 - h));
    const bool near_switch = std::fabs(T/7.0e2 - 1.0) < 1e-5 ||
                             std::fabs(T/2.0e3 - 1.0) < 1e-5 ||
                             std::fabs(T/1.0e6 - 1.0) < 1e-5;
    if (sp.i0 != s0.i0 || sm.i0 != s0.i0 || near_switch) {
      ++skipped;
      continue;
    }
    const Net::Substep sub = Net::Prepare(y, e, c);
    Real d_dual = 0.0;
    const Real edot = Net::EdotWithDerivative(y, e, c, sub, &d_dual);
    const Real edot_ref = Net::Edot(y, e, c, sub);
    worst_edot = std::fmax(worst_edot, std::fabs(edot - edot_ref)/
                           std::fmax(std::fabs(edot_ref), 1e-300));
    const Real ep = e*(1 + h), em = e*(1 - h);
    const Real fp = Net::Edot(y, ep, c, Net::Prepare(y, ep, c));
    const Real fm = Net::Edot(y, em, c, Net::Prepare(y, em, c));
    const Real d_fd = (fp - fm)/(ep - em);
    const auto hc = Net::HeatingCooling(y, e, c, sub);
    const Real scale = c.u_time*c.n_h*c.inv_u_energy*(hc.heating + hc.cooling)/e;
    err.push_back(std::fabs(d_dual - d_fd)/scale);
  }
  std::sort(err.begin(), err.end());
  const double e999 = err[static_cast<size_t>(0.999*(err.size() - 1))];
  const double emax = err.back();
  std::printf("  %zu states (%d skipped): |dual - difference| / ((heating + cooling)/e) "
              "99.9%% %.2e max %.2e; edot from the dual pass vs Edot %.2e\n",
              err.size(), skipped, e999, emax, worst_edot);
  report.ExpectNear(worst_edot, 0.0, 0.0, 1e-13, "edot of the dual pass equals Edot");
  report.ExpectNear(e999, 0.0, 0.0, 1e-6, "d(edot)/de, 99.9th percentile");
  report.ExpectNear(emax, 0.0, 0.0, 1e-4, "d(edot)/de, worst state");

  // Gas below the rate floor (t_min_rates 10 K, T 5 K, n_H 1e5, xi_CR 1e-22, net
  // cooling), dt 10 Myr: the solve must not shrink e one substep at a time up to
  // nsub_max (measured 100000 substeps before the fix, 1 after, 2026-11-20)
  {
    Net::Settings sf = st;
    sf.t_min_rates = 10.0;
    sf.dust_cooling = false;
    Real y[NSPEC] = {};
    y[Net::IH2] = 0.5;
    y[Net::ICO] = 1.5e-4;
    y[Net::IC_PLUS] = 1.0e-8;
    const Real n_h = 1.0e5;
    const Real e = n_h*chemistry2::Thermo::CvCold(0.5, 0.1, 1.0e-8, GAMMA)*5.0/U_ENERGY;
    const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
        n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
    const Net::Cell c = Net::MakeCell(1.0, GAMMA, sf, uf,
                                      chemistry2::CellInputs{1e-6, 1e-22, 0.0});
    Real ee = e;
    const chemistry2::SemiImplicitSettings si{0.1, 1.0e-12, 100000, true};
    const int nsub = chemistry2::SemiImplicit<Net>::Solve(si, c, 0.0, 10.0, y, ee);
    std::printf("  below the rate floor: %d substeps\n", nsub);
    report.Expect(nsub < 1000, "below t_min_rates: substeps < 1000");
  }
  return report.ExitCode();
}
