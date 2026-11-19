//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ncr_tabulated.cpp
//! \brief NCR with the rate table against NCR with the fits on random states: C and D
//!        of H2 and H+, heating and cooling; and the dual-number d(edot)/de against a
//!        centered difference with Prepare redone (which also moves the closures x_C+,
//!        x_CO with T; the dual pass holds them, so this part is reported, not checked
//!        to round-off).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/ncr.hpp"

namespace {
using Tab = chemistry2::NCR;
using Ana = chemistry2::NCRAnalytic;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;

double Rel(double a, double b) {
  const double s = std::fmax(std::fabs(a), std::fabs(b));
  return (s < 1e-300) ? 0.0 : std::fabs(a - b)/s;
}
double Pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[static_cast<size_t>(p*(v.size() - 1))];
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  Tab::Settings st{1.0, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20, true,
                   false, 10.0, 0.0};
  st.table = chemistry2::ncr::BuildNCRTable();
  const Ana::Settings sa{1.0, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20,
                         true, false, 10.0, 0.0};
  std::mt19937_64 rng(53);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> drate, dheat, dcool, dderiv;
  for (int n = 0; n < 2000; ++n) {
    const Real n_h = std::pow(10.0, -2.0 + 6.0*u(rng));
    const Real T = std::pow(10.0, 1.0 + 3.2*u(rng));
    const Real g = std::pow(10.0, -3.0 + 5.0*u(rng));
    const Real xi = std::pow(10.0, -17.0 + 2.0*u(rng));
    Real y[Tab::NSPEC];
    y[Tab::IH2] = 0.49*u(rng);
    y[Tab::IH_PLUS] = (1.0 - 2.0*y[Tab::IH2])*std::pow(10.0, -6.0*u(rng));
    y[Tab::IE] = y[Tab::IH_PLUS] + 1.6e-4*u(rng);
    const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
        n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
    const chemistry2::CellInputs in{g, xi, 0.0};
    const Tab::Cell ct = Tab::MakeCell(1.0, GAMMA, st, uf, in);
    const Ana::Cell ca = Ana::MakeCell(1.0, GAMMA, sa, uf, in);
    const Real e = n_h*chemistry2::Thermo::CvCold(y[Tab::IH2], 0.1, y[Tab::IE], GAMMA)*
                   T/U_ENERGY;
    const Tab::Substep s_t = Tab::Prepare(y, e, ct);
    const Ana::Substep s_a = Ana::Prepare(y, e, ca);
    for (const auto &p : {std::make_pair(s_t.c_h2, s_a.c_h2),
                          std::make_pair(s_t.d_h2, s_a.d_h2),
                          std::make_pair(s_t.c_hii, s_a.c_hii),
                          std::make_pair(s_t.d_hii, s_a.d_hii)}) {
      drate.push_back(Rel(p.first, p.second));
    }
    const auto ht = Tab::HeatingCooling(y, e, ct, s_t);
    const auto ha = Ana::HeatingCooling(y, e, ca, s_a);
    dheat.push_back(Rel(ht.heating, ha.heating));
    dcool.push_back(Rel(ht.cooling, ha.cooling));
    // derivative: dual against a centered difference, away from table rows
    const Real h = 1.0e-7;
    const chemistry2::Slot s0 = st.table.grid.Locate(T);
    if (st.table.grid.Locate(T*(1 + h)).i0 != s0.i0 ||
        st.table.grid.Locate(T*(1 - h)).i0 != s0.i0 || std::fabs(T/7e2 - 1) < 1e-5 ||
        std::fabs(T/3e3 - 1) < 1e-5) {
      continue;
    }
    Real d_dual = 0.0;
    Tab::EdotWithDerivative(y, e, ct, s_t, &d_dual);
    const Real ep = e*(1 + h), em = e*(1 - h);
    const Real fp = Tab::Edot(y, ep, ct, Tab::Prepare(y, ep, ct));
    const Real fm = Tab::Edot(y, em, ct, Tab::Prepare(y, em, ct));
    const Real scale = ct.u_time*ct.n_h*ct.inv_u_energy*(ht.heating + ht.cooling)/e;
    dderiv.push_back(std::fabs(d_dual - (fp - fm)/(ep - em))/scale);
  }
  std::printf("  rates: 99.9%% %.2e max %.2e; heating 99.9%% %.2e max %.2e; cooling "
              "99.9%% %.2e max %.2e\n", Pct(drate, 0.999), Pct(drate, 1.0),
              Pct(dheat, 0.999), Pct(dheat, 1.0), Pct(dcool, 0.999), Pct(dcool, 1.0));
  std::printf("  d(edot)/de vs centered difference (%zu states), per (heating + "
              "cooling)/e: median %.2e, 99%% %.2e, max %.2e\n", dderiv.size(),
              Pct(dderiv, 0.5), Pct(dderiv, 0.99), Pct(dderiv, 1.0));
  report.ExpectNear(Pct(drate, 1.0), 0.0, 0.0, 3e-2, "rates, table vs fits");
  report.ExpectNear(Pct(dheat, 1.0), 0.0, 0.0, 3e-2, "heating, table vs fits");
  report.ExpectNear(Pct(dcool, 1.0), 0.0, 0.0, 3e-2, "cooling, table vs fits");
  return report.ExitCode();
}
