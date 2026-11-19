//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_gow17_tabulated.cpp
//! \brief GOW17 with the rate table against GOW17 with the analytic fits, on random
//!        states: creation and destruction of every species, and heating and cooling
//!        separately, each within 1e-2 at the 99.9th percentile and 3e-2 at worst
//!        (relative).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/gow17.hpp"

namespace {
using Tab = chemistry2::GOW17;
using Ana = chemistry2::GOW17Analytic;
constexpr int NSPEC = Tab::NSPEC;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;

std::mt19937_64 rng(17);
Real LogU(Real lo, Real hi) {
  std::uniform_real_distribution<double> u(std::log(lo), std::log(hi));
  return std::exp(u(rng));
}

template <class N>
typename N::Cell MakeCell(const typename N::Settings &s, Real n_h, Real g0, Real xi) {
  const chemistry2::UnitFactors u = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  return N::MakeCell(1.0, GAMMA, s, u, chemistry2::CellInputs{g0, xi, 0.0});
}

template <class N, int K>
chemistry2::CD RatesOf(const Real *y, Real e, const typename N::Cell &c,
                       const typename N::Substep &s) {
  return N::template Rates<K>(y, e, c, s);
}

template <int... K>
void Compare(const Real *y, Real e, const Tab::Cell &ct, const Tab::Substep &st,
             const Ana::Cell &ca, const Ana::Substep &sa, std::vector<double> *dc,
             std::vector<double> *dd, std::integer_sequence<int, K...>) {
  auto one = [&](auto k) {
    constexpr int KK = decltype(k)::value;
    if constexpr (KK != Tab::IHCO_PLUS || true) {
      const chemistry2::CD a = RatesOf<Ana, KK>(y, e, ca, sa);
      const chemistry2::CD t = RatesOf<Tab, KK>(y, e, ct, st);
      if (a.c > 0) dc->push_back(std::fabs(t.c - a.c)/a.c);
      if (a.d > 0) dd->push_back(std::fabs(t.d - a.d)/a.d);
    }
  };
  (one(std::integral_constant<int, K>{}), ...);
}

double Percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[static_cast<size_t>(p*(v.size() - 1))];
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  const Real inf = Kokkos::Experimental::infinity_v<Real>;
  Tab::Settings st{1.0, 0.1, 1.6e-4, 3.2e-4, 1.7e-6, 1.0, inf, inf, 1.0, 1.0e9,
                   3.0e20, true, false};
  st.table = chemistry2::gow17::BuildGOW17Table(false);
  const Ana::Settings sa{1.0, 0.1, 1.6e-4, 3.2e-4, 1.7e-6, 1.0, inf, inf, 1.0, 1.0e9,
                         3.0e20, true, false};
  std::vector<double> dc, dd, de;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int n = 0; n < 2000; ++n) {
    const Real n_h = LogU(1.0, 1.0e4), g0 = LogU(1.0e-3, 1.0e2);
    const Real xi = LogU(1.0e-17, 1.0e-15), T = LogU(10.0, 1.0e6);
    Real y[NSPEC] = {};
    y[Tab::IHE_PLUS] = 0.1*u(rng)*u(rng);
    y[Tab::IC_PLUS] = 1.6e-4*u(rng);
    y[Tab::ICO] = (1.6e-4 - y[Tab::IC_PLUS])*u(rng);
    y[Tab::IHCO_PLUS] = 1e-9*u(rng);
    y[Tab::ICHX] = 1e-8*u(rng);
    y[Tab::IOHX] = 1e-7*u(rng);
    y[Tab::IH2] = 0.5*u(rng);
    y[Tab::IH_PLUS] = (1.0 - 2*y[Tab::IH2])*u(rng)*u(rng);
    y[Tab::IH2_PLUS] = 1e-10*u(rng);
    y[Tab::IH3_PLUS] = 1e-9*u(rng);
    y[Tab::IO_PLUS] = 1e-6*u(rng);
    y[Tab::ISI_PLUS] = 1.7e-6*u(rng);
    Real x_e = 0.0;
    for (int s = 0; s < NSPEC; ++s) x_e += Tab::rows[s].charge*y[s];
    const Real e = n_h*chemistry2::Thermo::CvCold(y[Tab::IH2], 0.1, x_e, GAMMA)*T/
                   U_ENERGY;
    const Tab::Cell ct = MakeCell<Tab>(st, n_h, g0, xi);
    const Ana::Cell ca = MakeCell<Ana>(sa, n_h, g0, xi);
    const Tab::Substep subt = Tab::Prepare(y, e, ct);
    const Ana::Substep suba = Ana::Prepare(y, e, ca);
    Compare(y, e, ct, subt, ca, suba, &dc, &dd, std::make_integer_sequence<int, NSPEC>{});
    // heating and cooling separately: their difference cancels where they balance
    const auto ha = Ana::HeatingCooling(y, e, ca, suba);
    const auto ht = Tab::HeatingCooling(y, e, ct, subt);
    if (ha.heating > 0) de.push_back(std::fabs(ht.heating - ha.heating)/ha.heating);
    if (ha.cooling > 0) de.push_back(std::fabs(ht.cooling - ha.cooling)/ha.cooling);
  }
  const double c999 = Percentile(dc, 0.999), cmax = Percentile(dc, 1.0);
  const double d999 = Percentile(dd, 0.999), dmax = Percentile(dd, 1.0);
  const double e999 = Percentile(de, 0.999), emax = Percentile(de, 1.0);
  std::printf("  creation: 99.9%% %.2e max %.2e; destruction: 99.9%% %.2e max %.2e; "
              "heating and cooling: 99.9%% %.2e max %.2e\n", c999, cmax, d999, dmax, e999,
              emax);
  report.Expect(c999 <= 1e-2 && cmax <= 3e-2, "creation rates, table vs analytic");
  report.Expect(d999 <= 1e-2 && dmax <= 3e-2, "destruction rates, table vs analytic");
  report.Expect(e999 <= 1e-2 && emax <= 3e-2, "heating and cooling, table vs analytic");
  return report.ExitCode();
}
