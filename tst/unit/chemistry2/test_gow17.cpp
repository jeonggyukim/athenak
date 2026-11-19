//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_gow17.cpp
//! \brief The chemistry2 GOW17 network against src/chemistry GOW17Network: right-hand
//!        sides (dy/dt of every species, de/dt) on random states, and one-cell solves
//!        with the semi-implicit solver.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>

#include <Kokkos_ArithTraits.hpp>  // src/chemistry's headers need it first
#include "athena.hpp"
#include "check.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"
#include "chemistry/chemistry_utils.hpp"
#include "chemistry/network/gow17.hpp"
#include "ode_solvers/semi_implicit.hpp"
#include "chemistry2/network/gow17.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {

using New = chemistry2::GOW17Analytic;  // same fits as src/chemistry, no table
using Old = chemistry::GOW17Network;
constexpr int NSPEC = New::NSPEC;

constexpr Real MU_H = 1.4;
constexpr Real GAMMA = 5.0/3.0;
constexpr Real U_TIME = 3.15576e13;  // 1 Myr [s]
constexpr Real U_ENERGY = 1.0e-12;   // [erg cm^-3]

std::mt19937_64 rng(11);
Real LogU(Real lo, Real hi) {
  std::uniform_real_distribution<double> u(std::log(lo), std::log(hi));
  return std::exp(u(rng));
}

struct State {
  Real n_h, g0, xi_cr, e;
  Real y[NSPEC];
};

// A random state: each element split among its stored species, the rest neutral
State Draw() {
  State s;
  s.n_h = LogU(1.0, 1.0e4);
  s.g0 = LogU(1.0e-3, 1.0e2);
  s.xi_cr = LogU(1.0e-17, 1.0e-15);
  for (Real &v : s.y) v = 0.0;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto split = [&](Real total, std::initializer_list<int> idx) {
    for (int i : idx) {
      s.y[i] = total*u(rng)*u(rng);
      total -= s.y[i];
    }
  };
  split(0.1, {New::IHE_PLUS});
  split(1.7e-6, {New::ISI_PLUS});
  split(1.6e-4, {New::IC_PLUS, New::ICO, New::ICHX, New::IHCO_PLUS});
  split(3.2e-4 - s.y[New::ICO] - s.y[New::IHCO_PLUS], {New::IO_PLUS, New::IOHX});
  const Real h_left = 1.0 - s.y[New::IOHX] - s.y[New::ICHX] - s.y[New::IHCO_PLUS];
  s.y[New::IH2] = 0.5*h_left*u(rng);
  split(h_left - 2*s.y[New::IH2], {New::IH_PLUS});
  s.y[New::IH2_PLUS] = 1e-10*u(rng);
  s.y[New::IH3_PLUS] = 1e-10*u(rng);
  const Real T = LogU(10.0, 1.0e4);
  Real x_e = 0.0;
  for (int n = 0; n < NSPEC; ++n) x_e += New::rows[n].charge*s.y[n];
  const Real cv = chemistry2::Thermo::CvCold(s.y[New::IH2], 0.1, x_e, GAMMA);
  s.e = s.n_h*cv*T/U_ENERGY;
  return s;
}

Real DensityCgs(Real n_h) { return n_h*MU_H*units::Units::hydrogen_mass_cgs; }

chemistry::GOW17Settings OldSettings() {
  chemistry::GOW17Settings s{};
  s.isothermal = false;
  s.isothermal_temperature = 0.0;
  s.zd = 1.0;
  s.xHe = 0.1;
  s.xC = 1.6e-4;
  s.xO = 3.2e-4;
  s.xSi = 1.7e-6;
  s.temperature_min_rates = 1.0;
  s.temperature_max_rates = Kokkos::Experimental::infinity_v<Real>;
  s.temperature_max_heating = Kokkos::Experimental::infinity_v<Real>;
  s.temperature_min_cooling = 1.0;
  s.temperature_max_cooling_nm = 1.0e9;
  s.Leff_CO_max = 3.0e20;
  s.H2_rovib_cooling = true;
  s.is_kgrH2_const = false;
  s.velocity_cgs = 1.0;
  s.length_cgs = 1.0;
  s.three_d = true;
  s.multi_d = true;
  return s;
}

New::Settings NewSettings() {
  const Real inf = Kokkos::Experimental::infinity_v<Real>;
  return New::Settings{1.0, 0.1, 1.6e-4, 3.2e-4, 1.7e-6, 1.0, inf, inf, 1.0, 1.0e9,
                       3.0e20, true, false};
}

// The old network, built for one cell (1,1,1) of a uniform 3^3 block at rest
struct OldCell {
  DvceArray5D<Real> w0;
  DualArray1D<RegionSize> sizes;
  DvceArray1D<Real> ir;
  explicit OldCell(const State &s)
      : w0("w0", 1, 5, 3, 3, 3), sizes("sizes", 1), ir("ir", Old::n_freq) {
    Kokkos::deep_copy(w0, 0.0);
    auto w0_h = Kokkos::create_mirror_view(w0);
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) w0_h(0, IDN, k, j, i) = 1.0;
    Kokkos::deep_copy(w0, w0_h);
    sizes.h_view(0).dx1 = sizes.h_view(0).dx2 = sizes.h_view(0).dx3 = 1.0;
    sizes.template modify<HostMemSpace>();
    sizes.template sync<DevExeSpace>();
    auto ir_h = Kokkos::create_mirror_view(ir);
    for (int f = 0; f < Old::n_freq; ++f) ir_h(f) = s.g0;
    ir_h(Old::n_freq - 1) = s.xi_cr;
    Kokkos::deep_copy(ir, ir_h);
  }
  Old Make(const State &s) const {
    Old net(OldSettings(), 0, 1, 1, 1, w0, sizes, ir, DensityCgs(s.n_h), MU_H, GAMMA,
            units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY);
    for (int n = 0; n < NSPEC; ++n) net.y(n) = s.y[n];
    net.y(Old::IIE) = s.e;
    return net;
  }
};

New::Cell NewCell(const State &s) {
  const chemistry2::UnitFactors u =
      chemistry2::MakeUnitFactors(DensityCgs(s.n_h), U_TIME, U_ENERGY, MU_H);
  return New::MakeCell(1.0, GAMMA, NewSettings(), u,
                       chemistry2::CellInputs{s.g0, s.xi_cr, 0.0});
}

template <int K>
Real NewRhsOf(const Real *y, Real e, const New::Cell &c, const New::Substep &sub) {
  const chemistry2::CD r = New::Rates<K>(y, e, c, sub);
  return r.c - y[K]*r.d;
}

template <int... K>
void NewRhs(const Real *y, Real e, const New::Cell &c, const New::Substep &sub,
            Real *f, std::integer_sequence<int, K...>) {
  ((f[K] = NewRhsOf<K>(y, e, c, sub)), ...);
}

double RelDiff(double a, double b, double floor) {
  return std::fabs(a - b)/std::fmax(std::fabs(b), floor);
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  constexpr int N_STATES = 500;

  // Right-hand sides
  double worst_f = 0.0, worst_edot = 0.0;
  for (int n = 0; n < N_STATES; ++n) {
    const State s = Draw();
    OldCell oc(s);
    Old old = oc.Make(s);
    Real f_old[Old::neqs];
    Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> fv(f_old, Old::neqs);
    old.evaluate_function(0.0, 0.0, old.y, fv);
    const New::Cell c = NewCell(s);
    const New::Substep sub = New::Prepare(s.y, s.e, c);
    Real f_new[NSPEC];
    NewRhs(s.y, s.e, c, sub, f_new, std::make_integer_sequence<int, NSPEC>{});
    // relative to the species' own scale: max(|f|, 1e-12 x_elem per Myr)
    for (int k = 0; k < NSPEC; ++k) {
      worst_f = std::fmax(worst_f, RelDiff(f_new[k], f_old[k], 1.0e-20));
    }
    worst_edot = std::fmax(worst_edot,
                           RelDiff(New::Edot(s.y, s.e, c, sub), f_old[Old::IIE], 1e-30));
  }
  std::printf("  right-hand sides on %d states: worst dy/dt %.2e, de/dt %.2e\n", N_STATES,
              worst_f, worst_edot);
  report.ExpectNear(worst_f, 0.0, 0.0, 1.0e-10, "dy/dt against src/chemistry");
  report.ExpectNear(worst_edot, 0.0, 0.0, 1.0e-10, "de/dt against src/chemistry");

  // One-cell solves. A long solve is sensitive to rounding through its step sequence:
  // src/chemistry alone, with e perturbed by 1e-14, moves by up to ~10% in some states
  // (2026-11-20). So parity is checked on maps whose substep sequence is fixed
  // (nsub_max = 1: one substep of h = dt; nsub_max = 2: the first step from the
  // limiter, then the rest), and long solves at default settings only as a smoke test.
  constexpr Real X_ELEM[NSPEC] = {0.1, 3.2e-4, 1.6e-4, 1.6e-4, 1.6e-4, 1.6e-4,
                                  1.0, 1.0, 1.0, 1.0, 3.2e-4, 1.7e-6};
  struct Diff {
    double y, e;
  };
  auto solve_both = [&](const State &s, Real dt, int nsub_max) {
    OldCell oc(s);
    Old old = oc.Make(s);
    ode_solvers::SemiImplicit<Old> solver(
        ode_solvers::SemiImplicitSettings{0.1, 1.0e-12, nsub_max, true}, old, 0.0, dt);
    solver.SolveODE();
    Real y[NSPEC];
    for (int k = 0; k < NSPEC; ++k) y[k] = s.y[k];
    Real e = s.e;
    chemistry2::SemiImplicit<New>::Solve(
        chemistry2::SemiImplicitSettings{0.1, 1.0e-12, nsub_max, true}, NewCell(s), 0.0,
        dt, y, e);
    Diff d{0.0, RelDiff(e, old.y(Old::IIE), 1e-300)};
    for (int k = 0; k < NSPEC; ++k) {
      d.y = std::fmax(d.y, RelDiff(y[k], old.y(k), 1.0e-3*X_ELEM[k]));  // S1b floor
    }
    if (!std::isfinite(e)) d.e = 1e300;
    return d;
  };

  for (int nsub_max : {1, 2}) {
    Diff worst{0.0, 0.0};
    for (int n = 0; n < N_STATES; ++n) {
      const State s = Draw();
      const Real dt = std::pow(10.0, -9.0 + 9.0*n/(N_STATES - 1));  // 1e-9 to 1 Myr
      const Diff d = solve_both(s, dt, nsub_max);
      worst.y = std::fmax(worst.y, d.y);
      worst.e = std::fmax(worst.e, d.e);
    }
    std::printf("  nsub_max %d, dt 1e-9-1 Myr: worst species %.2e, energy %.2e\n",
                nsub_max, worst.y, worst.e);
    char what[96];
    std::snprintf(what, sizeof(what), "nsub_max %d map, species", nsub_max);
    // measured 5.2e-9 (species) and 9.9e-9 (energy), 2026-11-20; 100x margin
    report.ExpectNear(worst.y, 0.0, 0.0, 1.0e-6, what);
    std::snprintf(what, sizeof(what), "nsub_max %d map, energy", nsub_max);
    report.ExpectNear(worst.e, 0.0, 0.0, 1.0e-6, what);
  }

  // Long solves at default settings: smoke test (accuracy is the S1b regression test)
  {
    constexpr int N_LONG = 50;
    double dy[N_LONG], max_e = 0.0;
    for (int n = 0; n < N_LONG; ++n) {
      const State s = Draw();
      const Diff d = solve_both(s, std::pow(10.0, -4.0 + 4.0*n/(N_LONG - 1)), 100000);
      dy[n] = d.y;
      max_e = std::fmax(max_e, d.e);
    }
    std::sort(dy, dy + N_LONG);
    std::printf("  long solves (dt 1e-4-1 Myr): median species %.2e, max %.2e, "
                "max energy %.2e\n", dy[N_LONG/2], dy[N_LONG - 1], max_e);
    report.Expect(dy[N_LONG/2] <= 1.0e-5, "long solves, median species <= 1e-5");
    report.Expect(dy[N_LONG - 1] <= 0.3 && max_e <= 1.0e-2,
                  "long solves, max species <= 0.3 and energy <= 1e-2");
  }
  return report.ExitCode();
}
