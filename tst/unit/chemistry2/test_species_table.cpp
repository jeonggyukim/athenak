//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_species_table.cpp
//! \brief Element budgets and electron abundance from a species table, against the
//!        hand-written GOW17 formulas of src/chemistry/network/gow17.hpp
//!        (ComputeGhostSpecies_), on host and device.
//!
//!   The species table is the GOW17 one, in the storage order of src/chemistry. Its
//!   budgets subtract the species in storage order, the hand-written ones in their own
//!   order, so the two agree to round-off, not bit for bit.

#include <array>
#include <cstdio>
#include <random>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/network_config.hpp"
#include "chemistry2/network/species_table.hpp"

using chemistry2::ELEM_C;
using chemistry2::ELEM_H;
using chemistry2::ELEM_HE;
using chemistry2::ELEM_O;
using chemistry2::ELEM_SI;
using chemistry2::Species;
using chemistry2::SpeciesRow;

namespace {

struct GOW17Rows {
  static constexpr std::array<SpeciesRow, 12> rows = {
      Species("He+").Charge(1).He(1),
      Species("OHx").H(1).O(1),
      Species("CHx").H(1).C(1),
      Species("CO").C(1).O(1),
      Species("C+").Charge(1).C(1),
      Species("HCO+").Charge(1).H(1).C(1).O(1),
      Species("H2").H(2),
      Species("H+").Charge(1).H(1),
      Species("H3+").Charge(1).H(3),
      Species("H2+").Charge(1).H(2),
      Species("O+").Charge(1).O(1),
      Species("Si+").Charge(1).Si(1)};
  static constexpr bool STORES_ELECTRONS = false;
  static constexpr int ROW_EL = -1;
  static constexpr int HYDROGEN_REMAINDER_CHARGE = 0;
};

enum : int {IHE_PLUS, IOHX, ICHX, ICO, IC_PLUS, IHCO_PLUS, IH2, IH_PLUS, IH3_PLUS,
            IH2_PLUS, IO_PLUS, ISI_PLUS, NSPEC};

static_assert(chemistry2::NumSpecies<GOW17Rows>() == NSPEC);
static_assert(chemistry2::UniqueNames(GOW17Rows::rows));
static_assert(chemistry2::SpeciesIndex(GOW17Rows::rows, "HCO+") == IHCO_PLUS);
static_assert(chemistry2::SpeciesIndex(GOW17Rows::rows, "CHx") == ICHX);
static_assert(chemistry2::SpeciesIndex(GOW17Rows::rows, "e-") == -1);
static_assert(chemistry2::AnyCharged(GOW17Rows::rows));

constexpr Real X_HE = 0.1, X_C = 1.6e-4, X_O = 3.2e-4, X_SI = 1.7e-6;

// Element budgets and electrons as src/chemistry/network/gow17.hpp writes them
struct Ghosts {
  Real H, He, C, O, Si, e;
};

KOKKOS_INLINE_FUNCTION Ghosts HandWritten(const Real *y) {
  Ghosts g;
  g.Si = Kokkos::fmax(X_SI - y[ISI_PLUS], 0.0);
  g.C = Kokkos::fmax(X_C - y[IHCO_PLUS] - y[ICHX] - y[ICO] - y[IC_PLUS], 0.0);
  g.O = Kokkos::fmax(X_O - y[IHCO_PLUS] - y[IOHX] - y[ICO] - y[IO_PLUS], 0.0);
  g.He = Kokkos::fmax(X_HE - y[IHE_PLUS], 0.0);
  g.e = y[IHE_PLUS] + y[IC_PLUS] + y[IHCO_PLUS] + y[IH3_PLUS] + y[IH2_PLUS] +
        y[IH_PLUS] + y[IO_PLUS] + y[ISI_PLUS];
  g.H = Kokkos::fmax(1.0 - (y[IOHX] + y[ICHX] + y[IHCO_PLUS] + 3.0*y[IH3_PLUS] +
                            2.0*y[IH2_PLUS] + y[IH_PLUS] + 2.0*y[IH2]), 0.0);
  return g;
}

KOKKOS_INLINE_FUNCTION Ghosts FromTable(const Real *y) {
  Ghosts g;
  g.Si = chemistry2::ElementBudget<GOW17Rows, ELEM_SI>(y, X_SI);
  g.C = chemistry2::ElementBudget<GOW17Rows, ELEM_C>(y, X_C);
  g.O = chemistry2::ElementBudget<GOW17Rows, ELEM_O>(y, X_O);
  g.He = chemistry2::ElementBudget<GOW17Rows, ELEM_HE>(y, X_HE);
  g.e = chemistry2::Electrons<GOW17Rows>(y);
  g.H = chemistry2::ElementBudget<GOW17Rows, ELEM_H>(y, 1.0);
  return g;
}

// A state with each element split at random among its stored species and its neutral
void RandomState(std::mt19937_64 &rng, Real *y) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto split = [&](Real total, std::initializer_list<int> idx, std::initializer_list<int>
                   atoms) {
    Real left = total;
    auto a = atoms.begin();
    for (int i : idx) {
      const Real take = left*u(rng)*u(rng);
      y[i] = take/(*a);
      left -= take;
      ++a;
    }
  };
  for (int n = 0; n < NSPEC; ++n) y[n] = 0.0;
  split(X_HE, {IHE_PLUS}, {1});
  split(X_SI, {ISI_PLUS}, {1});
  split(X_C, {IC_PLUS, ICO, ICHX, IHCO_PLUS}, {1, 1, 1, 1});
  split(X_O - y[ICO] - y[IHCO_PLUS], {IO_PLUS, IOHX}, {1, 1});
  split(1.0 - y[IOHX] - y[ICHX] - y[IHCO_PLUS], {IH2, IH_PLUS, IH2_PLUS, IH3_PLUS},
        {2, 1, 2, 3});
}

void CompareGhosts(unit::Report &report, const Ghosts &got, const Ghosts &want,
                   const char *label) {
  char what[128];
  const double rtol = 1.0e-13, atol = 1.0e-18;
  std::snprintf(what, sizeof(what), "%s H", label);
  report.ExpectNear(got.H, want.H, rtol, atol, what);
  std::snprintf(what, sizeof(what), "%s He", label);
  report.ExpectNear(got.He, want.He, rtol, atol, what);
  std::snprintf(what, sizeof(what), "%s C", label);
  report.ExpectNear(got.C, want.C, rtol, atol, what);
  std::snprintf(what, sizeof(what), "%s O", label);
  report.ExpectNear(got.O, want.O, rtol, atol, what);
  std::snprintf(what, sizeof(what), "%s Si", label);
  report.ExpectNear(got.Si, want.Si, rtol, atol, what);
  std::snprintf(what, sizeof(what), "%s e", label);
  report.ExpectNear(got.e, want.e, rtol, atol, what);
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;

  constexpr int n_states = 1000;
  std::mt19937_64 rng(20261119);
  Kokkos::View<Real*[NSPEC]> y_d("y", n_states);
  auto y_h = Kokkos::create_mirror_view(y_d);
  for (int s = 0; s < n_states; ++s) RandomState(rng, &y_h(s, 0));
  Kokkos::deep_copy(y_d, y_h);

  // Host: the table against the hand-written formulas, worst state per quantity printed
  Ghosts worst_got{}, worst_want{};
  double worst = -1.0;
  int n_nonpositive = 0;
  for (int s = 0; s < n_states; ++s) {
    const Ghosts got = FromTable(&y_h(s, 0));
    const Ghosts want = HandWritten(&y_h(s, 0));
    const Real g6[6] = {got.H, got.He, got.C, got.O, got.Si, got.e};
    const Real w6[6] = {want.H, want.He, want.C, want.O, want.Si, want.e};
    for (int q = 0; q < 6; ++q) {
      if (w6[q] <= 0.0) ++n_nonpositive;
      const double v = std::fabs(g6[q] - w6[q])/w6[q];
      if (v > worst) {
        worst = v;
        worst_got = got;
        worst_want = want;
      }
    }
  }
  report.Expect(n_nonpositive == 0, "every random state leaves every budget positive");
  CompareGhosts(report, worst_got, worst_want, "host, worst state:");

  // Device: the same states, compared with the host result
  Kokkos::View<Real*[6]> g_d("g", n_states);
  Kokkos::parallel_for("species_table", n_states, KOKKOS_LAMBDA(const int s) {
    Real y[NSPEC];
    for (int n = 0; n < NSPEC; ++n) y[n] = y_d(s, n);
    const Ghosts g = FromTable(y);
    g_d(s, 0) = g.H;
    g_d(s, 1) = g.He;
    g_d(s, 2) = g.C;
    g_d(s, 3) = g.O;
    g_d(s, 4) = g.Si;
    g_d(s, 5) = g.e;
  });
  auto g_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g_d);
  int n_differ = 0;
  for (int s = 0; s < n_states; ++s) {
    const Ghosts h = FromTable(&y_h(s, 0));
    const Real hv[6] = {h.H, h.He, h.C, h.O, h.Si, h.e};
    for (int q = 0; q < 6; ++q) {
      if (std::fabs(g_h(s, q) - hv[q]) > 1.0e-15*std::fabs(hv[q])) ++n_differ;
    }
  }
  report.Expect(n_differ == 0, "device equals host to 1e-15 relative, every state");

  // A depleted element floors at zero
  Real y[NSPEC] = {};
  y[ICO] = 2.0*X_C;
  report.Expect(chemistry2::ElementBudget<GOW17Rows, ELEM_C>(y, X_C) == 0.0,
                "C budget floors at zero when the stored species exceed x_C");
  report.Expect(chemistry2::ElementRemainder<GOW17Rows, ELEM_C>(y, X_C) == -X_C,
                "C remainder is not floored");

  return report.ExitCode();
}
