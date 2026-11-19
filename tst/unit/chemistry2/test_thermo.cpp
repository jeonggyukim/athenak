//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_thermo.cpp
//! \brief Every heating and cooling term of chemistry2 against src/chemistry, on random
//!        physical arguments. The terms were moved with their arithmetic unchanged, so
//!        they must agree bit for bit; the test guards later edits.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "athena.hpp"
#include "check.hpp"
#include <Kokkos_ArithTraits.hpp>
#include "chemistry/chemistry_utils.hpp"  // src/chemistry's thermo.hpp needs it first
#include "chemistry/thermo/thermo.hpp"
#include "chemistry2/thermo/thermo.hpp"

namespace {

using Old = chemistry::Thermo;
using New = chemistry2::Thermo;

std::mt19937_64 rng(7);

// log-uniform in [lo, hi]
Real LogU(Real lo, Real hi) {
  std::uniform_real_distribution<double> u(std::log(lo), std::log(hi));
  return std::exp(u(rng));
}
Real U(Real lo, Real hi) {
  std::uniform_real_distribution<double> u(lo, hi);
  return u(rng);
}

// Compares f_new and f_old on n draws of args(); counts values that differ
template <class Args, class FOld, class FNew>
void Compare(unit::Report &report, const std::string &name, Args args, FOld f_old,
             FNew f_new, int n = 20000) {
  int n_differ = 0, n_nonfinite = 0;
  for (int s = 0; s < n; ++s) {
    const auto a = args();
    const Real o = f_old(a), w = f_new(a);
    if (!std::isfinite(o)) ++n_nonfinite;
    if (!(o == w) && !(std::isnan(o) && std::isnan(w))) ++n_differ;
  }
  char what[160];
  std::snprintf(what, sizeof(what),
                "%s: equal to src/chemistry on %d draws (%d non-finite)", name.c_str(), n,
                n_nonfinite);
  report.Expect(n_differ == 0, what);
}

struct A {
  Real xH2, xHI, xe, xCII, xCI, xOI, xCO, nH, ne, T, G, Zd, k1, k2, k3, NCO;
};

A Draw() {
  A a;
  a.xH2 = U(0.0, 0.5);
  a.xHI = U(0.0, 1.0);
  a.xe = LogU(1e-8, 1.0);
  a.xCII = LogU(1e-10, 1.6e-4);
  a.xCI = LogU(1e-10, 1.6e-4);
  a.xOI = LogU(1e-10, 3.2e-4);
  a.xCO = LogU(1e-12, 1.6e-4);
  a.nH = LogU(1e-2, 1e6);
  a.ne = a.xe*a.nH;
  a.T = LogU(2.0, 1e8);
  a.G = LogU(1e-6, 1e4);
  a.Zd = LogU(0.1, 3.0);
  a.k1 = LogU(1e-20, 1e-8);
  a.k2 = LogU(1e-20, 1e-8);
  a.k3 = LogU(1e-20, 1e-8);
  a.NCO = LogU(1e10, 1e22);
  return a;
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report r;

  Compare(r, "CvCold", Draw,
          [](const A &a) { return Old::CvCold(a.xH2, 0.1, a.xe, 5.0/3.0); },
          [](const A &a) { return New::CvCold(a.xH2, 0.1, a.xe, 5.0/3.0); });
  Compare(r, "HeatingCr", Draw,
          [](const A &a) { return Old::HeatingCr(a.xe, a.nH, a.xHI, a.xH2, a.k1); },
          [](const A &a) { return New::HeatingCr(a.xe, a.nH, a.xHI, a.xH2, a.k1); });
  Compare(r, "HeatingPE", Draw,
          [](const A &a) { return Old::HeatingPE(a.G, a.Zd, a.T, a.ne); },
          [](const A &a) { return New::HeatingPE(a.G, a.Zd, a.T, a.ne); });
  Compare(r, "HeatingH2gr", Draw,
          [](const A &a) {
            return Old::HeatingH2gr(a.xHI, a.xH2, a.nH, a.T, a.k1, a.k2);
          },
          [](const A &a) {
            return New::HeatingH2gr(a.xHI, a.xH2, a.nH, a.T, a.k1, a.k2);
          });
  Compare(r, "HeatingH2pump", Draw,
          [](const A &a) { return Old::HeatingH2pump(a.xHI, a.xH2, a.nH, a.T, a.k1); },
          [](const A &a) { return New::HeatingH2pump(a.xHI, a.xH2, a.nH, a.T, a.k1); });
  Compare(r, "HeatingH2diss", Draw,
          [](const A &a) { return Old::HeatingH2diss(a.k1, a.xH2); },
          [](const A &a) { return New::HeatingH2diss(a.k1, a.xH2); });
  Compare(r, "CoolingCII", Draw,
          [](const A &a) {
            return Old::CoolingCII(a.xCII, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T);
          },
          [](const A &a) {
            return New::CoolingCII(a.xCII, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T);
          });
  Compare(r, "CoolingCI", Draw,
          [](const A &a) {
            return Old::CoolingCI(a.xCI, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T);
          },
          [](const A &a) {
            return New::CoolingCI(a.xCI, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T);
          });
  Compare(r, "CoolingOI", Draw,
          [](const A &a) {
            return Old::CoolingOI(a.xOI, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T);
          },
          [](const A &a) {
            return New::CoolingOI(a.xOI, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T);
          });
  Compare(r, "CoolingLya", Draw,
          [](const A &a) { return Old::CoolingLya(a.xHI, a.ne, a.T); },
          [](const A &a) { return New::CoolingLya(a.xHI, a.ne, a.T); });
  Compare(r, "CoolingCOR", Draw,
          [](const A &a) {
            return Old::CoolingCOR(a.xCO, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T, a.NCO);
          },
          [](const A &a) {
            return New::CoolingCOR(a.xCO, a.xHI*a.nH, a.xH2*a.nH, a.ne, a.T, a.NCO);
          });
  Compare(r, "CoolingH2", Draw,
          [](const A &a) {
            return Old::CoolingH2(a.xH2, a.xHI*a.nH, a.xH2*a.nH, 0.1*a.nH, 0.1*a.ne, a.ne,
                                  a.T);
          },
          [](const A &a) {
            return New::CoolingH2(a.xH2, a.xHI*a.nH, a.xH2*a.nH, 0.1*a.nH, 0.1*a.ne, a.ne,
                                  a.T);
          });
  Compare(r, "CoolingRec", Draw,
          [](const A &a) { return Old::CoolingRec(a.Zd, a.T, a.ne, a.G); },
          [](const A &a) { return New::CoolingRec(a.Zd, a.T, a.ne, a.G); });
  Compare(r, "CoolingH2diss", Draw,
          [](const A &a) { return Old::CoolingH2diss(a.xHI, a.xH2, a.k1, a.k2); },
          [](const A &a) { return New::CoolingH2diss(a.xHI, a.xH2, a.k1, a.k2); });
  Compare(r, "CoolingHIion", Draw,
          [](const A &a) { return Old::CoolingHIion(a.xHI, a.xe, a.k1); },
          [](const A &a) { return New::CoolingHIion(a.xHI, a.xe, a.k1); });

  return r.ExitCode();
}
