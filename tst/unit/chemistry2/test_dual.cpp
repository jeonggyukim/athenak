//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_dual.cpp
//! \brief Dual-number derivatives against the analytic derivative.

#include <cmath>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/util/dual.hpp"

namespace {
using chemistry2::dual::Dual;

template <class S>
S F(const S T) {
  using chemistry2::dual::exp;
  using chemistry2::dual::sqrt;
  using chemistry2::dual::pow;
  using chemistry2::dual::log;
  return T*exp(-91.2/T)/sqrt(T) + pow(T, 0.7)*log(T)/(1.0 + T);
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  for (const Real T : {10.0, 87.0, 1.0e3, 2.3e4}) {
    const Dual f = F(Dual(T, 1.0));
    const Real a = std::exp(-91.2/T), s = std::sqrt(T), p = std::pow(T, 0.7);
    const Real exact = a/s*(0.5 + 91.2/T)
        + (0.7*p/T*std::log(T) + p/T)/(1.0 + T) - p*std::log(T)/((1.0 + T)*(1.0 + T));
    report.ExpectNear(f.d, exact, 1e-13, 0.0, "dF/dT at T = " + std::to_string(T));
  }
  return report.ExitCode();
}
