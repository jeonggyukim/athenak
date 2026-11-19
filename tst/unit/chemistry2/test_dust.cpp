//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_dust.cpp
//! \brief Gas-dust collisional exchange: value against Z_d alpha_gd n_H T^(1/2)
//!        (T - T_d) with the Hollenbach & McKee (1989) alpha_gd, and the dual-number
//!        derivative against the analytic one.

#include <cmath>
#include <string>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/thermo/dust.hpp"

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  namespace d = chemistry2::dust;
  const Real n_h = 1.0e5, T_d = 10.0, z_d = 1.0;
  for (const Real T : {8.0, 30.0, 100.0, 1.0e3}) {
    const Real a = 3.8e-33*(1.0 - 0.8*std::exp(-75.0/T));
    const Real want = z_d*a*n_h*std::sqrt(T)*(T - T_d);
    report.ExpectNear(d::GasToDust(n_h, T, T_d, z_d, 0.0), want, 1e-14, 0.0,
                      "GasToDust at T = " + std::to_string(T));
    const Real da = 3.8e-33*(-0.8)*std::exp(-75.0/T)*75.0/(T*T);
    const Real dwant = z_d*n_h*(da*std::sqrt(T)*(T - T_d) +
                                a*(0.5/std::sqrt(T)*(T - T_d) + std::sqrt(T)));
    const auto g = d::GasToDust(n_h, chemistry2::dual::Dual(T, 1.0), T_d, z_d, 0.0);
    report.ExpectNear(g.d, dwant, 1e-13, 0.0, "d/dT at T = " + std::to_string(T));
  }
  report.ExpectNear(d::GasToDust(n_h, 50.0, T_d, z_d, 2.0e-33),
                    2.0e-33*n_h*std::sqrt(50.0)*40.0, 1e-14, 0.0, "fixed alpha_gd");
  return report.ExitCode();
}
