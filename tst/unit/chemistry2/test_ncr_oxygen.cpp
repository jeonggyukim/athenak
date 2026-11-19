//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ncr_oxygen.cpp
//! \brief NCR with the oxygen ladder (O+, O++), one zone, warm partly ionized gas: the
//!        oxygen budget is conserved, no fraction is negative, the electrons stay equal
//!        to their sum over carriers, and charge transfer locks O+/O to H+/H as
//!        x_O+/x_O = (k_ct,ion/k_ct,rec) x_H+/x_H I when charge transfer dominates
//!        (within 10%).

#include <cmath>
#include <cstdio>
#include <string>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/ncr_oxygen.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace {
using Net = chemistry2::NCROxygen<2>;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  std::string dir(__FILE__);
  dir = dir.substr(0, dir.find_last_of('/')) + "/../../../inputs/chemistry2/ions/";
  Net::Settings st{{1.0, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20, true,
                    false, 10.0, 0.0}, chemistry2::RowTable()};
  st.base.table = chemistry2::ncr::BuildNCRTable();
  st.ions = chemistry2::ions::BuildIonTable(chemistry2::ions::IonRateData(dir),
                                            st.base.table.grid, 8, 2);
  const Real n_h = 1.0;
  const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  const Net::Cell c = Net::MakeCell(1.0, GAMMA, st, uf,
                                    chemistry2::CellInputs{1.0e-3, 1.0e-16, 0.0});
  Real y[Net::NSPEC] = {};
  y[Net::IH2] = 1e-8;
  y[Net::IH_PLUS] = 0.1;
  y[Net::IE] = 0.1 + 1.6e-4;
  const Real T0 = 8.0e3;
  Real e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, y[Net::IE], GAMMA)*T0/
           U_ENERGY;
  const chemistry2::SemiImplicitSettings si{0.1, 1.0e-12, 100000, true};
  const int nsub = chemistry2::SemiImplicit<Net>::Solve(si, c, 0.0, 1.0, y, e);
  const Real T = Net::Temperature(y, e, c);
  const Real x_o = c.x_o, x_oi = x_o - y[3] - y[4];
  const Real x_hi = 1.0 - 2.0*y[Net::IH2] - y[Net::IH_PLUS];
  std::printf("  %d substeps, T %.4g K, x_H+ %.3g, x_e %.3g, O I %.3g O+ %.3g O++ %.3g\n",
              nsub, T, y[Net::IH_PLUS], y[Net::IE], x_oi, y[3], y[4]);
  report.Expect(y[3] >= 0 && y[4] >= 0 && x_oi >= 0, "no negative oxygen fraction");
  // charge-transfer locking at the final T
  const chemistry2::ions::IonRateData d(dir);
  const double k_ion = d.ct.CtIon(8, 8, T), k_rec = d.ct.CtRec(8, 7, T);
  const double want = (k_ion/k_rec)*y[Net::IH_PLUS]/x_hi;
  const double got = y[3]/x_oi;
  std::printf("  O+/O %.4g, CT locking %.4g\n", got, want);
  report.ExpectNear(got, want, 0.1, 0.0, "O+/O locked to H+/H by charge transfer");
  // electrons: their sum over carriers, with the oxygen ions in place of x_H+ x_O
  const Net::Substep sub = Net::Prepare(y, e, c);
  const Real x_e_sum = y[Net::IH_PLUS] + sub.x_cii + y[3] + 2.0*y[4] +
                       chemistry2::NCR::XeMol(y[Net::IE], y[Net::IH2], c.base, sub.tc);
  std::printf("  x_e %.6g, sum of carriers %.6g\n", y[Net::IE], x_e_sum);
  report.ExpectNear(y[Net::IE], x_e_sum, 2e-2, 0.0, "electrons = sum of carriers");
  return report.ExitCode();
}
