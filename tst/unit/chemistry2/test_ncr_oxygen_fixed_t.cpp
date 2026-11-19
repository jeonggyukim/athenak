//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ncr_oxygen_fixed_t.cpp
//! \brief NCR_oxygen (O+, O++) in one zone at fixed T (species updates only, e set to
//!        T), n_H 1 cm^-3, no radiation, no grains, run to equilibrium: x_H+ and the
//!        oxygen fractions at log T 4.0-4.3, for comparison with Cloudy 25.00 coronal
//!        equilibrium (charge transfer on, molecules off; results note
//!        chemistry2-p6-ncr-oxygen-vs-cloudy.md). Reported, not checked: NCR uses case B
//!        recombination for H+, Cloudy case A. Checked: O+/O follows H+/H through charge
//!        transfer (O+/O within 30% of (k_ct,ion/k_ct,rec) H+/H I; measured 4e-4).

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
  // Z_d = 1e-10: grain-assisted recombination off, as in the Cloudy model (no grains);
  // NCR divides by Z_d (x_e,mol, the CO critical density), so not 0
  Net::Settings st{{1.0e-10, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20,
                    true, false, 10.0, 0.0}, chemistry2::RowTable(),
                   chemistry2::ions::IonCoolingTable()};
  st.base.table = chemistry2::ncr::BuildNCRTable();
  st.base.case_a = (argc > 1 && std::string(argv[1]) == "case_a");  // as Cloudy CIE
  const chemistry2::ions::IonRateData data(dir);
  st.ions = chemistry2::ions::BuildIonTable(data, st.base.table.grid, 8, 2);
  st.cool = chemistry2::ions::BuildIonCoolingTable(st.base.table.grid,
                                                   dir + "ion_cooling/", {"o_2", "o_3"});
  const Real n_h = 1.0;
  const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  const Net::Cell c = Net::MakeCell(1.0, GAMMA, st, uf,
                                    chemistry2::CellInputs{1.0e-20, 1.0e-30, 0.0});
  const chemistry2::SemiImplicitSettings si{0.1, 1.0e-12, 100000, true};
  double worst_ct = 0.0;
  for (const Real lt : {4.0, 4.1, 4.2, 4.3}) {
    const Real T = std::pow(10.0, lt);
    Real y[Net::NSPEC] = {};
    y[Net::IH2] = 1e-10;
    y[Net::IH_PLUS] = 0.5;
    y[Net::IE] = 0.5;
    // the species updates alone, at e set to T each step (no energy update): 2000 steps
    // limited by MaxStep, up to 0.1 Myr each
    for (int step = 0; step < 2000; ++step) {
      const Real e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, y[Net::IE], GAMMA)*
                     T/U_ENERGY;
      const Net::Substep sub = Net::Prepare(y, e, c);
      const Real h = Kokkos::fmin(0.1, Net::MaxStep(si.dt_sub_frac, y, e, c, sub));
      Real y_n[Net::NSPEC];
      for (int k = 0; k < Net::NSPEC; ++k) y_n[k] = y[k];
      Net::Update<Net::IH2>(y, y_n, e, c, sub, h);
      Net::Update<Net::IO1>(y, y_n, e, c, sub, h);
    }
    const Real x_hi = 1.0 - 2.0*y[Net::IH2] - y[Net::IH_PLUS];
    const Real x_oi = c.x_o - y[3] - y[4];
    std::printf("  log T %.1f: x_H+ %.3e; O I %.3e O II %.3e O III %.3e (per O); H I "
                "collisional ionization NCR/Voronov %.3f\n", lt, y[Net::IH_PLUS],
                x_oi/c.x_o, y[3]/c.x_o, y[4]/c.x_o,
                chemistry2::ncr::CollIonHI(T)/data.ci.Rate(1, 1, T));
    const double lock = (data.ct.CtIon(8, 8, T)/data.ct.CtRec(8, 7, T))*
                        y[Net::IH_PLUS]/x_hi;
    worst_ct = std::fmax(worst_ct, std::fabs((y[3]/x_oi)/lock - 1.0));
  }
  std::printf("  O+/O against charge-transfer locking: worst %.3f\n", worst_ct);
  report.ExpectNear(worst_ct, 0.0, 0.0, 0.3, "O+/O follows H+/H through charge transfer");
  return report.ExitCode();
}
