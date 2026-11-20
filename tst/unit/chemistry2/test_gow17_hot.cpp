//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_gow17_hot.cpp
//! \brief GOW17 in hot gas: the blend of cooling, heating and grain recombination with
//!        the hot-gas weight w, the CIE electrons, and the dual derivative, checked
//!        term by term against the CIE table file and against the run with the hot
//!        branch off.

#include <cmath>
#include <cstdio>
#include <string>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/gow17.hpp"

namespace {
using Net = chemistry2::GOW17;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  namespace hot = chemistry2::hot;
  namespace g17 = chemistry2::gow17;
  std::string root(__FILE__);
  root = root.substr(0, root.find_last_of('/')) + "/../../../";
  const hot::CIESource src =
      hot::ReadCIESource(root + "inputs/chemistry2/cie/tigress_coolftn_ncr.txt");
  const Real inf = Kokkos::Experimental::infinity_v<Real>;
  const Real t0 = 2.0e4, t1 = 3.5e4;

  Net::Settings off{1.0, 0.1, 1.6e-4, 3.2e-4, 1.7e-6, 1.0, inf, inf, 1.0, 1.0e9,
                    3.0e20, true, false};
  off.he_coll_ion = false;
  off.table = g17::BuildGOW17Table(false, g17::ROWS_PER_OCTAVE, nullptr, &src);
  Net::Settings on = off;
  on.hot = true;
  on.temp_hot0 = t0;
  on.temp_hot1 = t1;

  const Real n_h = 0.3;
  const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  const chemistry2::CellInputs in{1.0, 2.0e-16, 0.0};
  const Net::Cell c_off = Net::MakeCell(1.0, GAMMA, off, uf, in);
  const Net::Cell c_on = Net::MakeCell(1.0, GAMMA, on, uf, in);

  // ionized state: H+ 1 - 1e-6, He+ 0.1 (He I 0), C+, O+, Si+ at their budgets
  Real y[Net::NSPEC] = {};
  y[Net::IH_PLUS] = 1.0 - 1.0e-6;
  y[Net::IHE_PLUS] = 0.1;
  y[Net::IC_PLUS] = 1.6e-4;
  y[Net::IO_PLUS] = 3.2e-4;
  y[Net::ISI_PLUS] = 1.7e-6;
  y[Net::IH2] = 1.0e-12;
  const Real x_e0 = 1.0 - 1.0e-6 + 0.1 + 1.6e-4 + 3.2e-4 + 1.7e-6;

  // energy for a target T with the hot electrons: iterate e(T, x_e)
  auto energy_at = [&](const Net::Cell &c, const Real T) {
    Real e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, x_e0, GAMMA)*T/U_ENERGY;
    for (int it = 0; it < 20; ++it) {
      const Net::Substep s = Net::Prepare(y, e, c);
      e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, s.g.e, GAMMA)*T/U_ENERGY;
    }
    return e;
  };

  // below T0: on and off agree exactly
  {
    double worst = 0.0;
    for (const Real T : {50.0, 3.0e3, 1.5e4, 1.99e4}) {
      Real yc[Net::NSPEC] = {};
      yc[Net::IH2] = 0.1;
      yc[Net::IH_PLUS] = 0.01;
      yc[Net::IC_PLUS] = 1.6e-4;
      const Real e = n_h*chemistry2::Thermo::CvCold(0.1, 0.1, 0.01016, GAMMA)*T/U_ENERGY;
      const Real a = Net::Edot(yc, e, c_off, Net::Prepare(yc, e, c_off));
      const Real b = Net::Edot(yc, e, c_on, Net::Prepare(yc, e, c_on));
      worst = std::fmax(worst, std::fabs(a - b));
    }
    report.Expect(worst == 0.0, "below T0 the hot branch changes nothing");
  }

  // in the window and above it: electrons, grain recombination, heating, CIE term
  for (const Real T : {2.6e4, 1.0e5, 1.0e6}) {
    const Real e = energy_at(c_on, T);
    const Net::Substep s_on = Net::Prepare(y, e, c_on);
    const Real T_on = Net::RawTemperature(y, e, s_on.g.e, c_on);
    const Real w = hot::Weight<Real>(T_on, t0, t1);
    const Real xe_he = src.Eval(hot::XE_HE, T_on), xe_met = src.Eval(hot::XE_METAL, T_on);
    const Real want_e = x_e0 + w*(std::fmax(0.0, xe_he - 0.1) +
                                  std::fmax(0.0, xe_met - (1.6e-4 + 3.2e-4 + 1.7e-6)));
    std::printf("  T %.3g K: w %.4f, x_e %.6f (tracked %.6f, want %.6f)\n", T_on, w,
                s_on.g.e, x_e0, want_e);
    // row table against the source: 4.9e-5 measured at 1e5 K
    report.ExpectNear(s_on.g.e, want_e, 1e-4, 0.0, "x_e = tracked + w CIE extra, T " +
                      std::to_string(T));
    // grain recombination with the same electrons and T, before the hot weight
    {
      const chemistry2::Slot sl = on.table.grid.Locate(s_on.T);
      const g17::GOW17TemperatureRates tr =
          g17::LookupRates(on.table, sl, s_on.T, s_on.T);
      const g17::GOW17Rates r0 = g17::MakeGOW17Rates(tr, y[Net::IH2], s_on.g.h, s_on.g.e,
                                                     n_h, 1.0, 2.0e-16, 1.0);
      report.ExpectNear(s_on.r.kgr[g17::IGR_HP], (1.0 - w)*r0.kgr[g17::IGR_HP], 1e-12,
                        1e-300, "grain recombination x (1 - w), T " + std::to_string(T));
    }
    // the same e and electrons with the hot branch off, for the cold terms
    Net::Substep s_off = Net::Prepare(y, e, c_off);
    s_off.g.e = s_on.g.e;
    const auto a = Net::HeatCoolTable<Real>(y, T_on, 1.0, c_off, s_off);
    const auto b = Net::HeatCoolTable<Real>(y, T_on, 1.0, c_on, s_on);
    report.ExpectNear(b.heating, (1.0 - w)*a.heating, 1e-12, 1e-300,
                      "heating x (1 - w), T " + std::to_string(T));
    // CIE part: with all cold cooling scaled, b - (1 - w) a = w (cie + cooling_h)
    const Real cie = n_h*(src.Eval(hot::LAMBDA_HE, T_on) +
                          src.Eval(hot::LAMBDA_METAL, T_on));
    std::printf("    cooling off %.4e, on %.4e, w n_H Lambda_CIE %.4e\n", a.cooling,
                b.cooling, w*cie);
    report.Expect(b.cooling >= (1.0 - w)*a.cooling + w*cie*(1.0 - 3e-4),
                  "cooling >= (1 - w) cold + w CIE, T " + std::to_string(T));
  }

  // dual derivative of net cooling against a centered difference, mid-row
  {
    double worst = 0.0;
    for (const Real T0 : {2.2e4, 2.75e4, 3.3e4, 1.0e5, 1.0e6}) {
      const chemistry2::Slot sl = on.table.grid.Locate(T0);
      const Real T = 0.5*(on.table.grid.RowTemperature(sl.i0) +
                          on.table.grid.RowTemperature(sl.i1));
      const Real e = energy_at(c_on, T);
      const Net::Substep s = Net::Prepare(y, e, c_on);
      const auto hd = Net::HeatCoolTable<chemistry2::dual::Dual>(y, T, 1.0, c_on, s);
      auto net = [&](const Real t) {
        const auto r = Net::HeatCoolTable<Real>(y, t, 1.0, c_on, s);
        return r.cooling - r.heating;
      };
      const Real h = 1e-7*T;
      const Real fd = (net(T + h) - net(T - h))/(2*h);
      const double r = std::fabs((hd.cooling.d - hd.heating.d) - fd)/std::fabs(fd);
      std::printf("  T %.3g K: d(net cooling)/dT dual %.6e, FD %.6e, rel %.2e\n", T,
                  hd.cooling.d - hd.heating.d, fd, r);
      worst = std::fmax(worst, r);
    }
    report.ExpectNear(worst, 0.0, 0.0, 3e-3, "d(net cooling)/dT (dual) vs FD");
  }
  return report.ExitCode();
}
