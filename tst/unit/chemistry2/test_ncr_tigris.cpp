//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ncr_tigris.cpp
//! \brief NCR rates and closures against Tigris photchem on 500 random states
//!        (data/tigris_ncr_states.txt, written by data/dump_tigris_ncr.cpp): C and D of
//!        H2 and H+, x_e after the update and x_C+ to 1e-10 relative, x_CO to 1e-10 of
//!        2 x_H2 x_C,tot; heating to 2e-4 and cooling to 2e-2, with the cooling also
//!        reported by Tigris's groups (H I, H II, H2, other).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/ncr.hpp"

namespace {
using Net = chemistry2::NCR;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;

double Rel(double a, double b) {
  const double s = std::fmax(std::fabs(a), std::fabs(b));
  return (s < 1e-300) ? 0.0 : std::fabs(a - b)/s;
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  std::string path(__FILE__);
  path = path.substr(0, path.find_last_of('/')) + "/data/tigris_ncr_states.txt";
  std::ifstream in(path);
  report.Expect(in.good(), "read " + path);
  const Net::Settings st{1.0, 1.0, 0.1, 3.0e-17, true, true, 3.5e4, 5.7e-11, 3.086e20,
                         true, false, 10.0, 0.0};
  const char *names[7] = {"C_H2", "D_H2", "C_H+", "D_H+", "x_e", "x_C+", "x_CO"};
  double worst[7] = {};
  double hc_err[2][3] = {};  // heating, cooling: count over 10%, median, max
  std::vector<double> dh, dc, grp[4];
  int n = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    double n_h, T, x_h2, x_hii, x_e, g, xi, ref[7], heat_ref, cool_ref, grp_ref[4];
    ls >> n_h >> T >> x_h2 >> x_hii >> x_e >> g >> xi;
    for (double &r : ref) ls >> r;
    ls >> heat_ref >> cool_ref;
    for (double &r : grp_ref) ls >> r;
    const chemistry2::UnitFactors u = chemistry2::MakeUnitFactors(
        n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
    const Net::Cell c = Net::MakeCell(1.0, GAMMA, st, u,
                                      chemistry2::CellInputs{g, xi, 0.0});
    Real y[Net::NSPEC];
    y[Net::IH2] = x_h2;
    y[Net::IH_PLUS] = x_hii;
    y[Net::IE] = x_e;
    const Real e = n_h*chemistry2::Thermo::CvCold(x_h2, 0.1, x_e, GAMMA)*T/U_ENERGY;
    const Net::Substep s = Net::Prepare(y, e, c);
    const double got[7] = {s.c_h2, s.d_h2, s.c_hii, s.d_hii,
                           Net::FreeElectrons(x_e, x_h2, x_hii, s.T, c), s.x_cii, s.x_co};
    for (int k = 0; k < 6; ++k) worst[k] = std::fmax(worst[k], Rel(got[k], ref[k]));
    // x_CO carries x_C,tot - x_C+, which cancels where carbon is all C+: compare on the
    // scale 2 x_H2 x_C,tot
    worst[6] = std::fmax(worst[6], std::fabs(got[6] - ref[6])/(2.0*x_h2*1.6e-4 + 1e-300));
    // heating and cooling: the shared GOW17 terms against Tigris NCR, reported only
    // until the NCR-only terms (P5b) are in
    const Net::HeatCool hc = Net::HeatingCooling(y, e, c, s);
    dh.push_back(Rel(hc.heating, heat_ref));
    dc.push_back(Rel(hc.cooling, cool_ref));
    // the cooling by Tigris's groups: H I (Ly-alpha, collisional ionization), H II,
    // H2 (rovibrational, collisional dissociation), other (C II, C I, O I, CO, grains)
    {
      namespace th = chemistry2;
      using Th = th::Thermo;
      const Real x_hi = s.x_hi, nn_hi = n_h*x_hi, nn_h2 = n_h*x_h2, nn_e = n_h*x_e;
      const Real g_hi = (T > 3.0e3 ? Th::CoolingLya(x_hi, nn_e, T) : 0.0) +
                        Th::CoolingHIion(x_hi, x_e, s.k_coll_hi*n_h);
      const Real g_h2 = chemistry2::ncr_terms::CoolingH2Moseley(x_h2, x_hi, n_h, T) +
                        Th::CoolingH2diss(x_hi, x_h2, s.k_h2_h*n_h, s.k_h2_h2*n_h);
      const Real g_hii = chemistry2::ncr_terms::CoolingHII(
          x_e, x_hii, n_h, T, chemistry2::ncr::AlphaBHII(T));
      const Real vth = std::sqrt(2.*units::Units::k_boltzmann_cgs*T/
                                 units::Units::CO_mass_cgs);
      const Real g_oth = Th::CoolingCII(s.x_cii, nn_hi, nn_h2, nn_e, T) +
                         Th::CoolingCI(s.x_ci, nn_hi, nn_h2, nn_e, T) +
                         Th::CoolingOI(s.x_oi, nn_hi, nn_h2, nn_e, T) +
                         (s.x_co > 0 ? Th::CoolingCOR(s.x_co, nn_hi, nn_h2, nn_e, T,
                                                      n_h*s.x_co/(vth/3.086e20)) : 0.0) +
                         Th::CoolingRec(1.0, T, nn_e, g) +
                         chemistry2::ncr_terms::CoolingNebular(x_e, x_hii, n_h, T, 1.0);
      const Real got_g[4] = {g_hi, g_hii, g_h2, g_oth};
      for (int k = 0; k < 4; ++k) {
        if (grp_ref[k] > 1e-3*cool_ref) grp[k].push_back(Rel(got_g[k], grp_ref[k]));
      }
    }
    ++n;
  }
  std::printf("  %d states:", n);
  for (int k = 0; k < 7; ++k) std::printf(" %s %.1e", names[k], worst[k]);
  std::printf("\n");
  for (int q = 0; q < 2; ++q) {
    std::vector<double> &v = q ? dc : dh;
    std::sort(v.begin(), v.end());
    hc_err[q][0] = std::count_if(v.begin(), v.end(), [](double d) { return d > 0.1; });
    hc_err[q][1] = v[v.size()/2];
    hc_err[q][2] = v.back();
    std::printf("  %s vs Tigris NCR: median %.2e, max %.2e, %d of %zu over 10%%\n",
                q ? "cooling" : "heating", hc_err[q][1], hc_err[q][2],
                static_cast<int>(hc_err[q][0]), v.size());
  }
  const char *gname[4] = {"H I", "H II", "H2", "other"};
  for (int k = 0; k < 4; ++k) {
    std::sort(grp[k].begin(), grp[k].end());
    if (grp[k].empty()) continue;
    std::printf("  cooling group %-5s (where > 1e-3 of total): median %.2e, max %.2e, "
                "%zu states\n", gname[k], grp[k][grp[k].size()/2], grp[k].back(),
                grp[k].size());
  }
  report.Expect(n == 500, "500 states read");
  for (int k = 0; k < 7; ++k) {
    report.ExpectNear(worst[k], 0.0, 0.0, 1e-10, std::string(names[k]) + " vs Tigris");
  }
  // measured 1.1e-4 (heating; eV 1.602e-12 vs 1.60218e-12 erg) and 1.26e-2 (cooling,
  // in the C II/C I/O I/CO/grain group), 2026-11-20
  report.ExpectNear(hc_err[0][2], 0.0, 0.0, 2e-4, "heating vs Tigris NCR, worst state");
  report.ExpectNear(hc_err[1][2], 0.0, 0.0, 2e-2, "cooling vs Tigris NCR, worst state");
  return report.ExitCode();
}
