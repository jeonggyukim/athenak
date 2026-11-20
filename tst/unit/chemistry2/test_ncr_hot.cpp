//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_ncr_hot.cpp
//! \brief NCR in hot gas: the hot-gas weight w (smootherstep on [temp_hot0, temp_hot1]),
//!        the CIE table read, and the blend in cooling, heating and electrons, checked
//!        term by term against values taken from the table file directly.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/network/ncr/ncr.hpp"

namespace {
using Net = chemistry2::NCR;
constexpr Real MU_H = 1.4, GAMMA = 5.0/3.0, U_TIME = 3.15576e13, U_ENERGY = 1.0e-12;

//! The source rows of the table: log10 T, Lambda_He, Lambda_metal, xe_He, xe_metal
std::vector<std::vector<double>> SourceRows(const std::string &path) {
  std::ifstream f(path);
  std::vector<std::vector<double>> rows;
  std::string line;
  int header = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (header++ < 2) continue;
    std::istringstream is(line);
    std::vector<double> v(10);
    for (double &x : v) is >> x;
    rows.push_back({v[0], v[1], v[2], v[3], v[4], v[5]});
  }
  return rows;
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;
  std::string root(__FILE__);
  root = root.substr(0, root.find_last_of('/')) + "/../../../";
  const std::string path = root + "inputs/chemistry2/cie/tigress_coolftn_ncr.txt";
  const Real t0 = 2.0e4, t1 = 3.5e4;

  // weight: endpoints, midpoint, derivative against a centered difference
  {
    namespace h = chemistry2::hot;
    report.Expect(h::Weight<Real>(t0, t0, t1) == 0.0 && h::Weight<Real>(t1, t0, t1) == 1.0
                  && h::Weight<Real>(1.0e4, t0, t1) == 0.0
                  && h::Weight<Real>(1.0e6, t0, t1) == 1.0, "w = 0 at and below T0, 1 at "
                  "and above T1");
    report.ExpectNear(h::Weight<Real>(0.5*(t0 + t1), t0, t1), 0.5, 1e-15, 0.0,
                      "w = 1/2 at the window centre");
    double worst = 0.0;
    for (const Real T : {2.1e4, 2.5e4, 2.75e4, 3.0e4, 3.4e4}) {
      const Real d = 1e-3;
      const Real fd = (h::Weight<Real>(T + d, t0, t1) - h::Weight<Real>(T - d, t0, t1))/
                      (2*d);
      const auto wd = h::Weight<chemistry2::dual::Dual>(chemistry2::dual::Dual(T, 1.0),
                                                       t0, t1);
      worst = std::fmax(worst, std::fabs(wd.d - fd)/std::fabs(fd));
    }
    report.ExpectNear(worst, 0.0, 0.0, 1e-7, "dw/dT (dual) vs centered difference");
  }

  // table read: the host source and its device copy return the source values at the
  // source temperatures
  namespace hot = chemistry2::hot;
  const hot::CIESource src = hot::ReadCIESource(path);
  const hot::CIETable cie = hot::MakeCIETable(src);
  const auto rows = SourceRows(path);
  {
    double worst = 0.0;
    for (std::size_t i = 0; i < rows.size(); i += 7) {
      const Real T = std::pow(10.0, rows[i][0]);
      Real v[hot::NCOL];
      cie.Eval(T, v);
      for (int k = 0; k < hot::NCOL; ++k) {
        worst = std::fmax(worst, std::fabs(v[k]/rows[i][k + 1] - 1.0));
        worst = std::fmax(worst, std::fabs(src.Eval(k, T)/rows[i][k + 1] - 1.0));
      }
    }
    std::printf("  table at source points: worst %.2e\n", worst);
    report.ExpectNear(worst, 0.0, 0.0, 1e-10, "CIE table at source points");
  }

  // the blend, term by term, in one state at a source temperature inside the window
  // (row 20: log T 4.40) and one above it (row 40: log T 4.80)
  Net::Settings base{1.0, 1.0, 0.1, 3.0e-17, true, true, t1, 5.7e-11, 3.086e20, true,
                     false, 10.0, 0.0};
  base.table = chemistry2::ncr::BuildNCRTable();
  base.temp_hot0 = t0;
  Net::Settings no_cie = base;           // blend, no CIE terms
  Net::Settings with_cie = base;         // blend and CIE terms (row-table columns)
  with_cie.table = chemistry2::ncr::BuildNCRTable(chemistry2::gow17::ROWS_PER_OCTAVE,
                                                  &src);
  Net::Settings cold = base;             // no blend in the window (T0 above it)
  cold.temp_hot0 = 1.0e9;
  cold.temp_hot1 = 2.0e9;
  const Real n_h = 0.3;
  const chemistry2::UnitFactors uf = chemistry2::MakeUnitFactors(
      n_h*MU_H*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, MU_H);
  const chemistry2::CellInputs in{1.0, 2.0e-16, 0.0};
  for (const int row : {20, 40}) {
    const Real T = std::pow(10.0, rows[row][0]);
    Real y[Net::NSPEC];
    y[Net::IH2] = 1.0e-8;
    y[Net::IH_PLUS] = 0.6;
    y[Net::IE] = 0.62;
    const Real e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, y[Net::IE], GAMMA)*T/
                   U_ENERGY;
    auto eval = [&](const Net::Settings &s) {
      const Net::Cell c = Net::MakeCell(1.0, GAMMA, s, uf, in);
      return Net::HeatingCooling(y, e, c, Net::Prepare(y, e, c));
    };
    const Real w = chemistry2::hot::Weight<Real>(Net::Temperature(y, e,
                       Net::MakeCell(1.0, GAMMA, base, uf, in)), t0, t1);
    const auto a = eval(cold), b = eval(no_cie), c = eval(with_cie);
    // other = a - (hydrogen part); hydrogen part = b at w = 1, so use b = H + (1-w) O
    const Real cie_direct = n_h*(rows[row][1 + hot::LAMBDA_HE] +
                                 rows[row][1 + hot::LAMBDA_METAL]);
    std::printf("  log T %.2f: w %.4f; cooling cold %.4e, blend %.4e, with CIE %.4e "
                "(CIE term %.4e); heating cold %.4e, blend %.4e\n", rows[row][0], w,
                a.cooling, b.cooling, c.cooling, cie_direct, a.heating, b.heating);
    // row table, linear in Lambda between rows: measured 1.9e-4 at log T 4.8
    report.ExpectNear(c.cooling - b.cooling, w*cie_direct, 3e-4, 0.0,
                      "CIE term = w n_H (Lambda_He + Z_g Lambda_metal), log T " +
                      std::to_string(rows[row][0]));
    report.ExpectNear(b.heating, (1.0 - w)*a.heating, 1e-12, 1e-300,
                      "heating x (1 - w), log T " + std::to_string(rows[row][0]));
    if (row == 40) {
      report.Expect(w == 1.0, "w = 1 above T1");
    }
  }

  // electrons above T1: x_H+ + xe_He + Z_g xe_metal
  {
    const Real T = std::pow(10.0, rows[60][0]);
    const Net::Cell c = Net::MakeCell(1.0, GAMMA, with_cie, uf, in);
    chemistry2::Slot slot;
    const Net::TCoef tc = Net::Coefficients(T, c, &slot);
    const Real x_e = Net::FreeElectrons(1.1, 1e-8, 0.999, T, c, tc);
    const Real x_e_ref = 0.999 + rows[60][1 + hot::XE_HE] + rows[60][1 + hot::XE_METAL];
    report.ExpectNear(x_e, x_e_ref, 1e-5, 0.0,  // 7.4e-6
                      "x_e above T1 = x_H+ + xe_He + xe_metal");
  }

  // below T0 the CIE table changes nothing
  {
    double worst = 0.0;
    for (const Real T : {50.0, 800.0, 8.0e3, 1.9e4}) {
      Real y[Net::NSPEC] = {0.2, 0.01, 0.0102};
      const Real e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, y[Net::IE], GAMMA)*
                     T/U_ENERGY;
      const Net::Cell cb = Net::MakeCell(1.0, GAMMA, no_cie, uf, in);
      const Net::Cell cc = Net::MakeCell(1.0, GAMMA, with_cie, uf, in);
      const Real eb = Net::Edot(y, e, cb, Net::Prepare(y, e, cb));
      const Real ec = Net::Edot(y, e, cc, Net::Prepare(y, e, cc));
      worst = std::fmax(worst, std::fabs(eb - ec));
    }
    report.Expect(worst == 0.0, "below T0 the CIE table changes nothing");
  }

  // d(cooling - heating)/dT with dual numbers against a centered difference of the same
  // function, in ionized gas (x_H I = 1e-6, so the H I collisional ionization, whose
  // value is fixed per substep, does not matter); T away from row edges
  {
    namespace d = chemistry2::dual;
    const Net::Cell c = Net::MakeCell(1.0, GAMMA, with_cie, uf, in);
    double worst = 0.0;
    for (const Real T0 : {2.05e4, 2.2e4, 2.75e4, 3.3e4, 3.45e4, 5.0e4, 3.0e5, 3.0e6}) {
      const chemistry2::Slot s0 = c.s.table.grid.Locate(T0);
      const Real T = c.s.table.grid.RowTemperature(s0.i0)*0.5 +
                     c.s.table.grid.RowTemperature(s0.i1)*0.5;
      Real y[Net::NSPEC] = {1e-12, 1.0 - 1e-6, 1.2};
      const Real e = n_h*chemistry2::Thermo::CvCold(y[Net::IH2], 0.1, y[Net::IE], GAMMA)*
                     T/U_ENERGY;
      const Net::Substep sub = Net::Prepare(y, e, c);
      const auto hd = Net::HeatCoolTable<d::Dual>(y, T, 1.0, c, sub);
      auto net = [&](const Real t) {
        const auto r = Net::HeatCoolTable<Real>(y, t, 1.0, c, sub);
        return r.cooling - r.heating;
      };
      const Real h = 1e-7*T;
      const Real fd = (net(T + h) - net(T - h))/(2*h);
      const double r = std::fabs((hd.cooling.d - hd.heating.d) - fd)/std::fabs(fd);
      std::printf("  T %.3g K: d(net cooling)/dT dual %.6e, FD %.6e, rel %.2e\n", T,
                  hd.cooling.d - hd.heating.d, fd, r);
      worst = std::fmax(worst, r);
    }
    // measured 1.8e-3 at 3e6 K, from the H I collisional ionization term (value fixed
    // per substep, derivative from the table); <= 1.3e-4 below 5e4 K
    report.ExpectNear(worst, 0.0, 0.0, 3e-3,
                      "d(net cooling)/dT (dual) vs centered difference");
  }
  return report.ExitCode();
}
