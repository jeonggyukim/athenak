#ifndef CHEMISTRY2_IONS_ION_COOLING_HPP_
#define CHEMISTRY2_IONS_ION_COOLING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion_cooling.hpp
//! \brief Line cooling of tracked metal ions, ported from Tigris photchem
//!        (gow17_ion_cooling.hpp): Lambda_q(T, n_e) from CHIANTI 11 (collisional
//!        excitation; inputs/chemistry2/ions/ion_cooling/lambda_<ion>.txt), line cooling
//!        per volume n_e n_q Lambda_q. Stored as ln Lambda on the rows of a RowGrid
//!        times a log n_e axis (17 columns, -2 to 6 in steps of 0.5), ions innermost.

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "chemistry2/util/row_table.hpp"

namespace chemistry2 {
namespace ions {

//! Cooling per ion on a (log T, log n_e) grid, on the device
struct IonCoolingTable {
  static constexpr int N_NE = 17;
  static constexpr Real LOGNE_MIN = -2.0;
  static constexpr Real INV_DLOGNE = 2.0;
  static constexpr int MAX_ION = 16;
  DvceArray2D<Real> data;  // (n_rows, N_NE*nion), ln Lambda [erg cm^3 s^-1]
  int nion = 0;

  //! ln Lambda of ion k at slot s and n_e [cm^-3] (clamped to the axis), and its change
  //! per unit table weight w (for the temperature derivative)
  KOKKOS_INLINE_FUNCTION void LnLambda(const Slot &s, const Real n_e, const int k,
                                       Real *ln_l, Real *dln_l_dw) const {
    Real u = (Kokkos::log10(Kokkos::fmax(n_e, 1.0e-30)) - LOGNE_MIN)*INV_DLOGNE;
    u = Kokkos::fmin(Kokkos::fmax(u, 0.0), static_cast<Real>(N_NE - 1) - 1.0e-9);
    const int j0 = static_cast<int>(u);
    const Real wn = u - static_cast<Real>(j0);
    const Real a0 = data(s.i0, j0*nion + k), a1 = data(s.i0, (j0 + 1)*nion + k);
    const Real b0 = data(s.i1, j0*nion + k), b1 = data(s.i1, (j0 + 1)*nion + k);
    const Real lo = a0 + s.w*(b0 - a0), hi = a1 + s.w*(b1 - a1);
    *ln_l = lo + wn*(hi - lo);
    *dln_l_dw = (b0 - a0) + wn*((b1 - a1) - (b0 - a0));
  }
};

namespace ion_cooling_detail {
// one ion's cooling file as read on the host
struct SourceTable {
  std::vector<Real> log_t;
  std::vector<std::vector<Real>> log_l;  // [row][n_e column]
};

// read log T and log Lambda at each n_e column from one file
inline SourceTable Read(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("IonCoolingTable: cannot open " + path);
  SourceTable t;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("# logT_grid:", 0) == 0) {  // the log10 T of the rows
      std::istringstream is(line.substr(line.find(':') + 1));
      Real v;
      while (is >> v) t.log_t.push_back(v);
      continue;
    }
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::vector<Real> row;
    Real v;
    while (is >> v) row.push_back(v);
    if (static_cast<int>(row.size()) != IonCoolingTable::N_NE) {
      throw std::runtime_error("IonCoolingTable: " + path + " is not on the expected "
                               "grid (17 log n_e columns)");
    }
    t.log_l.push_back(row);
  }
  if (t.log_t.size() < 2 || t.log_t.size() != t.log_l.size()) {
    throw std::runtime_error("IonCoolingTable: " + path + ": rows and logT_grid differ");
  }
  return t;
}
}  // namespace ion_cooling_detail

//! The ions in names (CHIANTI names, e.g. "o_2"), from dir, at the temperatures of the
//! rows of grid: linear in log T between the source rows, held at the end rows outside
inline IonCoolingTable BuildIonCoolingTable(const RowGrid &grid, const std::string &dir,
                                            const std::vector<std::string> &names) {
  IonCoolingTable tab;
  tab.nion = static_cast<int>(names.size());
  if (tab.nion > IonCoolingTable::MAX_ION) {
    throw std::runtime_error("IonCoolingTable: more ions than MAX_ION");
  }
  constexpr int NNE = IonCoolingTable::N_NE;
  tab.data = DvceArray2D<Real>("chemistry2_ion_cooling", grid.n_rows, NNE*tab.nion);
  auto h = Kokkos::create_mirror_view(tab.data);
  constexpr Real LN10 = 2.302585092994045684;
  for (int k = 0; k < tab.nion; ++k) {
    const ion_cooling_detail::SourceTable src =
        ion_cooling_detail::Read(dir + "lambda_" + names[k] + ".txt");
    const int nsrc = static_cast<int>(src.log_t.size());
    const Real lt0 = src.log_t.front(), dlt = src.log_t[1] - src.log_t[0];
    for (int i = 0; i < grid.n_rows; ++i) {
      const Real lt = std::log10(grid.RowTemperature(i));
      Real u = (lt - lt0)/dlt;
      u = std::fmin(std::fmax(u, 0.0), static_cast<Real>(nsrc - 1) - 1.0e-9);
      const int r0 = static_cast<int>(u);
      const Real w = u - static_cast<Real>(r0);
      for (int j = 0; j < NNE; ++j) {
        const Real a = src.log_l[r0][j], b = src.log_l[r0 + 1][j];
        h(i, j*tab.nion + k) = LN10*(a + w*(b - a));
      }
    }
  }
  Kokkos::deep_copy(tab.data, h);
  return tab;
}

}  // namespace ions
}  // namespace chemistry2

#endif  // CHEMISTRY2_IONS_ION_COOLING_HPP_
