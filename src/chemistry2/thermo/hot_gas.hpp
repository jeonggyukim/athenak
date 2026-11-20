#ifndef CHEMISTRY2_THERMO_HOT_GAS_HPP_
#define CHEMISTRY2_THERMO_HOT_GAS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hot_gas.hpp
//! \brief Hot gas: the weight w(T) that hands non-hydrogen cooling, heating and grain
//!        recombination from the cold-gas terms to collisional ionization equilibrium
//!        (CIE) between temp_hot0 and temp_hot1, and the CIE helium and metal cooling
//!        and electrons from the TIGRESS NCR table (inputs/chemistry2/cie/
//!        tigress_coolftn_ncr.txt: log10 T, Lambda_H, Lambda_He, Lambda_metal
//!        [erg cm^3 s^-1, per n_H^2, metals at Z = 1], xe_He, xe_metal [per H], and
//!        four unused columns, uniform in log10 T).
//!
//!        w is the quintic smootherstep s^3 (6 s^2 - 15 s + 10), s = (T - T0)/(T1 - T0)
//!        clamped to [0, 1]: exactly 0 below T0 and 1 above T1, with first and second
//!        derivatives zero at both ends.

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "chemistry2/rates/dual.hpp"

namespace chemistry2 {
namespace hot {

//! w(T) on [t0, t1]; with S = dual::Dual the derivative part carries dw/dT
template <class S>
KOKKOS_INLINE_FUNCTION S Weight(const S &T, const Real t0, const Real t1) {
  const Real tv = dual::Value(T);
  if (tv <= t0) return S(0.0);
  if (tv >= t1) return S(1.0);
  const S s = (T - t0)*(1.0/(t1 - t0));
  return s*s*s*(s*(6.0*s - 15.0) + 10.0);
}

struct CIETable {
  enum : int {LAMBDA_HE = 0, LAMBDA_METAL, XE_HE, XE_METAL, NCOL};
  DvceArray2D<Real> ln;  // (n, NCOL): ln of each column
  Real lt0 = 0.0, inv_dlt = 0.0;
  int n = 0;

  //! The NCOL values at T [K] (S = Real or dual::Dual), linear in log10 T of their
  //! logarithms, held at the end rows outside the table
  template <class S>
  KOKKOS_INLINE_FUNCTION void Eval(const S &T, S *out) const {
    constexpr Real INV_LN10 = 0.434294481903251828;
    S u = (dual::log(T)*INV_LN10 - lt0)*inv_dlt;
    const Real uv = dual::Value(u);
    int i0;
    if (uv <= 0.0) {
      i0 = 0;
      u = S(0.0);
    } else if (uv >= static_cast<Real>(n - 1)) {
      i0 = n - 2;
      u = S(static_cast<Real>(n - 1));
    } else {
      i0 = static_cast<int>(uv);
    }
    const S f = u - static_cast<Real>(i0);
    for (int k = 0; k < NCOL; ++k) {
      const Real a = ln(i0, k), b = ln(i0 + 1, k);
      out[k] = dual::exp(a + f*(b - a));
    }
  }
};

inline CIETable ReadCIETable(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("hot::ReadCIETable: cannot open " + path);
  std::vector<std::vector<Real>> rows;
  std::string line;
  int header = 0;  // "nvar nx1", then "logT_min logT_max"
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::vector<Real> v;
    Real x;
    while (is >> x) v.push_back(x);
    if (header < 2) {
      ++header;
      continue;
    }
    if (v.size() < 7) throw std::runtime_error("hot::ReadCIETable: short row in " + path);
    rows.push_back(v);
  }
  if (rows.size() < 2) throw std::runtime_error("hot::ReadCIETable: no rows in " + path);
  CIETable t;
  t.n = static_cast<int>(rows.size());
  t.lt0 = rows[0][0];
  t.inv_dlt = 1.0/(rows[1][0] - rows[0][0]);
  t.ln = DvceArray2D<Real>("chemistry2_hot_cie", t.n, CIETable::NCOL);
  auto h = Kokkos::create_mirror_view(t.ln);
  constexpr int SRC[CIETable::NCOL] = {2, 3, 4, 5};  // Lambda_He, Lambda_metal, xe's
  for (int i = 0; i < t.n; ++i) {
    for (int k = 0; k < CIETable::NCOL; ++k) {
      h(i, k) = std::log(std::fmax(rows[i][SRC[k]], 1.0e-300));
    }
  }
  Kokkos::deep_copy(t.ln, h);
  return t;
}

}  // namespace hot
}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_HOT_GAS_HPP_
