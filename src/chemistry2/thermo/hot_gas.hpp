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
//!
//!        CIESource holds the table on the host; a chemical network with a row table
//!        evaluates it once per row at build time. CIETable is the device copy, for
//!        paths without a row table.

#include <algorithm>
#include <array>
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

enum : int {LAMBDA_HE = 0, LAMBDA_METAL, XE_HE, XE_METAL, NCOL};

//! w(T) on [t0, t1]; with S = dual::Dual the derivative part carries dw/dT
template <class S>
KOKKOS_INLINE_FUNCTION S Weight(const S &T, const Real t0, const Real t1) {
  const Real tv = dual::Value(T);
  if (tv <= t0) return S(0.0);
  if (tv >= t1) return S(1.0);
  const S s = (T - t0)*(1.0/(t1 - t0));
  return s*s*s*(s*(6.0*s - 15.0) + 10.0);
}

//! The table on the host: ln of the NCOL columns, uniform in log10 T
struct CIESource {
  std::vector<std::array<Real, NCOL>> ln;
  Real lt0 = 0.0, inv_dlt = 0.0;

  //! Column k at T [K], linear in log10 T of its logarithm, held at the end rows
  Real Eval(const int k, const Real T) const {
    const int n = static_cast<int>(ln.size());
    const Real u = std::fmin(std::fmax((std::log10(T) - lt0)*inv_dlt, 0.0),
                             static_cast<Real>(n - 1));
    const int i0 = std::min(static_cast<int>(u), n - 2);
    const Real f = u - static_cast<Real>(i0);
    return std::exp(ln[i0][k] + f*(ln[i0 + 1][k] - ln[i0][k]));
  }
};

inline CIESource ReadCIESource(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("hot::ReadCIESource: cannot open " + path);
  CIESource src;
  std::vector<Real> lt;
  std::string line;
  int header = 0;  // "nvar nx1", then "logT_min logT_max"
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (header < 2) {
      ++header;
      continue;
    }
    std::istringstream is(line);
    Real v[6];
    for (Real &x : v) {
      if (!(is >> x)) {
        throw std::runtime_error("hot::ReadCIESource: short row in " + path);
      }
    }
    lt.push_back(v[0]);
    src.ln.push_back({std::log(std::fmax(v[2], 1.0e-300)),
                      std::log(std::fmax(v[3], 1.0e-300)),
                      std::log(std::fmax(v[4], 1.0e-300)),
                      std::log(std::fmax(v[5], 1.0e-300))});
  }
  if (lt.size() < 2) throw std::runtime_error("hot::ReadCIESource: no rows in " + path);
  src.lt0 = lt[0];
  src.inv_dlt = 1.0/(lt[1] - lt[0]);
  return src;
}

//! Device copy of a CIESource
struct CIETable {
  DvceArray2D<Real> ln;  // (n, NCOL)
  Real lt0 = 0.0, inv_dlt = 0.0;
  int n = 0;

  //! The NCOL values at T [K] (S = Real or dual::Dual), as CIESource::Eval
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

inline CIETable MakeCIETable(const CIESource &src) {
  CIETable t;
  t.n = static_cast<int>(src.ln.size());
  t.lt0 = src.lt0;
  t.inv_dlt = src.inv_dlt;
  t.ln = DvceArray2D<Real>("chemistry2_hot_cie", t.n, NCOL);
  auto h = Kokkos::create_mirror_view(t.ln);
  for (int i = 0; i < t.n; ++i) {
    for (int k = 0; k < NCOL; ++k) h(i, k) = src.ln[i][k];
  }
  Kokkos::deep_copy(t.ln, h);
  return t;
}

}  // namespace hot
}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_HOT_GAS_HPP_
