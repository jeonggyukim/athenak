#ifndef CHEMISTRY2_UTIL_ROW_TABLE_HPP_
#define CHEMISTRY2_UTIL_ROW_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file row_table.hpp
//! \brief Coefficients that depend on temperature alone, tabulated in rows of one
//!        temperature grid: every coefficient of a chemical network at one grid
//!        temperature sits in one row (T-major). One Locate(T) gives the two rows and the
//!        weight; each coefficient is then two loads and a linear interpolation.
//!
//!   The grid is uniform in nqt1_log(T), a piecewise-linear approximation of log2(T)
//!   (Hammond et al. 2025, ApJS 277, 65): Locate costs bit operations, no logarithm.
//!   Ported from Tigris src/photchem/rates/row_table.hpp.

#include <cstdint>
#include <cstring>

#include "athena.hpp"

namespace chemistry2 {

//! nqt1_log(x): the IEEE-754 bits of x minus those of 1.0, over 2^52 (x > 0)
KOKKOS_INLINE_FUNCTION Real NqtLog(const double x) {
  std::int64_t b;
  std::memcpy(&b, &x, sizeof(double));
  b -= 4607182418800017408LL;  // bits of 1.0
  return static_cast<double>(b)/4503599627370496.0;
}

//! Inverse of NqtLog
KOKKOS_INLINE_FUNCTION Real NqtExp(const double y) {
  const std::int64_t b = static_cast<std::int64_t>(y*4503599627370496.0) +
                         4607182418800017408LL;
  double x;
  std::memcpy(&x, &b, sizeof(double));
  return x;
}

//! The two rows that bracket a temperature and the linear weight of the upper one
struct Slot {
  int i0;
  int i1;
  Real w;
};

//! Temperature grid, rows uniform in NqtLog(T) from t_min to t_max
struct RowGrid {
  Real t_min, nqt_t_min, nqt_idt;
  int n_rows;

  static RowGrid Make(const Real t_min, const Real t_max, const int n_rows) {
    RowGrid g;
    g.t_min = t_min;
    g.n_rows = n_rows;
    g.nqt_t_min = NqtLog(t_min);
    g.nqt_idt = (n_rows - 1)/(NqtLog(t_max) - g.nqt_t_min);
    return g;
  }
  //! Rows and weight for T; T below t_min reads row 0, above t_max the top pair
  KOKKOS_INLINE_FUNCTION Slot Locate(const Real T) const {
    const Real u = (NqtLog(Kokkos::fmax(T, t_min)) - nqt_t_min)*nqt_idt;
    const Real uc = Kokkos::fmin(Kokkos::fmax(u, 0.0),
                                 static_cast<Real>(n_rows - 1) - 1.0e-9);
    const int i0 = static_cast<int>(uc);
    return Slot{i0, i0 + 1, uc - static_cast<Real>(i0)};
  }
  //! As Locate(T), and dw/dT: NqtLog has slope 2^-e in the octave [2^e, 2^(e+1)), and
  //! 2^e is T with its mantissa bits cleared; zero where T is clamped
  KOKKOS_INLINE_FUNCTION Slot Locate(const Real T, Real *dw_dT) const {
    const Slot s = Locate(T);
    const Real u = (NqtLog(Kokkos::fmax(T, t_min)) - nqt_t_min)*nqt_idt;
    std::int64_t b;
    std::memcpy(&b, &T, sizeof(double));
    b &= 0x7FF0000000000000LL;
    double octave;
    std::memcpy(&octave, &b, sizeof(double));
    *dw_dT = (T > t_min && u < static_cast<Real>(n_rows - 1)) ? nqt_idt/octave : 0.0;
    return s;
  }
  Real RowTemperature(const int i) const { return NqtExp(nqt_t_min + i/nqt_idt); }
};

//! Coefficients by row and column, contiguous by row, on the device
struct RowTable {
  RowGrid grid;
  DvceArray2D<Real> data;

  //! Linear interpolation of column c between the two rows of s
  KOKKOS_INLINE_FUNCTION Real At(const Slot &s, const int c) const {
    const Real v0 = data(s.i0, c);
    return v0 + s.w*(data(s.i1, c) - v0);
  }
  //! Upper row minus lower row of column c: dAt/dw
  KOKKOS_INLINE_FUNCTION Real Diff(const Slot &s, const int c) const {
    return data(s.i1, c) - data(s.i0, c);
  }
};

//! Build a table on the host: fill(T, row) writes the n_cols values at temperature T;
//! then copy to the device.
template <class F>
RowTable BuildRowTable(const RowGrid &grid, const int n_cols, F fill) {
  RowTable t;
  t.grid = grid;
  t.data = DvceArray2D<Real>("chemistry2_row_table", grid.n_rows, n_cols);
  auto h = Kokkos::create_mirror_view(t.data);
  for (int i = 0; i < grid.n_rows; ++i) fill(grid.RowTemperature(i), &h(i, 0));
  Kokkos::deep_copy(t.data, h);
  return t;
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_UTIL_ROW_TABLE_HPP_
