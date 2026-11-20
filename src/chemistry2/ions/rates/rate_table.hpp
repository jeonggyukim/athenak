#ifndef CHEMISTRY2_IONS_RATES_RATE_TABLE_HPP_
#define CHEMISTRY2_IONS_RATES_RATE_TABLE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rate_table.hpp
//! \brief Machinery for tabulating a rate coefficient against temperature.
//!
//!   Recombination, collisional ionization and charge transfer are
//!   Maxwellian-averaged two-body rate coefficients, and averaging a cross
//!   section over a thermal distribution leaves a function of temperature
//!   alone.  That holds for every ion, so they all tabulate the same way, and
//!   the NCR rates and the multi-ion rates share what is here rather than
//!   each carrying a copy.
//!
//!   Photoionization is the exception and does not belong here.  Its cross
//!   section is a function of photon energy, so it is averaged over the
//!   spectrum once at startup instead; see
//!   `radiative_properties/band_average.hpp`.
//!
//!   Tables are laid out temperature slow, rate index fastest, so one
//!   temperature index addresses a contiguous run of every rate at that
//!   temperature, which is what a cell sweep reads.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "athena.hpp"

namespace Rates {

//! How a tabulated rate is stored and read back.
//!
//!   kExact evaluates the analytic form and builds nothing, which is the
//!   reference the others are measured against.  kLogLog stores log10(rate) on
//!   a uniform log10(T) grid.  kNqt1 and kNqt2 store the not-quite-
//!   transcendental encodings of Hammond et al. (2025), whose decode is
//!   integer bit operations alone, or those plus one square root.
enum class InterpMode { kExact = 0, kLogLog = 1, kNqt2 = 2, kNqt1 = 3 };

//! Floor applied before taking a logarithm, for rates that are exactly zero
//! below some temperature.  Reading back gives the floor rather than zero,
//! which is harmless where the rate is negligible and keeps the table finite.
//! Below it the table carries no information about the rate.
constexpr Real kRateFloor = 1.0e-100;

//----------------------------------------------------------------------------------------
// Not-quite-transcendental logarithm and exponential, Hammond et al. (2025),
// ApJS 277:65.  memcpy is C++11-safe against strict aliasing and compiles to a
// single move at -O3.

//! \brief Piecewise-linear approximant of log2, by IEEE 754 bit manipulation.
//!   as_int(1.0) = 0x3FF0000000000000 and as_int(2.0) - as_int(1.0) = 2^52.
inline double nqt1_log(double x) {
  std::int64_t i;
  std::memcpy(&i, &x, sizeof(double));
  return static_cast<double>(i - 4607182418800017408LL)/4503599627370496.0;
}

//! \brief Inverse of nqt1_log.
inline double nqt1_exp(double y) {
  std::int64_t i = static_cast<std::int64_t>(y*4503599627370496.0)
                   + 4607182418800017408LL;
  double x;
  std::memcpy(&x, &i, sizeof(double));
  return x;
}

//! \brief Second-order approximant, C1-continuous, about 0.1 per cent maximum
//!   error.  Adds the quadratic correction b(1-b)/3 to nqt1_log, with b the
//!   fractional mantissa.
inline double nqt2_log(double x) {
  std::int64_t i;
  std::memcpy(&i, &x, sizeof(double));
  const double b = static_cast<double>(i & 0x000FFFFFFFFFFFFFLL)
                   /4503599627370496.0;
  return static_cast<double>(i - 4607182418800017408LL)/4503599627370496.0
         + b*(1.0 - b)/3.0;
}

//! \brief Inverse of nqt2_log, solving b + b(1-b)/3 = f as b = 2 - sqrt(4-3f).
//!
//!   The integer part must come from a true floor rather than truncation:
//!   stored rates reach kRateFloor, where y is about -332, and truncation
//!   toward zero would give the wrong integer part and a negative fractional
//!   part, so the square root would return NaN.
inline double nqt2_exp(double y) {
  const std::int64_t e = static_cast<std::int64_t>(std::floor(y));
  const double f = y - static_cast<double>(e);
  const double b = 2.0 - std::sqrt(4.0 - 3.0*f);
  const std::int64_t mantissa =
      static_cast<std::int64_t>(b*4503599627370496.0);
  std::int64_t i = (e + 1023LL)*4503599627370496LL + mantissa;
  double x;
  std::memcpy(&x, &i, sizeof(double));
  return x;
}

//----------------------------------------------------------------------------------------
//! \fn int ClampTableIndex(Real idx, int n_tab)
//! \brief Row index for a fractional grid position, kept in range so that idx
//!   and idx+1 are both valid rows.
//!
//!   Only the index is clamped.  A caller keeps the raw fractional position
//!   for the interpolation weight, so outside the table the last interval's
//!   slope continues rather than the value freezing.  In log-log that
//!   extrapolation is exact for a power law, which is what a radiative
//!   recombination rate nearly is.
inline int ClampTableIndex(Real idx, int n_tab) {
  idx = std::max(0.0, std::min(idx, static_cast<Real>(n_tab - 1)));
  return std::min(static_cast<int>(idx), n_tab - 2);
}

//----------------------------------------------------------------------------------------
//! \class RateGrid
//! \brief A uniform grid in log10(T), shared by every rate tabulated on it.
//!
//!   Sharing the grid is what makes a lookup cheap: the index is computed once
//!   per cell and reused for every rate, rather than once per rate.  Rates
//!   living in different temperature ranges therefore want a small number of
//!   grids, each covering the ions assigned to it, rather than one grid per
//!   ion, which would need one index computation apiece.
class RateGrid {
 public:
  RateGrid() = default;
  RateGrid(Real logt_min, Real logt_max, int npoints)
      : logt_min_(logt_min), npoints_(npoints) {
    if (npoints < 2 || !(logt_max > logt_min)) {
      std::stringstream msg;
      msg << "RateGrid: need at least two points and log T max above min, got "
          << npoints << " points over [" << logt_min << ", " << logt_max
          << "].";
      throw std::runtime_error(msg.str());
    }
    dlogt_ = (logt_max - logt_min)/(npoints - 1);
  }

  int NumPoints() const { return npoints_; }
  Real LogTMin() const { return logt_min_; }
  Real LogTMax() const { return logt_min_ + (npoints_ - 1)*dlogt_; }
  //! Grid spacing in dex, which is what sets the interpolation error.
  Real DLogT() const { return dlogt_; }
  Real TemperatureAt(int i) const {
    return std::pow(10.0, logt_min_ + i*dlogt_);
  }

 private:
  Real logt_min_ = 0.0, dlogt_ = 1.0;
  int npoints_ = 0;
};

//----------------------------------------------------------------------------------------
// The encoding a mode stores values in, and its inverse.  A table holds
// Encode of the rate on a grid uniform in Encode of the temperature, so one
// pair of functions describes both axes.

template <InterpMode Mode> inline Real TabEncode(Real x);
template <InterpMode Mode> inline Real TabDecode(Real y);

template <> inline Real TabEncode<InterpMode::kLogLog>(Real x) {
  return std::log10(x);
}
template <> inline Real TabDecode<InterpMode::kLogLog>(Real y) {
  return std::pow(10.0, y);
}
template <> inline Real TabEncode<InterpMode::kNqt1>(Real x) {
  return nqt1_log(x);
}
template <> inline Real TabDecode<InterpMode::kNqt1>(Real y) {
  return nqt1_exp(y);
}
//! kNqt2 shares the kNqt1 grid, so its temperature encoding is nqt1_log while
//! the values it stores are nqt2_log.  TabEncode is the temperature axis and
//! TabDecode the value axis, which is why the two differ here alone.
template <> inline Real TabEncode<InterpMode::kNqt2>(Real x) {
  return nqt1_log(x);
}
template <> inline Real TabDecode<InterpMode::kNqt2>(Real y) {
  return nqt2_exp(y);
}

//----------------------------------------------------------------------------------------
//! \fn Real GridPosition(Real temp, Real enc_min, Real inv_spacing)
//! \brief Fractional grid position of a temperature, unclamped, so a caller
//!   keeps the raw value for the interpolation weight.
template <InterpMode Mode>
inline Real GridPosition(Real temp, Real enc_min, Real inv_spacing) {
  return (TabEncode<Mode>(temp) - enc_min)*inv_spacing;
}

//----------------------------------------------------------------------------------------
//! \fn Real TabInterp(const Table &tab, int irate, Real idx, int n_tab)
//! \brief Linear interpolation in the stored encoding, decoded on the way out.
//!
//!   Only the row index is clamped, so a temperature outside the table
//!   continues the last interval's slope.  The table is any object with
//!   `operator()(int, int)`, which is what `AthenaArray<Real>` provides.
template <InterpMode Mode, class Table>
inline Real TabInterp(const Table &tab, int irate, Real idx, int n_tab) {
  const int i = ClampTableIndex(idx, n_tab);
  const Real frac = idx - i;
  return TabDecode<Mode>(tab(i, irate)
                         + frac*(tab(i + 1, irate) - tab(i, irate)));
}

}  // namespace Rates

#endif  // CHEMISTRY2_IONS_RATES_RATE_TABLE_HPP_
