#ifndef CHEMISTRY2_THERMO_INTERPOLATION_HPP_
#define CHEMISTRY2_THERMO_INTERPOLATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file interpolation.hpp
//! \brief Linear interpolation in the small tables of the heating and cooling terms.

#include "athena.hpp"
#include "chemistry2/util/dual.hpp"

namespace chemistry2 {
namespace interpolation {
// ---------------------------------------------------------------------------------------
/*!
 * \brief Find the proper index for interpolation
 *
 * \param len The size of xarr
 * \param xarr The array to find the interpolation index in
 * \param x The value to interpolate around
 * \return int The index of the xarr with the first instance where x > xarr[i]
 */
KOKKOS_INLINE_FUNCTION size_t LinearInterpIndex(const size_t len,
                                                const Real xarr[],
                                                const Real x) {
  if (x < xarr[0]) {
    return 0;
  } else if (x > xarr[len - 1]) {
    return len - 2;
  } else {
    int i = 0;
    while (x > xarr[i]) {
      i++;
    }
    return (i > 0) ? i - 1 : 0;  // x == xarr[0] gives the first interval
  }
}

template <std::size_t N>
KOKKOS_INLINE_FUNCTION size_t LinearInterpIndex(
    const size_t len, const Kokkos::Array<Real, N> xarr, const Real x) {
  if (x < xarr[0]) {
    return 0;
  } else if (x > xarr[len - 1]) {
    return len - 2;
  } else {
    int i = 0;
    while (x > xarr[i]) {
      i++;
    }
    return (i > 0) ? i - 1 : 0;  // x == xarr[0] gives the first interval
  }
}

//---------------------------------------------------------------------------------------
//! The point x (Real or dual::Dual) between the nodes x0 and x1; y0 and y1 are Real or
//! dual::Dual values at the nodes
template <class Y, class X>
KOKKOS_INLINE_FUNCTION auto LinearInterp(const Real x0, const Real x1, const Y y0,
                                         const Y y1, const X x) {
  return y0 + ((y1 - y0) * (1.0 / (x1 - x0))) * (x - x0);
}

// The CO tables are static constexpr class members. Under CUDA they are passed by value,
// so device code does not reference the host copy; elsewhere by reference, since the
// CPU compiler copies them on every call.
#if defined(KOKKOS_ENABLE_CUDA)
template <std::size_t N>
using TableArg = const Kokkos::Array<Real, N>;
#else
template <std::size_t N>
using TableArg = const Kokkos::Array<Real, N> &;
#endif

//---------------------------------------------------------------------------------------
/*!
 * \brief Interpolation with index provided.
 */
template <std::size_t N_xarr, std::size_t N_data, class X>
KOKKOS_INLINE_FUNCTION X LP1Di(TableArg<N_xarr> xarr,
                               TableArg<N_data> data,
                               const int ix, const X x) {
  return LinearInterp(xarr[ix], xarr[ix + 1], data[ix], data[ix + 1], x);
}

//---------------------------------------------------------------------------------------
/*!
 * \brief 2D array bi-linear interpolation with index provided
 */
template <std::size_t N_xarr, std::size_t N_yarr, std::size_t N_data, class X, class Y>
KOKKOS_INLINE_FUNCTION auto LP2Di(TableArg<N_xarr> xarr,
                                  TableArg<N_yarr> yarr,
                                  const int lenx, const int ix, const int iy,
                                  TableArg<N_data> data,
                                  const X x, const Y y) {
  const Real x0 = xarr[ix];
  const Real x1 = xarr[ix + 1];
  const X fl1 = LinearInterp(x0, x1, data[iy * lenx + ix], data[iy * lenx + ix + 1], x);
  const X fl2 = LinearInterp(x0, x1, data[(iy + 1) * lenx + ix],
                             data[(iy + 1) * lenx + ix + 1], x);
  return LinearInterp(yarr[iy], yarr[iy + 1], fl1, fl2, y);
}

}  // namespace interpolation

}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_INTERPOLATION_HPP_
