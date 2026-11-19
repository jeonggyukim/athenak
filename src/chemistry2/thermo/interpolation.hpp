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
    return i - 1;
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
    return i - 1;
  }
}

//---------------------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION Real LinearInterp(const Real x0, const Real x1,
                                         const Real y0, const Real y1,
                                         const Real x) {
  return y0 + ((y1 - y0) / (x1 - x0)) * (x - x0);
}

//---------------------------------------------------------------------------------------
/*!
 * \brief Interpolation with index provided.
 */
template <std::size_t N_xarr, std::size_t N_data>
KOKKOS_INLINE_FUNCTION Real LP1Di(const Kokkos::Array<Real, N_xarr> xarr,
                                  const Kokkos::Array<Real, N_data> data,
                                  const int ix, const Real x) {
  return LinearInterp(xarr[ix], xarr[ix + 1], data[ix], data[ix + 1], x);
}

//---------------------------------------------------------------------------------------
/*!
 * \brief 2D array bi-linear interpolation with index provided
 */
template <std::size_t N_xarr, std::size_t N_yarr, std::size_t N_data>
KOKKOS_INLINE_FUNCTION Real LP2Di(const Kokkos::Array<Real, N_xarr> xarr,
                                  const Kokkos::Array<Real, N_yarr> yarr,
                                  const int lenx, const int ix, const int iy,
                                  const Kokkos::Array<Real, N_data> data,
                                  const Real x, const Real y) {
  Real fl1, fl2;
  const Real x0 = xarr[ix];
  const Real x1 = xarr[ix + 1];
  fl1 = LinearInterp(x0, x1, data[iy * lenx + ix], data[iy * lenx + ix + 1], x);
  fl2 = LinearInterp(x0, x1, data[(iy + 1) * lenx + ix],
                     data[(iy + 1) * lenx + ix + 1], x);
  return LinearInterp(yarr[iy], yarr[iy + 1], fl1, fl2, y);
}

}  // namespace interpolation

}  // namespace chemistry2

#endif  // CHEMISTRY2_THERMO_INTERPOLATION_HPP_
