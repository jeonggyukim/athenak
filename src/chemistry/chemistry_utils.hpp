#ifndef CHEMISTRY_CHEMISTRY_UTILS_HPP_
#define CHEMISTRY_CHEMISTRY_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry_utils.hpp
//  \brief utilities for chemistry

#include "athena.hpp"
#include "utils/register_array.hpp"

namespace chemistry {

/*!
 * \brief A struct to hold the creation and destruction rates
 *
 * \tparam N The size of each array
 */
template <std::size_t N>
struct CDRates_t {
  RegisterArray<Real, N> creation, destruction;
};

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

/*!
 * \brief Compute the numerical Jacobian for a chemical network.
 *
 * \tparam network_t The network object
 * \tparam vec_type The type of y_in
 * \tparam mat_type The type of jac
 * \param[in] network The chemical network to compute the Jacobian of
 * \param[in] t Current time
 * \param[in] dt Time step
 * \param[in] y_in The current state
 * \param[out] jac The Jacobian
 */
template <class network_t, class vec_type, class mat_type>
KOKKOS_FUNCTION void numerical_jacobian(const network_t& network, const Real t,
                                        const Real dt, const vec_type& y_in,
                                        const mat_type& jac) {
  RegisterArray<Real, network.neqs> f0, fp;

  // PERFORMANCE: this builds the Jacobian with 1 + neqs evaluations of the full
  // network. For GOW17 that is 14 calls, each running UpdateRates_ with its 58
  // transcendentals, so roughly 810 transcendental evaluations per Jacobian --
  // the dominant cost of the chemistry module in the measured profile.
  //
  // 248 of UpdateRates_'s 260 lines depend on the state only through T, so most
  // of that work is redundant across the species columns. Hoisting it is not
  // free, because T itself depends on x(H2) and on the electron abundance; see
  // ~/ai-notes/docs-claude/athenak-chemistry/gow17-jacobian-hoist-design.md
  // for the derivation and the rank-1 thermal correction that makes the hoist
  // exact rather than approximate.

  // Evaluate the unperturbed f0
  network.evaluate_function(t, dt, y_in, f0);

  // The perturbation to add to each element in turn
  const Real perturbation_factor =
      Kokkos::sqrt(Kokkos::ArithTraits<Real>::epsilon());

  // Columns 0..neqs-2 perturb a species and column neqs-1 perturbs the internal
  // energy. The rate coefficients see a species perturbation only through T, so
  // this loop recomputes the same k(T) values up to 13 times over.
  for (int j = 0; j < network.neqs; ++j) {
    // Add the perturbation to the jth element
    const Real perturbation =
        perturbation_factor * Kokkos::fmax(Kokkos::abs(y_in(j)), Real(1.0));
    const Real y_unperturbed = y_in(j);
    y_in(j) += perturbation;

    // Compute the perturbed values of fp
    network.evaluate_function(t, dt, y_in, fp);

    // realized step, robust to rounding
    const Real inverse_diff = Real(1.0) / perturbation;

    // Update the Jacobian
    for (int k = 0; k < network.neqs; ++k) {
      jac(k, j) = (fp(k) - f0(k)) * inverse_diff;
    }

    // Reset the perturbed field
    y_in(j) = y_unperturbed;
  }
}
}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_UTILS_HPP_
