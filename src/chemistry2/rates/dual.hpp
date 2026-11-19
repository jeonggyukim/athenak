#ifndef CHEMISTRY2_RATES_DUAL_HPP_
#define CHEMISTRY2_RATES_DUAL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dual.hpp
//! \brief Dual numbers for forward-mode differentiation in one variable: v is the
//!        value, d its derivative. Comparisons, fmin and fmax act on the value.

#include "athena.hpp"

namespace chemistry2 {
namespace dual {

struct Dual {
  Real v, d;
  Dual() = default;
  KOKKOS_INLINE_FUNCTION constexpr Dual(const Real value, const Real deriv)
      : v(value), d(deriv) {}
  //! a constant: derivative zero
  KOKKOS_INLINE_FUNCTION constexpr Dual(const Real value)  // NOLINT(runtime/explicit)
      : v(value), d(0.0) {}
  KOKKOS_INLINE_FUNCTION Dual &operator+=(const Dual b) {
    v += b.v;
    d += b.d;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION Dual &operator-=(const Dual b) {
    v -= b.v;
    d -= b.d;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION Dual &operator*=(const Dual b) {
    d = d*b.v + v*b.d;
    v *= b.v;
    return *this;
  }
};

KOKKOS_INLINE_FUNCTION Dual operator-(const Dual a) { return Dual(-a.v, -a.d); }
KOKKOS_INLINE_FUNCTION Dual operator+(const Dual a, const Dual b) {
  return Dual(a.v + b.v, a.d + b.d);
}
KOKKOS_INLINE_FUNCTION Dual operator-(const Dual a, const Dual b) {
  return Dual(a.v - b.v, a.d - b.d);
}
KOKKOS_INLINE_FUNCTION Dual operator*(const Dual a, const Dual b) {
  return Dual(a.v*b.v, a.d*b.v + a.v*b.d);
}
KOKKOS_INLINE_FUNCTION Dual operator/(const Dual a, const Dual b) {
  const Real q = a.v/b.v;
  return Dual(q, (a.d - q*b.d)/b.v);
}
KOKKOS_INLINE_FUNCTION Dual exp(const Dual a) {
  const Real e = Kokkos::exp(a.v);
  return Dual(e, e*a.d);
}
KOKKOS_INLINE_FUNCTION Dual log(const Dual a) { return Dual(Kokkos::log(a.v), a.d/a.v); }
KOKKOS_INLINE_FUNCTION Dual sqrt(const Dual a) {
  const Real r = Kokkos::sqrt(a.v);
  return Dual(r, 0.5*a.d/r);
}
KOKKOS_INLINE_FUNCTION Dual pow(const Dual a, const Real p) {
  const Real r = Kokkos::pow(a.v, p);
  return Dual(r, p*r*a.d/a.v);
}
KOKKOS_INLINE_FUNCTION Dual fmax(const Dual a, const Dual b) { return (a.v >= b.v) ? a : b; }
KOKKOS_INLINE_FUNCTION Dual fmin(const Dual a, const Dual b) { return (a.v <= b.v) ? a : b; }
KOKKOS_INLINE_FUNCTION bool operator<(const Dual a, const Dual b) { return a.v < b.v; }
KOKKOS_INLINE_FUNCTION bool operator>(const Dual a, const Dual b) { return a.v > b.v; }
KOKKOS_INLINE_FUNCTION bool operator<=(const Dual a, const Dual b) { return a.v <= b.v; }
KOKKOS_INLINE_FUNCTION bool operator>=(const Dual a, const Dual b) { return a.v >= b.v; }

//! The value of a Real or a Dual, for code templated on either
KOKKOS_INLINE_FUNCTION Real Value(const Real a) { return a; }
KOKKOS_INLINE_FUNCTION Real Value(const Dual a) { return a.v; }

}  // namespace dual
}  // namespace chemistry2

#endif  // CHEMISTRY2_RATES_DUAL_HPP_
