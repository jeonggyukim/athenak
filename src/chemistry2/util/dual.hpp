#ifndef CHEMISTRY2_UTIL_DUAL_HPP_
#define CHEMISTRY2_UTIL_DUAL_HPP_
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

// arithmetic between dual numbers: the sum, product and quotient rules
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
  const Real r = 1.0/b.v;
  const Real q = a.v*r;
  return Dual(q, (a.d - q*b.d)*r);
}
// Mixed with a constant: no work on its zero derivative
KOKKOS_INLINE_FUNCTION Dual operator+(const Dual a, const Real b) {
  return Dual(a.v + b, a.d);
}
KOKKOS_INLINE_FUNCTION Dual operator+(const Real a, const Dual b) {
  return Dual(a + b.v, b.d);
}
KOKKOS_INLINE_FUNCTION Dual operator-(const Dual a, const Real b) {
  return Dual(a.v - b, a.d);
}
KOKKOS_INLINE_FUNCTION Dual operator-(const Real a, const Dual b) {
  return Dual(a - b.v, -b.d);
}
KOKKOS_INLINE_FUNCTION Dual operator*(const Dual a, const Real b) {
  return Dual(a.v*b, a.d*b);
}
KOKKOS_INLINE_FUNCTION Dual operator*(const Real a, const Dual b) {
  return Dual(a*b.v, a*b.d);
}
KOKKOS_INLINE_FUNCTION Dual operator/(const Dual a, const Real b) {
  const Real r = 1.0/b;
  return Dual(a.v*r, a.d*r);
}
KOKKOS_INLINE_FUNCTION Dual operator/(const Real a, const Dual b) {
  const Real r = 1.0/b.v;
  const Real q = a*r;
  return Dual(q, -q*b.d*r);
}
// elementary functions: f(v) and f'(v) d (chain rule)
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
  return Dual(r, p*r*(a.d/a.v));
}
KOKKOS_INLINE_FUNCTION Dual pow(const Real b, const Dual a) {
  const Real r = Kokkos::pow(b, a.v);
  return Dual(r, r*Kokkos::log(b)*a.d);
}
KOKKOS_INLINE_FUNCTION Dual pow(const Dual a, const Dual p) {
  const Real la = Kokkos::log(a.v);
  const Real r = Kokkos::exp(p.v*la);
  return Dual(r, r*(p.d*la + p.v*a.d/a.v));
}
KOKKOS_INLINE_FUNCTION Dual log10(const Dual a) {
  return Dual(Kokkos::log10(a.v), a.d/(a.v*2.302585092994046));
}
// fmin/fmax pick a whole dual number by comparing the values
KOKKOS_INLINE_FUNCTION Dual fmax(const Dual a, const Dual b) {
  return (a.v >= b.v) ? a : b;
}
KOKKOS_INLINE_FUNCTION Dual fmin(const Dual a, const Dual b) {
  return (a.v <= b.v) ? a : b;
}
KOKKOS_INLINE_FUNCTION Dual fmax(const Dual a, const Real b) {
  return (a.v >= b) ? a : Dual(b);
}
KOKKOS_INLINE_FUNCTION Dual fmin(const Dual a, const Real b) {
  return (a.v <= b) ? a : Dual(b);
}
// comparisons act on the values
KOKKOS_INLINE_FUNCTION bool operator<(const Dual a, const Dual b) { return a.v < b.v; }
KOKKOS_INLINE_FUNCTION bool operator>(const Dual a, const Dual b) { return a.v > b.v; }
KOKKOS_INLINE_FUNCTION bool operator<=(const Dual a, const Dual b) { return a.v <= b.v; }
KOKKOS_INLINE_FUNCTION bool operator>=(const Dual a, const Dual b) { return a.v >= b.v; }

// Real overloads, so code templated on Real or Dual calls dual::exp etc. for both
KOKKOS_INLINE_FUNCTION Real exp(const Real a) { return Kokkos::exp(a); }
KOKKOS_INLINE_FUNCTION Real log(const Real a) { return Kokkos::log(a); }
KOKKOS_INLINE_FUNCTION Real log10(const Real a) { return Kokkos::log10(a); }
KOKKOS_INLINE_FUNCTION Real sqrt(const Real a) { return Kokkos::sqrt(a); }
KOKKOS_INLINE_FUNCTION Real pow(const Real a, const Real p) { return Kokkos::pow(a, p); }
KOKKOS_INLINE_FUNCTION Real fmax(const Real a, const Real b) {
  return Kokkos::fmax(a, b);
}
KOKKOS_INLINE_FUNCTION Real fmin(const Real a, const Real b) {
  return Kokkos::fmin(a, b);
}

//! The value of a Real or a Dual, for code templated on either
KOKKOS_INLINE_FUNCTION Real Value(const Real a) { return a; }
KOKKOS_INLINE_FUNCTION Real Value(const Dual a) { return a.v; }

}  // namespace dual
}  // namespace chemistry2

#endif  // CHEMISTRY2_UTIL_DUAL_HPP_
