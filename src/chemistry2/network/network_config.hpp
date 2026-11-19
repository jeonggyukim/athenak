#ifndef CHEMISTRY2_NETWORK_NETWORK_CONFIG_HPP_
#define CHEMISTRY2_NETWORK_NETWORK_CONFIG_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file network_config.hpp
//! \brief Per-cell element budgets and electron abundance of a chemical network, from its
//!        species table.
//!
//!   A chemical network describes itself at compile time with
//!     rows                       std::array<SpeciesRow, NSPEC>, in storage order
//!     STORES_ELECTRONS           x_e is a stored species (ROW_EL), not a charge sum
//!     ROW_EL                     that species, when STORES_ELECTRONS
//!     HYDROGEN_REMAINDER_CHARGE  charge of the hydrogen in no stored species: 1 when
//!                                that hydrogen is H+ (NCR), 0 when it is H I (GOW17)
//!   Every function below is unrolled over the rows at compile time; a row without the
//!   element or without charge contributes no instruction.

#include <cstddef>
#include <utility>

#include "athena.hpp"
#include "chemistry2/network/species_table.hpp"

namespace chemistry2 {

template <class Network>
constexpr int NumSpecies() { return static_cast<int>(Network::rows.size()); }

//! Subtract the atoms of element E in species I from x
template <class Network, int E, std::size_t I>
KOKKOS_INLINE_FUNCTION void SubtractAtoms(const Real *y, Real &x) {
  constexpr int n = Network::rows[I].comp[E];
  if constexpr (n == 1) {
    x -= y[I];
  } else if constexpr (n != 0) {
    x -= n*y[I];
  }
}

template <class Network, int E, std::size_t... I>
KOKKOS_INLINE_FUNCTION Real ElementRemainder(const Real *y, const Real x_elem,
                                             std::index_sequence<I...>) {
  Real x = x_elem;
  (SubtractAtoms<Network, E, I>(y, x), ...);
  return x;
}

//! \brief Abundance per H of element E in no stored species, x_elem minus the atoms of E
//!        in each stored species, subtracted in storage order. Not floored.
template <class Network, int E>
KOKKOS_INLINE_FUNCTION Real ElementRemainder(const Real *y, const Real x_elem) {
  return ElementRemainder<Network, E>(y, x_elem,
                                      std::make_index_sequence<NumSpecies<Network>()>{});
}

//! \brief ElementRemainder floored at zero: the abundance of the neutral atom of E when
//!        the network does not store it (the ghost species).
template <class Network, int E>
KOKKOS_INLINE_FUNCTION Real ElementBudget(const Real *y, const Real x_elem) {
  return Kokkos::fmax(ElementRemainder<Network, E>(y, x_elem), 0.0);
}

template <class Network, std::size_t I>
KOKKOS_INLINE_FUNCTION void AddCharge(const Real *y, Real &x_e) {
  constexpr int q = Network::rows[I].charge;
  if constexpr (q == 1) {
    x_e += y[I];
  } else if constexpr (q != 0) {
    x_e += q*y[I];
  }
}

template <class Network, std::size_t... I>
KOKKOS_INLINE_FUNCTION Real ChargeSum(const Real *y, std::index_sequence<I...>) {
  Real x_e = 0.0;
  (AddCharge<Network, I>(y, x_e), ...);
  return x_e;
}

//! Charge per H of the stored species, summed in storage order
template <class Network>
KOKKOS_INLINE_FUNCTION Real ChargeSum(const Real *y) {
  return ChargeSum<Network>(y, std::make_index_sequence<NumSpecies<Network>()>{});
}

//! \brief Electron abundance per H: the stored species, or charge neutrality over the
//!        stored species and the hydrogen in no stored species.
template <class Network>
KOKKOS_INLINE_FUNCTION Real Electrons(const Real *y) {
  if constexpr (Network::STORES_ELECTRONS) {
    return y[Network::ROW_EL];
  } else {
    Real x_e = ChargeSum<Network>(y);
    if constexpr (Network::HYDROGEN_REMAINDER_CHARGE != 0) {
      x_e += Network::HYDROGEN_REMAINDER_CHARGE*ElementRemainder<Network, ELEM_H>(y, 1.0);
    }
    return x_e;
  }
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_NETWORK_CONFIG_HPP_
