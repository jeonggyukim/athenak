#ifndef CHEMISTRY2_NETWORK_SPECIES_TABLE_HPP_
#define CHEMISTRY2_NETWORK_SPECIES_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file species_table.hpp
//! \brief Stored species of a chemical network: name, charge and atoms of each element.
//!
//!   A chemical network lists its stored species once, in storage order:
//!
//!     static constexpr std::array<SpeciesRow, 2> rows = {
//!         Species("H2").H(2),
//!         Species("H").H(1)};
//!
//!   The number of passive scalars, the output names, the abundance of each element in
//!   no stored species and the electron abundance follow from these rows at compile
//!   time; network_config.hpp holds the per-cell functions.

#include <array>
#include <cstddef>
#include <string_view>

namespace chemistry2 {

// Elements of GOW17, NCR and the planned ion sets. Add one when an ion set needs it.
enum Element : int {ELEM_H, ELEM_HE, ELEM_C, ELEM_N, ELEM_O, ELEM_SI, ELEM_S, ELEM_NE,
                    NELEM};

//! \brief One stored species: its output name, its charge in units of e and the number
//!        of atoms of each element it contains.
//!
//!   Built with Species() and one call per element, e.g. Species("HCO+").Charge(1)
//!   .H(1).C(1).O(1); elements not named have no atoms.
struct SpeciesRow {
  std::string_view name;
  int charge = 0;
  std::array<int, NELEM> comp{};

  constexpr SpeciesRow Charge(int q) const {
    SpeciesRow r = *this;
    r.charge = q;
    return r;
  }
  constexpr SpeciesRow Atoms(Element e, int n) const {
    SpeciesRow r = *this;
    r.comp[e] = n;
    return r;
  }
  constexpr SpeciesRow H(int n) const { return Atoms(ELEM_H, n); }
  constexpr SpeciesRow He(int n) const { return Atoms(ELEM_HE, n); }
  constexpr SpeciesRow C(int n) const { return Atoms(ELEM_C, n); }
  constexpr SpeciesRow N(int n) const { return Atoms(ELEM_N, n); }
  constexpr SpeciesRow O(int n) const { return Atoms(ELEM_O, n); }
  constexpr SpeciesRow Si(int n) const { return Atoms(ELEM_SI, n); }
  constexpr SpeciesRow S(int n) const { return Atoms(ELEM_S, n); }
  constexpr SpeciesRow Ne(int n) const { return Atoms(ELEM_NE, n); }
};

constexpr SpeciesRow Species(std::string_view name) {
  SpeciesRow r;
  r.name = name;
  return r;
}

//! Index of the species called `name` in `rows`, or -1 if there is none
template <std::size_t N>
constexpr int SpeciesIndex(const std::array<SpeciesRow, N> &rows, std::string_view name) {
  for (std::size_t n = 0; n < N; ++n) {
    if (rows[n].name == name) return static_cast<int>(n);
  }
  return -1;
}

template <std::size_t N>
constexpr bool AnyCharged(const std::array<SpeciesRow, N> &rows) {
  for (std::size_t n = 0; n < N; ++n) {
    if (rows[n].charge != 0) return true;
  }
  return false;
}

template <std::size_t N>
constexpr bool UniqueNames(const std::array<SpeciesRow, N> &rows) {
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t m = n + 1; m < N; ++m) {
      if (rows[n].name == rows[m].name) return false;
    }
  }
  return true;
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_SPECIES_TABLE_HPP_
