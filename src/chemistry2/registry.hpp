#ifndef CHEMISTRY2_REGISTRY_HPP_
#define CHEMISTRY2_REGISTRY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file registry.hpp
//! \brief The (chemical network, solver) pairs chemistry2 is built with.
//!
//!   The input names are turned into an index once, by Find; Dispatch calls a function
//!   with the pair of that index, on the host, once per task call. A pair not listed in
//!   Pairs is not compiled.

#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "chemistry2/network/gow17.hpp"
#include "chemistry2/network/h2.hpp"
#include "chemistry2/solvers/kokkos_bdf.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"

namespace chemistry2 {

template <class Net, template <class> class Solver>
struct Pair {
  using Network = Net;
  using Integrator = Solver<Net>;
};

using Pairs = std::tuple<Pair<H2, SemiImplicit>, Pair<GOW17, SemiImplicit>,
                         Pair<GOW17Analytic, SemiImplicit>, Pair<GOW17, KokkosBDF>>;

constexpr int NUM_PAIRS = static_cast<int>(std::tuple_size_v<Pairs>);

namespace registry_detail {
template <std::size_t... I>
int Find(std::string_view net, std::string_view solver, std::index_sequence<I...>) {
  int found = -1;
  ((found = (found < 0 && std::tuple_element_t<I, Pairs>::Network::NAME == net &&
             std::tuple_element_t<I, Pairs>::Integrator::NAME == solver)
                ? static_cast<int>(I) : found), ...);
  return found;
}

template <std::size_t... I>
std::string List(std::index_sequence<I...>) {
  std::string s;
  ((s += (I == 0 ? "" : ", ") +
         std::string(std::tuple_element_t<I, Pairs>::Network::NAME) + " + " +
         std::string(std::tuple_element_t<I, Pairs>::Integrator::NAME)), ...);
  return s;
}

template <std::size_t... I>
int SpeciesCount(std::string_view net, std::index_sequence<I...>) {
  int n = -1;
  ((n = (n < 0 && std::tuple_element_t<I, Pairs>::Network::NAME == net)
            ? std::tuple_element_t<I, Pairs>::Network::NSPEC : n), ...);
  return n;
}

template <class F, std::size_t... I>
bool Dispatch(int idx, F &&f, std::index_sequence<I...>) {
  return ((idx == static_cast<int>(I) ? (f(std::tuple_element_t<I, Pairs>{}), true)
                                      : false) || ...);
}
}  // namespace registry_detail

//! Index of the pair (network, solver) in Pairs, or -1
inline int Find(std::string_view net, std::string_view solver) {
  return registry_detail::Find(net, solver, std::make_index_sequence<NUM_PAIRS>{});
}

//! Number of stored species of the network called net, or -1 if none is registered
inline int SpeciesCount(std::string_view net) {
  return registry_detail::SpeciesCount(net, std::make_index_sequence<NUM_PAIRS>{});
}

//! The registered pairs, "network + solver, ..."
inline std::string List() {
  return registry_detail::List(std::make_index_sequence<NUM_PAIRS>{});
}

//! Call f(Pair{}) for the pair of index idx; false if there is none
template <class F>
bool Dispatch(int idx, F &&f) {
  return registry_detail::Dispatch(idx, std::forward<F>(f),
                                   std::make_index_sequence<NUM_PAIRS>{});
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_REGISTRY_HPP_
