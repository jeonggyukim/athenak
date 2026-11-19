#ifndef CHEMISTRY2_UPDATE_HPP_
#define CHEMISTRY2_UPDATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file update.hpp
//! \brief The cell loop of chemistry2: load a cell into registers, solve, store.
//!
//!   The species are read and written only by LoadSpecies and StoreSpecies, so a change
//!   of storage (per-network arrays, sparse variables) changes these two functions only.

#include <type_traits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "chemistry2/chemistry2.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

namespace chemistry2 {

template <int N>
KOKKOS_INLINE_FUNCTION void LoadSpecies(const DvceArray5D<Real> &w0, int m, int first,
                                        int k, int j, int i, Real *y) {
  for (int n = 0; n < N; ++n) y[n] = w0(m, first + n, k, j, i);
}

template <int N>
KOKKOS_INLINE_FUNCTION void StoreSpecies(const DvceArray5D<Real> &w0, int m, int first,
                                         int k, int j, int i, const Real *y) {
  for (int n = 0; n < N; ++n) w0(m, first + n, k, j, i) = y[n];
}

template <class Net>
const typename Net::Settings &SettingsOf(const NetworkSettings &s) {
  static_assert(std::is_same_v<Net, H2>, "add the network's settings to SettingsOf");
  return s.h2;
}

template <class Pair>
void Chemistry2::Update(const Real t, const Real dt) {
  using Net = typename Pair::Network;
  using Solver = typename Pair::Integrator;
  constexpr int NSPEC = Net::NSPEC;

  const auto w0 = is_mhd_ ? pmy_pack_->pmhd->w0 : pmy_pack_->phydro->w0;
  const Real gamma = is_mhd_ ? pmy_pack_->pmhd->peos->eos_data.gamma
                             : pmy_pack_->phydro->peos->eos_data.gamma;
  const int first = first_scalar_;
  const UnitFactors units = units_;
  const SemiImplicitSettings solver = semi_implicit_;
  const typename Net::Settings net = SettingsOf<Net>(networks_);
  const InputSource inputs = inputs_;

  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  par_for("chemistry2", DevExeSpace(), 0, pmy_pack_->nmb_thispack - 1, indcs.ks,
          indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real y[NSPEC];  // NOLINT(runtime/arrays)
    LoadSpecies<NSPEC>(w0, m, first, k, j, i, y);
    Real e = w0(m, IEN, k, j, i);
    const CellInputs in = inputs.Load(m, k, j, i);
    const typename Net::Cell cell = Net::MakeCell(w0(m, IDN, k, j, i), gamma, net,
                                                  units, in);
    Solver::Solve(solver, cell, t, dt, y, e);
    StoreSpecies<NSPEC>(w0, m, first, k, j, i, y);
    w0(m, IEN, k, j, i) = e;
  });
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_UPDATE_HPP_
