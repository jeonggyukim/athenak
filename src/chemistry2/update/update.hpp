#ifndef CHEMISTRY2_UPDATE_UPDATE_HPP_
#define CHEMISTRY2_UPDATE_UPDATE_HPP_
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

// copy the N species abundances of cell (m,k,j,i) from the passive scalars into y
template <int N>
KOKKOS_INLINE_FUNCTION void LoadSpecies(const DvceArray5D<Real> &w0, int m, int first,
                                        int k, int j, int i, Real *y) {
  for (int n = 0; n < N; ++n) y[n] = w0(m, first + n, k, j, i);
}

// write y back to the passive scalars of cell (m,k,j,i)
template <int N>
KOKKOS_INLINE_FUNCTION void StoreSpecies(const DvceArray5D<Real> &w0, int m, int first,
                                         int k, int j, int i, const Real *y) {
  for (int n = 0; n < N; ++n) w0(m, first + n, k, j, i) = y[n];
}

//! |grad v| in code units for the LVG escape of CO lines: the mean over the active
//! dimensions of |dv_d/dx_d|, centered differences, as in src/chemistry
KOKKOS_INLINE_FUNCTION Real VelocityGradient(const DvceArray5D<Real> &w0, int m, int k,
                                             int j, int i, const RegionSize &size,
                                             bool multi_d, bool three_d) {
  Real sum = Kokkos::fabs(w0(m, IVX, k, j, i + 1) - w0(m, IVX, k, j, i - 1))/size.dx1;
  Real n_dim = 1.0;
  if (multi_d) {
    sum += Kokkos::fabs(w0(m, IVY, k, j + 1, i) - w0(m, IVY, k, j - 1, i))/size.dx2;
    n_dim += 1.0;
  }
  if (three_d) {
    sum += Kokkos::fabs(w0(m, IVZ, k + 1, j, i) - w0(m, IVZ, k - 1, j, i))/size.dx3;
    n_dim += 1.0;
  }
  return 0.5*sum/n_dim;
}

// the settings of network Net among those read from <chemistry2>
template <class Net>
const typename Net::Settings &SettingsOf(const NetworkSettings &s) {
  if constexpr (std::is_same_v<Net, H2>) {
    return s.h2;
  } else if constexpr (std::is_same_v<Net, GOW17>) {
    return s.gow17;
  } else if constexpr (std::is_same_v<Net, GOW17Analytic>) {
    return s.gow17_analytic;
  } else if constexpr (std::is_same_v<Net, NCR>) {
    return s.ncr;
  } else if constexpr (std::is_same_v<Net, NCRAnalytic>) {
    return s.ncr_analytic;
  } else {
    static_assert(std::is_same_v<Net, NCROxygen<2>>,
                  "add the network's settings to SettingsOf");
    return s.ncr_oxygen;
  }
}

//! Advance the chemistry and internal energy of every active cell from t over dt with
//! the network/solver pair Pair
template <class Pair>
void Chemistry2::Update(const Real t, const Real dt) {
  using Net = typename Pair::Network;
  using Solver = typename Pair::Integrator;
  constexpr int NSPEC = Net::NSPEC;

  // copy everything the kernel needs into locals (captured by value on the device)
  const auto w0 = is_mhd_ ? pmy_pack_->pmhd->w0 : pmy_pack_->phydro->w0;
  const Real gamma = is_mhd_ ? pmy_pack_->pmhd->peos->eos_data.gamma
                             : pmy_pack_->phydro->peos->eos_data.gamma;
  const int first = first_scalar_;
  const UnitFactors units = units_;
  typename Solver::Settings solver;
  if constexpr (std::is_same_v<typename Solver::Settings, KokkosBDFSettings>) {
    solver = kokkos_bdf_;
  } else {
    solver = semi_implicit_;
  }
  const typename Net::Settings net = SettingsOf<Net>(networks_);
  const InputSource inputs = inputs_;

  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const auto sizes = pmy_pack_->pmb->mb_size.d_view;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const Real inv_time_cgs = 1.0/units.time_cgs;
  par_for("chemistry2", DevExeSpace(), 0, pmy_pack_->nmb_thispack - 1, indcs.ks,
          indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // load the cell: abundances, internal energy, radiation and CR inputs, |grad v|
    Real y[NSPEC];  // NOLINT(runtime/arrays)
    LoadSpecies<NSPEC>(w0, m, first, k, j, i, y);
    Real e = w0(m, IEN, k, j, i);
    CellInputs in = inputs.Load(m, k, j, i);
    in.grad_v = VelocityGradient(w0, m, k, j, i, sizes(m), multi_d, three_d)*
                inv_time_cgs;
    // the per-cell constants of the network (n_H, unit factors, fields), then the solve
    const typename Net::Cell cell = Net::MakeCell(w0(m, IDN, k, j, i), gamma, net,
                                                  units, in);
    Solver::Solve(solver, cell, t, dt, y, e);
    // store the new abundances and internal energy
    StoreSpecies<NSPEC>(w0, m, first, k, j, i, y);
    w0(m, IEN, k, j, i) = e;
  });
}

//! Solve every active cell to equilibrium with the semi-implicit pair Pair; returns the
//! cells that did not converge and the largest substep count of those that did
template <class Pair>
EquilibriumStats Chemistry2::EquilibrateCells(const EquilibriumSettings &q) {
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

  // one flat index over (m, k, j, i) of the active cells, for the reduction
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nji = nx2*nx1, nkji = nx3*nji;
  const int nmkji = pmy_pack_->nmb_thispack*nkji;
  int not_converged = 0, max_substeps = 0;
  Kokkos::parallel_reduce("chemistry2_equilibrate",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
                          KOKKOS_LAMBDA(const int idx, int &nc, int &ns) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji;
    const int j = (idx - m*nkji - k*nji)/nx1;
    const int i = idx - m*nkji - k*nji - j*nx1 + is;
    Real y[NSPEC];  // NOLINT(runtime/arrays)
    LoadSpecies<NSPEC>(w0, m, first, k + ks, j + js, i, y);
    Real e = w0(m, IEN, k + ks, j + js, i);
    // grad_v stays 0: no LVG escape term in an equilibrium
    const CellInputs in = inputs.Load(m, k + ks, j + js, i);
    const typename Net::Cell cell = Net::MakeCell(w0(m, IDN, k + ks, j + js, i), gamma,
                                                  net, units, in);
    const int n = Solver::SolveToEquilibrium(solver, cell, q, y, e);
    if (n < 0) {
      nc += 1;
    } else if (n > ns) {
      ns = n;
    }
    StoreSpecies<NSPEC>(w0, m, first, k + ks, j + js, i, y);
    w0(m, IEN, k + ks, j + js, i) = e;
  }, Kokkos::Sum<int>(not_converged), Kokkos::Max<int>(max_substeps));
  return EquilibriumStats{not_converged, max_substeps};
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_UPDATE_UPDATE_HPP_
