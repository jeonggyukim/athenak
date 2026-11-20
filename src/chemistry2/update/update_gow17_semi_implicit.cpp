//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file update_gow17_semi_implicit.cpp
//! \brief Instantiation of the chemistry2 cell loop for GOW17 with the semi-implicit
//!        solver.

#include "chemistry2/registry.hpp"
#include "chemistry2/update/update.hpp"

namespace chemistry2 {
template void Chemistry2::Update<Pair<GOW17, SemiImplicit>>(Real t, Real dt);
template void Chemistry2::Update<Pair<GOW17Analytic, SemiImplicit>>(Real t, Real dt);
template EquilibriumStats Chemistry2::EquilibrateCells<Pair<GOW17, SemiImplicit>>(
    const EquilibriumSettings &q);
template EquilibriumStats Chemistry2::EquilibrateCells<Pair<GOW17Analytic, SemiImplicit>>(
    const EquilibriumSettings &q);
}  // namespace chemistry2
