//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file update_h2_semi_implicit.cpp
//! \brief Instantiation of the chemistry2 cell loop for H2 with the semi-implicit solver.
//!        One file per registered pair, so the pairs compile in parallel.

#include "chemistry2/registry.hpp"
#include "chemistry2/update.hpp"

namespace chemistry2 {
template void Chemistry2::Update<Pair<H2, SemiImplicit>>(Real t, Real dt);
}  // namespace chemistry2
