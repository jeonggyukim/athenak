//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file update_ncr_oxygen_semi_implicit.cpp
//! \brief Instantiation of the chemistry2 cell loop for NCR with the oxygen ladder
//!        (O+, O++) and the semi-implicit solver.

#include "chemistry2/registry.hpp"
#include "chemistry2/update.hpp"

namespace chemistry2 {
template void Chemistry2::Update<Pair<NCROxygen<2>, SemiImplicit>>(Real t, Real dt);
}  // namespace chemistry2
