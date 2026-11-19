//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file update_gow17_kokkos_bdf.cpp
//! \brief Instantiation of the chemistry2 cell loop for GOW17 with the Kokkos Kernels
//!        BDF solver.

#include "chemistry2/registry.hpp"
#include "chemistry2/update.hpp"

namespace chemistry2 {
template void Chemistry2::Update<Pair<GOW17, KokkosBDF>>(Real t, Real dt);
}  // namespace chemistry2
