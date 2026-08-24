//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file GOW17_uniform.cpp
//! \brief Problem generator for chemistry problem with a uniform state using
//! the GOW17 network

#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GOW17_uniform()
//! \brief Problem Generator for the GOW17 test problem with a uniform state

void ProblemGenerator::GOW17Uniform(ParameterInput* pin, const bool restart) {
  if (restart) return;

  // capture variables for the kernel
  auto& indcs = pmy_mesh_->mb_indcs;
  int& is = indcs.is;
  int& ie = indcs.ie;
  int& js = indcs.js;
  int& je = indcs.je;
  int& ks = indcs.ks;
  int& ke = indcs.ke;
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;
  auto& w0 = pmbp->phydro->w0;
  auto& u0 = pmbp->phydro->u0;

  // Chemistry is optional here. With no <chemistry> block the module is never
  // constructed, and this problem generator sets up the same uniform state as a
  // pure hydro run, which is what separates chemistry cost from hydro cost in a
  // performance comparison.
  const bool chemistry_on = (pmbp->pchemistry != nullptr);

  // ----- Get the input parameters from the input file -----
  // Hydro values
  const Real n_H = pin->GetReal("problem", "n_H");
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  HydPrim1D hydro;
  // Matches the <chemistry> mu_H default so a hydro-only run has the same density.
  const Real mu_H = chemistry_on ? pmbp->pchemistry->mu_H
                                 : pin->GetOrAddReal("problem", "mu_H", 1.4);
  hydro.d = n_H * pmbp->punit->hydrogen_mass_cgs * mu_H /
            pmbp->punit->density_cgs();
  hydro.vx = pin->GetOrAddReal("problem", "vx_kms", 0.0);
  hydro.vy = 0.0;
  hydro.vz = 0.0;
  hydro.e = n_H * SQR(iso_cs) / (pmbp->phydro->peos->eos_data.gamma - 1.0);

  // Chemistry values. The array is always allocated (its length is a compile-time
  // constant) but is only filled and read when the chemistry module is present.
  DualArray1D<Real> initial_chemistry("initial_chemistry",
                                      chemistry::GOW17Network::neqs - 1);
  if (chemistry_on) {
    const Real init_default = pin->GetOrAddReal("problem", "init_default", 0.0);
    for (size_t i = 0; i < chemistry::GOW17Network::neqs - 1; i++) {
      // Determine the name in the parameter file
      const auto name = chemistry::GOW17Network::species_names[i];
      const auto init_name = std::string("init_") + std::string(name);

      // Get the value and save it
      const Real val = pin->GetOrAddReal("problem", init_name, init_default);
      initial_chemistry.view_host()(i) = val;
    }
  }

  // Copy intializing data to the device
  initial_chemistry.modify_host();
  initial_chemistry.sync_device();
  auto initial_chemistry_d = initial_chemistry.view_device();

  // Assign values
  const int chem_start =
      chemistry_on ? pmbp->pchemistry->get_chemistry_scalars_first_idx() : 0;
  par_for(
      "pgen_GOW17_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
      js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = hydro.d;
        w0(m, IVX, k, j, i) = hydro.vx;
        w0(m, IVY, k, j, i) = hydro.vy;
        w0(m, IVZ, k, j, i) = hydro.vz;
        w0(m, IEN, k, j, i) = hydro.e;

        // Assign chemistry values to this cell
        if (chemistry_on) {
          for (size_t s = 0; s < chemistry::GOW17Network::neqs - 1; s++) {
            w0(m, chem_start + s, k, j, i) = initial_chemistry_d(s);
          }
        }
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
