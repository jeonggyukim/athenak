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
#include "chemistry2/chemistry2.hpp"
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

  // ----- Get the input parameters from the input file -----
  // Hydro values
  const Real n_H = pin->GetReal("problem", "n_H");
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  HydPrim1D hydro;
  hydro.d = n_H * pmbp->punit->hydrogen_mass_cgs * chemistry2::ChemistryMuH(pmbp) /
            pmbp->punit->density_cgs();
  hydro.vx = pin->GetOrAddReal("problem", "vx_kms", 0.0);
  hydro.vy = 0.0;
  hydro.vz = 0.0;
  hydro.e = n_H * SQR(iso_cs) / (pmbp->phydro->peos->eos_data.gamma - 1.0);

  // Chemistry values
  const Real init_default = pin->GetOrAddReal("problem", "init_default", 0.0);
  DualArray1D<Real> initial_chemistry("initial_chemistry",
                                      chemistry::GOW17Network::neqs - 1);
  for (size_t i = 0; i < chemistry::GOW17Network::neqs - 1; i++) {
    // Determine the name in the parameter file
    const auto name = chemistry::GOW17Network::species_names[i];
    const auto init_name = std::string("init_") + std::string(name);

    // Get the value and save it
    const Real val = pin->GetOrAddReal("problem", init_name, init_default);
    initial_chemistry.view_host()(i) = val;
  }

  // Copy intializing data to the device
  initial_chemistry.modify_host();
  initial_chemistry.sync_device();
  auto initial_chemistry_d = initial_chemistry.view_device();

  // Assign values
  // chemistry2's GOW17 stores the same species in the same order
  const int chem_start = chemistry2::ChemistryScalarStart(pmbp);
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
        for (size_t s = 0; s < chemistry::GOW17Network::neqs - 1; s++) {
          w0(m, chem_start + s, k, j, i) = initial_chemistry_d(s);
        }
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
