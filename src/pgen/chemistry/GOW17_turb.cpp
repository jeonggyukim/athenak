//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file GOW17_turb.cpp
//! \brief Driven turbulent box with the GOW17 network, after Gong et al. (2023)
//! section 4.5.
//!
//! The initial state is uniform and at rest; all structure comes from the
//! turbulence driver, which is enabled by the presence of a <turb_driving>
//! block and needs nothing from the problem generator. Unlike GOW17_uniform,
//! which holds every cell in the same state so that a substep count measured on
//! it is divergence-free, this problem deliberately produces a multiphase
//! medium: cells within one warp land in different chemical and thermal
//! regimes, and the adaptive controller has to disagree across the warp.
//!
//! The initial temperature is set directly rather than through an isothermal
//! sound speed, since a driven box is adiabatic and its temperature is a
//! physical initial condition rather than a property of the equation of state.

#include <cmath>
#include <string>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GOW17Turb()
//! \brief Uniform, static initial state for a driven turbulent box.

void ProblemGenerator::GOW17Turb(ParameterInput* pin, const bool restart) {
  if (restart) return;

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

  // As in GOW17_uniform: with no <chemistry> block the module is absent and
  // this sets up the same state for a pure hydro run, which is what isolates
  // chemistry cost from hydro cost.
  const bool chemistry_on = (pmbp->pchemistry != nullptr);

  const Real n_H = pin->GetReal("problem", "n_H");
  const Real T_init = pin->GetOrAddReal("problem", "T_init", 1.0e4);
  const Real mu_H = chemistry_on ? pmbp->pchemistry->mu_H
                                 : pin->GetOrAddReal("problem", "mu_H", 1.4);
  const Real dens = n_H * pmbp->punit->hydrogen_mass_cgs * mu_H /
                    pmbp->punit->density_cgs();

  // Internal energy per unit volume in code units, e = n k T / (gamma - 1).
  // The particle count per H follows the initial abundances; the gas starts
  // atomic, so 1.1 (H + He) is the right multiplier and the chemistry moves it
  // from there.
  const Real n_particle_per_H = 1.1;
  const Real kB = pmbp->punit->k_boltzmann_cgs;
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  const Real eint = n_H * n_particle_per_H * kB * T_init / gm1 /
                    pmbp->punit->pressure_cgs();

  DualArray1D<Real> initial_chemistry("initial_chemistry",
                                      chemistry::GOW17Network::neqs - 1);
  if (chemistry_on) {
    const Real init_default = pin->GetOrAddReal("problem", "init_default", 0.0);
    for (size_t i = 0; i < chemistry::GOW17Network::neqs - 1; i++) {
      const auto name = chemistry::GOW17Network::species_names[i];
      const auto init_name = std::string("init_") + std::string(name);
      initial_chemistry.view_host()(i) =
          pin->GetOrAddReal("problem", init_name, init_default);
    }
  }
  initial_chemistry.modify_host();
  initial_chemistry.sync_device();
  auto initial_chemistry_d = initial_chemistry.view_device();

  const int chem_start =
      chemistry_on ? pmbp->pchemistry->get_chemistry_scalars_first_idx() : 0;

  par_for(
      "pgen_GOW17_turb", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
      js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        w0(m, IDN, k, j, i) = dens;
        w0(m, IVX, k, j, i) = 0.0;
        w0(m, IVY, k, j, i) = 0.0;
        w0(m, IVZ, k, j, i) = 0.0;
        w0(m, IEN, k, j, i) = eint;

        if (chemistry_on) {
          for (size_t s = 0; s < chemistry::GOW17Network::neqs - 1; s++) {
            w0(m, chem_start + s, k, j, i) = initial_chemistry_d(s);
          }
        }
      });

  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
