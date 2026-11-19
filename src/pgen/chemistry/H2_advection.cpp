//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file H2_advection.cpp
//! \brief Problem generator for chemistry problem that advects a gaussian state
//! using the H2 network that has an analytical solution

#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "chemistry2/chemistry2.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn void RefinementCondition()
//! Implements custom AMR refinement condition for the H2 advection problem
void RefinementCondition(MeshBlockPack* pmbp) {
  auto& refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  auto& indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto& multi_d = pmbp->pmesh->multi_d;
  auto& three_d = pmbp->pmesh->three_d;
  auto& w0 = pmbp->phydro->w0;

  // Set the refinement parameters
  const Real valmax = 0.2;
  const int field = chemistry2::ChemistryScalarStart(pmbp) + chemistry::H2Network::IH;

  // Mark blocks for refinement
  par_for_outer(
      "H2_Advection_RefinementCondition", DevExeSpace(), 0, 0, 0, (nmb - 1),
      KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
        Real team_qmax;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(tmember, nkji),
            [=](const int idx, Real& qmax) {
              int k = (idx) / nji;
              int j = (idx - k * nji) / nx1;
              int i = (idx - k * nji - j * nx1) + is;
              j += js;
              k += ks;
              qmax = Kokkos::fmax(w0(m, field, k, j, i), qmax);
            },
            Kokkos::Max<Real>(team_qmax));

        // only derefine when flag has not been set by other criteria
        int& flag = refine_flag.d_view(m + mbs);
        if (team_qmax > valmax) {
          flag = 1;
        }
        if ((team_qmax < valmax) && (flag == 0)) {
          flag = -1;
        }
      });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::H2_advection()
//! \brief Problem Generator for the H2 test problem that advects a gaussian
//! state
void ProblemGenerator::H2Advection(ParameterInput* pin, const bool restart) {
  user_ref_func = RefinementCondition;

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
  auto& size = pmbp->pmb->mb_size;
  auto& w0 = pmbp->phydro->w0;
  auto& u0 = pmbp->phydro->u0;

  // Get the input parameters from the input file
  const Real n_H = pin->GetReal("problem", "n_H");
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  HydPrim1D hydro;
  hydro.d = n_H * pmbp->punit->hydrogen_mass_cgs * chemistry2::ChemistryMuH(pmbp) /
            pmbp->punit->density_cgs();
  hydro.vx = pin->GetReal("problem", "vx_kms");
  hydro.vy = 0.0;
  hydro.vz = 0.0;
  hydro.e = hydro.d * SQR(iso_cs) / (pmbp->phydro->peos->eos_data.gamma - 1.0);
  const Real init_H = pin->GetOrAddReal("problem", "init_H", 0.0);

  // mean and std of the initial gaussian profile
  const Real gaussian_mean = pin->GetOrAddReal("problem", "gaussian_mean", 0.5);
  const Real gaussian_std = pin->GetOrAddReal("problem", "gaussian_std", 0.1);

  // Assign values
  const int chem_start = chemistry2::ChemistryScalarStart(pmbp);
  par_for(
      "pgen_H2_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js,
      je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = hydro.d;
        w0(m, IVX, k, j, i) = hydro.vx;
        w0(m, IVY, k, j, i) = hydro.vy;
        w0(m, IVZ, k, j, i) = hydro.vz;
        w0(m, IEN, k, j, i) = hydro.e;

        // Compute the location
        Real& x1min = size.d_view(m).x1min;
        Real& x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        const Real x = CellCenterX(i - is, nx1, x1min, x1max);

        // Compute the H and H2 abundances
        Real H2_abundance, H_abundance;
        if (x <= 1.0) {
          H_abundance =
              Kokkos::exp(-SQR(x - gaussian_mean) / (2. * SQR(gaussian_std)));
          H2_abundance = 0.5 * (1.0 - H_abundance);
        } else {
          H_abundance = init_H;
          H2_abundance = (1. - init_H) * 0.5;
        }

        // Assign chemistry values to this cell
        w0(m, chem_start + chemistry::H2Network::IH2, k, j, i) = H2_abundance;
        w0(m, chem_start + chemistry::H2Network::IH, k, j, i) = H_abundance;
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
