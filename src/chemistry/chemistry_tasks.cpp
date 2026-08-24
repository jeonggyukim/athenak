//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry_tasks.cpp
//! \brief functions that control Chemistry tasks stored in tasklists in
//! MeshBlockPack

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "chemistry/chemistry.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "ode_solvers/ode_solvers.hpp"
#include "pgen/pgen.hpp"
#include "tasklist/task_list.hpp"

namespace chemistry {

void Chemistry::AssembleChemistryTasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);  // indicator of no dependency for a given task

  // assemble "after_timeintegrator" task list. The ordering follows the hydro
  // "stagen" task list: update the conserved variables, communicate them into
  // the ghost cells, then recover the primitives over the whole grid
  // Post the non-blocking MPI receives for the post-chemistry ghost cell
  // exchange. This must be done before RecvU polls for the received data
  id.init_recv =
      tl["after_timeintegrator"]->AddTask(&Chemistry::InitRecv, this, none);

  // Do the chemistry integration on the active cells
  id.update_chemistry = tl["after_timeintegrator"]->AddTask(
      &Chemistry::UpdateChemistryTask, this, none);

  // Update the conserved variables with the new values from the primitive
  // arrays. The ghost cells are then refreshed by the exchange below
  id.prim_to_cons = tl["after_timeintegrator"]->AddTask(
      &Chemistry::PrimToCons, this, id.update_chemistry);

  // With SMR/AMR restrict the conserved variables into the coarse arrays so
  // that they can be packed and sent by SendU
  id.restrict_u = tl["after_timeintegrator"]->AddTask(&Chemistry::RestrictU,
                                                      this, id.prim_to_cons);

  // Communicate the conserved variables that have been updated by the
  // chemistry integration into the ghost cells. SendU must wait for both the
  // conserved variables to be ready and the non-blocking receives to be
  // posted, so its dependency is the union of those two TaskIDs (bound to a
  // named local since AddTask takes the dependency by non-const reference)
  TaskID send_u_dep = id.restrict_u | id.init_recv;
  id.send_u =
      tl["after_timeintegrator"]->AddTask(&Chemistry::SendU, this, send_u_dep);
  id.recv_u =
      tl["after_timeintegrator"]->AddTask(&Chemistry::RecvU, this, id.send_u);

  // Re-apply the physical boundary conditions so that non-periodic boundary
  // ghost cells also see the updated energy and abundances
  id.bcs = tl["after_timeintegrator"]->AddTask(&Chemistry::ApplyPhysicalBCs,
                                               this, id.recv_u);

  // With SMR/AMR prolongate the conserved variables into fine/coarse
  // boundaries
  id.prolongate =
      tl["after_timeintegrator"]->AddTask(&Chemistry::Prolongate, this, id.bcs);

  // Update the primitive variables, over all cells including the ghost cells,
  // with the new values from the conserved arrays
  id.cons_to_prim = tl["after_timeintegrator"]->AddTask(&Chemistry::ConsToPrim,
                                                        this, id.prolongate);

  // Confirm that all the MPI communications have completed so that the
  // boundary buffers are safe to reuse in the next cycle
  id.csend = tl["after_timeintegrator"]->AddTask(&Chemistry::ClearSend, this,
                                                 id.cons_to_prim);
  id.crecv = tl["after_timeintegrator"]->AddTask(&Chemistry::ClearRecv, this,
                                                 id.csend);
}

namespace {
/*!
 * \brief Abort on an unrecognised <chemistry> ode_solver.
 *
 * \details Without this an unmatched selector falls through the dispatch and
 * the chemistry solve is silently skipped, which in a timing run looks like a
 * very fast solver rather than an absent one.
 */
void UnknownODESolver(const std::string &network, const std::string &solver) {
  std::cout << "### FATAL ERROR: <chemistry> ode_solver = '" << solver
            << "' is not a valid ODE solver for the '" << network
            << "' network. Please check the athinput file." << std::endl;
  std::exit(EXIT_FAILURE);
}
}  // namespace

/*!
 * \brief Selects the proper template of Chemistry::UpdateChemistry to call and
 * passes in the proper arguments
 */
TaskStatus Chemistry::UpdateChemistryTask(Driver* d, int stage) {
  const std::string network = my_pin->GetString("chemistry", "network");
  const std::string ode_solver = my_pin->GetString("chemistry", "ode_solver");

  if (network == "H2") {
    if (ode_solver == "forward_euler") {
      UpdateChemistry<ode_solvers::ForwardEuler, H2Network>();
    } else if (ode_solver == "kokkos_BDF") {
      UpdateChemistry<ode_solvers::KokkosBDF, H2Network>();
    } else {
      UnknownODESolver(network, ode_solver);
    }
  } else if (network == "GOW17") {
    if (ode_solver == "forward_euler") {
      UpdateChemistry<ode_solvers::ForwardEuler, GOW17Network>();
    } else if (ode_solver == "kokkos_BDF") {
      UpdateChemistry<ode_solvers::KokkosBDF, GOW17Network>();
    } else if (ode_solver == "semi_implicit_sweep") {
      UpdateChemistry<ode_solvers::SemiImplicitSweep, GOW17Network>();
    } else {
      UnknownODESolver(network, ode_solver);
    }
  }

  return TaskStatus::complete;
}

/*!
 * \brief Syncs the conserved array to the values in the primitive array.
 * Primarily intended to update the energy since the chemistry solve updates the
 * internal energy.
 */
TaskStatus Chemistry::PrimToCons(Driver* pdrive, int stage) {
  auto& indcs = pmy_pack->pmesh->mb_indcs;
  int& ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2 * ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng - 1) : 0;

  if (is_hydro_enabled) {
    auto peos = pmy_pack->phydro->peos;
    auto u0 = pmy_pack->phydro->u0;
    auto w0 = pmy_pack->phydro->w0;
    peos->PrimToCons(w0, u0, 0, n1m1, 0, n2m1, 0, n3m1);
  } else {  // if (is_mhd_enabled) {
    auto peos = pmy_pack->pmhd->peos;
    auto u0 = pmy_pack->pmhd->u0;
    auto bcc = pmy_pack->pmhd->bcc0;
    auto w0 = pmy_pack->pmhd->w0;
    peos->PrimToCons(w0, bcc, u0, 0, n1m1, 0, n2m1, 0, n3m1);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI),
//! and initialize all boundary receive status flags to waiting (with or without
//! MPI) for the post-chemistry ghost cell exchange

TaskStatus Chemistry::InitRecv(Driver* pdrive, int stage) {
  TaskStatus tstat;
  if (is_hydro_enabled) {
    auto phydro = pmy_pack->phydro;
    tstat = phydro->pbval_u->InitRecv(phydro->nhydro + phydro->nscalars);
  } else if (is_mhd_enabled) {
    auto pmhd = pmy_pack->pmhd;
    tstat = pmhd->pbval_u->InitRecv(pmhd->nmhd + pmhd->nscalars);
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::RestrictU
//! \brief Wrapper task list function to restrict the conserved variables into
//! the coarse arrays with SMR/AMR so they can be packed and sent by SendU

TaskStatus Chemistry::RestrictU(Driver* pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    if (is_hydro_enabled) {
      auto phydro = pmy_pack->phydro;
      pmy_pack->pmesh->pmr->RestrictCC(phydro->u0, phydro->coarse_u0);
    } else if (is_mhd_enabled) {
      auto pmhd = pmy_pack->pmhd;
      pmy_pack->pmesh->pmr->RestrictCC(pmhd->u0, pmhd->coarse_u0);
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved
//! variables

TaskStatus Chemistry::SendU(Driver* pdrive, int stage) {
  TaskStatus tstat;
  if (is_hydro_enabled) {
    tstat = pmy_pack->phydro->pbval_u->PackAndSendCC(
        pmy_pack->phydro->u0, pmy_pack->phydro->coarse_u0);
  } else if (is_mhd_enabled) {
    tstat = pmy_pack->pmhd->pbval_u->PackAndSendCC(pmy_pack->pmhd->u0,
                                                   pmy_pack->pmhd->coarse_u0);
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved
//! variables

TaskStatus Chemistry::RecvU(Driver* pdrive, int stage) {
  TaskStatus tstat;
  if (is_hydro_enabled) {
    tstat = pmy_pack->phydro->pbval_u->RecvAndUnpackCC(
        pmy_pack->phydro->u0, pmy_pack->phydro->coarse_u0);
  } else if (is_mhd_enabled) {
    tstat = pmy_pack->pmhd->pbval_u->RecvAndUnpackCC(pmy_pack->pmhd->u0,
                                                     pmy_pack->pmhd->coarse_u0);
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::ApplyPhysicalBCs
//! \brief Wrapper task list function to re-apply the physical boundary
//! conditions after the chemistry update so that ghost cells at non-periodic
//! physical boundaries also see the updated energy and abundances

TaskStatus Chemistry::ApplyPhysicalBCs(Driver* pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  if (is_hydro_enabled) {
    auto pbval_u = pmy_pack->phydro->pbval_u;
    pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), pmy_pack->phydro->u0);
  } else if (is_mhd_enabled) {
    auto pbval_u = pmy_pack->pmhd->pbval_u;
    pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), pmy_pack->pmhd->u0);
  }

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::Prolongate
//! \brief Wrapper task list function to prolongate the conserved (or
//! primitive) variables at fine/coarse boundaries with SMR/AMR. This mirrors
//! Hydro::Prolongate/MHD::Prolongate; the magnetic field is not touched by the
//! chemistry update so the face-centered arrays are left as the time
//! integrator computed them

TaskStatus Chemistry::Prolongate(Driver* pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    if (is_hydro_enabled) {
      auto phydro = pmy_pack->phydro;
      phydro->pbval_u->FillCoarseInBndryCC(phydro->u0, phydro->coarse_u0);
      if (pmy_pack->pmesh->pmr->prolong_prims) {
        phydro->pbval_u->ConsToPrimCoarseBndry(phydro->coarse_u0,
                                               phydro->coarse_w0);
        phydro->pbval_u->ProlongateCC(phydro->w0, phydro->coarse_w0);
        phydro->pbval_u->PrimToConsFineBndry(phydro->w0, phydro->u0);
      } else {
        phydro->pbval_u->ProlongateCC(phydro->u0, phydro->coarse_u0);
      }
    } else if (is_mhd_enabled) {
      auto pmhd = pmy_pack->pmhd;
      pmhd->pbval_u->FillCoarseInBndryCC(pmhd->u0, pmhd->coarse_u0);
      if (pmy_pack->pmesh->pmr->prolong_prims) {
        pmhd->pbval_u->ConsToPrimCoarseBndry(pmhd->coarse_u0, pmhd->coarse_b0,
                                             pmhd->coarse_w0);
        pmhd->pbval_u->ProlongateCC(pmhd->w0, pmhd->coarse_w0);
        pmhd->pbval_u->PrimToConsFineBndry(pmhd->w0, pmhd->b0, pmhd->u0);
      } else {
        pmhd->pbval_u->ProlongateCC(pmhd->u0, pmhd->coarse_u0);
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::ConsToPrim
//! \brief Wrapper task list function to update the primitive variables, over
//! all cells including the ghost cells, with the new values from the conserved
//! arrays after the ghost cell exchange

TaskStatus Chemistry::ConsToPrim(Driver* pdrive, int stage) {
  auto& indcs = pmy_pack->pmesh->mb_indcs;
  int& ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2 * ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng - 1) : 0;

  if (is_hydro_enabled) {
    auto phydro = pmy_pack->phydro;
    phydro->peos->ConsToPrim(phydro->u0, phydro->w0, false, 0, n1m1, 0, n2m1, 0,
                             n3m1);
  } else {  // if (is_mhd_enabled) {
    auto pmhd = pmy_pack->pmhd;
    pmhd->peos->ConsToPrim(pmhd->u0, pmhd->b0, pmhd->w0, pmhd->bcc0, false, 0,
                           n1m1, 0, n2m1, 0, n3m1);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::ClearSend
//! \brief Wrapper task list function that checks all MPI sends from the
//! chemistry ghost cell exchange have completed so the buffers are safe to
//! reuse in the next cycle

TaskStatus Chemistry::ClearSend(Driver* pdrive, int stage) {
  TaskStatus tstat;
  if (is_hydro_enabled) {
    tstat = pmy_pack->phydro->pbval_u->ClearSend();
  } else if (is_mhd_enabled) {
    tstat = pmy_pack->pmhd->pbval_u->ClearSend();
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Chemistry::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives from the
//! chemistry ghost cell exchange have completed so the buffers are safe to
//! reuse in the next cycle

TaskStatus Chemistry::ClearRecv(Driver* pdrive, int stage) {
  TaskStatus tstat;
  if (is_hydro_enabled) {
    tstat = pmy_pack->phydro->pbval_u->ClearRecv();
  } else if (is_mhd_enabled) {
    tstat = pmy_pack->pmhd->pbval_u->ClearRecv();
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }

  return tstat;
}

}  // namespace chemistry
