//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry2_tasks.cpp
//! \brief Chemistry2 tasks in the "after_timeintegrator" task list: the update on the
//!        primitives, then the same ghost-cell exchange as the hydro stagen list.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "chemistry2/chemistry2.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "tasklist/task_list.hpp"

namespace chemistry2 {

//! Operator split: after the full hydro step, update the chemistry on w0, rebuild u0,
//! and refill the ghost cells (restrict, send/receive, physical BCs, prolongate) so the
//! next hydro step sees the new abundances and energy everywhere
void Chemistry2::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  auto &list = tl["after_timeintegrator"];
  id.init_recv = list->AddTask(&Chemistry2::InitRecv, this, none);
  // the chemistry update, then the conserved variables from the new primitives
  id.update = list->AddTask(&Chemistry2::UpdateTask, this, none);
  id.prim_to_cons = list->AddTask(&Chemistry2::PrimToCons, this, id.update);
  // ghost-cell exchange of u0, as in the hydro stagen list
  id.restrict_u = list->AddTask(&Chemistry2::RestrictU, this, id.prim_to_cons);
  TaskID send_dep = id.restrict_u | id.init_recv;  // AddTask takes a non-const reference
  id.send_u = list->AddTask(&Chemistry2::SendU, this, send_dep);
  id.recv_u = list->AddTask(&Chemistry2::RecvU, this, id.send_u);
  id.bcs = list->AddTask(&Chemistry2::ApplyPhysicalBCs, this, id.recv_u);
  id.prolongate = list->AddTask(&Chemistry2::Prolongate, this, id.bcs);
  // primitives again in all cells (ghosts included), then release the buffers
  id.cons_to_prim = list->AddTask(&Chemistry2::ConsToPrim, this, id.prolongate);
  id.csend = list->AddTask(&Chemistry2::ClearSend, this, id.cons_to_prim);
  id.crecv = list->AddTask(&Chemistry2::ClearRecv, this, id.csend);
}

namespace {
// Index limits of a MeshBlock including ghost cells
struct AllCells {
  int n1m1, n2m1, n3m1;
};

AllCells AllCellsOf(MeshBlockPack *pp) {
  const auto &indcs = pp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  return AllCells{indcs.nx1 + 2*ng - 1, (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0,
                  (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0};
}
}  // namespace

// u0 from w0 in all cells, ghosts included (hydro or MHD)
TaskStatus Chemistry2::PrimToCons(Driver *pdrive, int stage) {
  const AllCells a = AllCellsOf(pmy_pack_);
  if (is_mhd_) {
    auto pmhd = pmy_pack_->pmhd;
    pmhd->peos->PrimToCons(pmhd->w0, pmhd->bcc0, pmhd->u0, 0, a.n1m1, 0, a.n2m1, 0,
                           a.n3m1);
  } else {
    auto phydro = pmy_pack_->phydro;
    phydro->peos->PrimToCons(phydro->w0, phydro->u0, 0, a.n1m1, 0, a.n2m1, 0, a.n3m1);
  }
  return TaskStatus::complete;
}

// The tasks below call the hydro (or MHD) boundary routines on u0, with the scalars
// included, in the same order as the hydro stagen list.
TaskStatus Chemistry2::InitRecv(Driver *pdrive, int stage) {
  if (is_mhd_) {
    auto pmhd = pmy_pack_->pmhd;
    return pmhd->pbval_u->InitRecv(pmhd->nmhd + pmhd->nscalars);
  }
  auto phydro = pmy_pack_->phydro;
  return phydro->pbval_u->InitRecv(phydro->nhydro + phydro->nscalars);
}

// with mesh refinement: restrict u0 to the coarse buffer sent to coarser neighbours
TaskStatus Chemistry2::RestrictU(Driver *pdrive, int stage) {
  if (pmy_pack_->pmesh->multilevel) {
    if (is_mhd_) {
      pmy_pack_->pmesh->pmr->RestrictCC(pmy_pack_->pmhd->u0, pmy_pack_->pmhd->coarse_u0);
    } else {
      pmy_pack_->pmesh->pmr->RestrictCC(pmy_pack_->phydro->u0,
                                        pmy_pack_->phydro->coarse_u0);
    }
  }
  return TaskStatus::complete;
}

TaskStatus Chemistry2::SendU(Driver *pdrive, int stage) {
  if (is_mhd_) {
    auto pmhd = pmy_pack_->pmhd;
    return pmhd->pbval_u->PackAndSendCC(pmhd->u0, pmhd->coarse_u0);
  }
  auto phydro = pmy_pack_->phydro;
  return phydro->pbval_u->PackAndSendCC(phydro->u0, phydro->coarse_u0);
}

TaskStatus Chemistry2::RecvU(Driver *pdrive, int stage) {
  if (is_mhd_) {
    auto pmhd = pmy_pack_->pmhd;
    return pmhd->pbval_u->RecvAndUnpackCC(pmhd->u0, pmhd->coarse_u0);
  }
  auto phydro = pmy_pack_->phydro;
  return phydro->pbval_u->RecvAndUnpackCC(phydro->u0, phydro->coarse_u0);
}

// physical boundaries, then any user boundary function of the problem generator
TaskStatus Chemistry2::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  if (pmy_pack_->pmesh->strictly_periodic) return TaskStatus::complete;
  if (is_mhd_) {
    auto pbval_u = pmy_pack_->pmhd->pbval_u;
    pbval_u->HydroBCs(pmy_pack_, pbval_u->u_in, pmy_pack_->pmhd->u0);
  } else {
    auto pbval_u = pmy_pack_->phydro->pbval_u;
    pbval_u->HydroBCs(pmy_pack_, pbval_u->u_in, pmy_pack_->phydro->u0);
  }
  if (pmy_pack_->pmesh->pgen->user_bcs) {
    (pmy_pack_->pmesh->pgen->user_bcs_func)(pmy_pack_->pmesh);
  }
  return TaskStatus::complete;
}

// with mesh refinement: fill ghost cells next to coarser neighbours by prolongation,
// of the primitives or of the conserved variables as the mesh refinement is set up
TaskStatus Chemistry2::Prolongate(Driver *pdrive, int stage) {
  if (!pmy_pack_->pmesh->multilevel) return TaskStatus::complete;
  const bool prims = pmy_pack_->pmesh->pmr->prolong_prims;
  if (is_mhd_) {
    auto pmhd = pmy_pack_->pmhd;
    pmhd->pbval_u->FillCoarseInBndryCC(pmhd->u0, pmhd->coarse_u0);
    if (prims) {
      pmhd->pbval_u->ConsToPrimCoarseBndry(pmhd->coarse_u0, pmhd->coarse_b0,
                                           pmhd->coarse_w0);
      pmhd->pbval_u->ProlongateCC(pmhd->w0, pmhd->coarse_w0);
      pmhd->pbval_u->PrimToConsFineBndry(pmhd->w0, pmhd->b0, pmhd->u0);
    } else {
      pmhd->pbval_u->ProlongateCC(pmhd->u0, pmhd->coarse_u0);
    }
  } else {
    auto phydro = pmy_pack_->phydro;
    phydro->pbval_u->FillCoarseInBndryCC(phydro->u0, phydro->coarse_u0);
    if (prims) {
      phydro->pbval_u->ConsToPrimCoarseBndry(phydro->coarse_u0, phydro->coarse_w0);
      phydro->pbval_u->ProlongateCC(phydro->w0, phydro->coarse_w0);
      phydro->pbval_u->PrimToConsFineBndry(phydro->w0, phydro->u0);
    } else {
      phydro->pbval_u->ProlongateCC(phydro->u0, phydro->coarse_u0);
    }
  }
  return TaskStatus::complete;
}

// w0 from u0 in all cells, ghosts included
TaskStatus Chemistry2::ConsToPrim(Driver *pdrive, int stage) {
  const AllCells a = AllCellsOf(pmy_pack_);
  if (is_mhd_) {
    auto pmhd = pmy_pack_->pmhd;
    pmhd->peos->ConsToPrim(pmhd->u0, pmhd->b0, pmhd->w0, pmhd->bcc0, false, 0, a.n1m1,
                           0, a.n2m1, 0, a.n3m1);
  } else {
    auto phydro = pmy_pack_->phydro;
    phydro->peos->ConsToPrim(phydro->u0, phydro->w0, false, 0, a.n1m1, 0, a.n2m1, 0,
                             a.n3m1);
  }
  return TaskStatus::complete;
}

TaskStatus Chemistry2::ClearSend(Driver *pdrive, int stage) {
  return is_mhd_ ? pmy_pack_->pmhd->pbval_u->ClearSend()
                 : pmy_pack_->phydro->pbval_u->ClearSend();
}

TaskStatus Chemistry2::ClearRecv(Driver *pdrive, int stage) {
  return is_mhd_ ? pmy_pack_->pmhd->pbval_u->ClearRecv()
                 : pmy_pack_->phydro->pbval_u->ClearRecv();
}

}  // namespace chemistry2
