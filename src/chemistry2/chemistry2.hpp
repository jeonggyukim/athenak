#ifndef CHEMISTRY2_CHEMISTRY2_HPP_
#define CHEMISTRY2_CHEMISTRY2_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry2.hpp
//! \brief Chemistry2 module: non-equilibrium chemistry and energy update in every cell,
//!        after the hydro time integrator, on the species stored as passive scalars.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "chemistry2/inputs/cell_inputs.hpp"
#include "chemistry2/network/gow17/gow17.hpp"
#include "chemistry2/network/h2.hpp"
#include "chemistry2/network/ncr/ncr.hpp"
#include "chemistry2/network/ncr/ncr_oxygen.hpp"
#include "chemistry2/solvers/kokkos_bdf.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"
#include "chemistry2/util/code_units.hpp"

class MeshBlockPack;
class Driver;

namespace chemistry2 {

// IDs of the tasks chemistry2 adds to the after_timeintegrator list
struct Chemistry2TaskIDs {
  TaskID init_recv, update, prim_to_cons, restrict_u, send_u, recv_u, bcs, prolongate,
      cons_to_prim, csend, crecv;
};

//! Result of Equilibrate over all ranks
struct EquilibriumStats {
  int not_converged;  // cells that reached nsub_max or t_max first
  int max_substeps;   // largest substep count of a converged cell
};

//! Settings of every registered network, read once; only the active one is used
struct NetworkSettings {
  H2::Settings h2;
  GOW17::Settings gow17;
  GOW17Analytic::Settings gow17_analytic;
  NCR::Settings ncr;
  NCROxygen<2>::Settings ncr_oxygen;
};

class Chemistry2 {
 public:
  //! Read <chemistry2> (network, ode_solver, settings, radiation inputs) and the units
  Chemistry2(MeshBlockPack *ppack, ParameterInput *pin);

  //! Number of passive scalars chemistry2 adds, from <chemistry2>/network
  static int NumScalars(ParameterInput *pin);

  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);

  int FirstScalar() const { return first_scalar_; }
  int LastScalar() const { return first_scalar_ + nscalars_ - 1; }
  //! Output name of the scalar with index n, which must lie in [FirstScalar, LastScalar]
  std::string SpeciesName(int n) const;
  Real MuH() const { return mu_h_; }

  //! Advance the chemistry of every cell of the pack over dt with one registered pair
  template <class Pair>
  void Update(Real t, Real dt);
  //! Advance the chemistry of every cell over dt with the active pair (outside the task
  //! list, e.g. to equilibrate an initial state)
  void Advance(Real dt);
  //! Solve every cell of the pack to equilibrium with one registered semi-implicit pair
  template <class Pair>
  EquilibriumStats EquilibrateCells(const EquilibriumSettings &q);
  //! Solve every cell to equilibrium with the active pair (semi-implicit only), e.g. to
  //! set an initial state; the counts are summed (maxed) over ranks
  EquilibriumStats Equilibrate(const EquilibriumSettings &q);

  // the tasks (chemistry2_tasks.cpp): the update over the hydro dt, then the
  // conserved variables and the ghost-cell exchange
  TaskStatus UpdateTask(Driver *pdrive, int stage);
  TaskStatus PrimToCons(Driver *pdrive, int stage);
  TaskStatus InitRecv(Driver *pdrive, int stage);
  TaskStatus RestrictU(Driver *pdrive, int stage);
  TaskStatus SendU(Driver *pdrive, int stage);
  TaskStatus RecvU(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *pdrive, int stage);
  TaskStatus Prolongate(Driver *pdrive, int stage);
  TaskStatus ConsToPrim(Driver *pdrive, int stage);
  TaskStatus ClearSend(Driver *pdrive, int stage);
  TaskStatus ClearRecv(Driver *pdrive, int stage);

  Chemistry2TaskIDs id;

 private:
  MeshBlockPack *const pmy_pack_;
  const bool is_mhd_;
  const int nscalars_;      // number of species (passive scalars) of the network
  const int first_scalar_;  // index of the first species in w0/u0
  const Real mu_h_;         // mean mass per H nucleus [m_H]
  const int active_;  // index of the (network, solver) pair in the registry
  const UnitFactors units_;  // code units <-> cgs, n_H per code density
  const SemiImplicitSettings semi_implicit_;
  const KokkosBDFSettings kokkos_bdf_;
  const NetworkSettings networks_;
  const InputSource inputs_;  // where the FUV field and CR rate of each cell come from
};

//! First chemistry scalar index and mu_H of whichever chemistry module exists, for the
//! problem generators shared by <chemistry> and <chemistry2>
int ChemistryScalarStart(MeshBlockPack *pmbp);
Real ChemistryMuH(MeshBlockPack *pmbp);

}  // namespace chemistry2

#endif  // CHEMISTRY2_CHEMISTRY2_HPP_
