#ifndef CHEMISTRY_CHEMISTRY_HPP_
#define CHEMISTRY_CHEMISTRY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry.hpp
//  \brief definitions for Chemistry class

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "chemistry/network/chemistry_networks.hpp"
#include "chemistry/radiation.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

namespace chemistry {

//! \struct ChemistryTaskIDs
//  \brief container to hold TaskIDs of all chemistry tasks
struct ChemistryTaskIDs {
  TaskID init_recv;
  TaskID update_chemistry;
  TaskID prim_to_cons;
  TaskID restrict_u;
  TaskID send_u;
  TaskID recv_u;
  TaskID bcs;
  TaskID prolongate;
  TaskID cons_to_prim;
  TaskID csend;
  TaskID crecv;
};

//! \class Chemistry
class Chemistry {
 public:
  // ==========================
  // Constructor and Destructor
  // ==========================
  Chemistry(MeshBlockPack* ppack, ParameterInput* pin);
  ~Chemistry();

  // ================
  // Member Variables
  // ================
  // Container to hold names of TaskIDs
  ChemistryTaskIDs id;

  // The number of passive scalars used in the chemistry module
  int const nscalars_chemistry;

  // Mean molecular weight per hydrogen atom
  Real const mu_H;

  // ----- ODE substep diagnostics -----
  // Set from <chemistry> report_substeps. Off by default: the counters cost an
  // atomic per cell and the report costs a device-to-host copy every cycle.
  bool const report_substeps;
  // Total and maximum internal ODE steps over all cells in the last cycle.
  // Real rather than int so the atomics and the copy back match the rest of the
  // module; the counts are small enough to be exact in double.
  DvceArray0D<Real> ode_substeps_total;
  DvceArray0D<Real> ode_substeps_max;

  // ================
  // Member Functions
  // ================
  void AssembleChemistryTasks(
      std::map<std::string, std::shared_ptr<TaskList>> tl);

  /*!
   * \brief Updates the chemistry abundances and internal energy
   */
  TaskStatus UpdateChemistryTask(Driver* d, int stage);

  template <template <typename> class ODE_Solver_t, typename Network_t>
  void UpdateChemistry();

  /*!
   * \brief Updates the conserved grid with the updated values from the
   * primitive grid
   */
  TaskStatus PrimToCons(Driver* pdrive, int stage);

  /// Tasks for communicating ghost cells. InitRecv posts the non-blocking MPI
  /// receives and must run before SendU/RecvU; ClearSend/ClearRecv verify all
  /// communications have completed so the buffers are safe to reuse
  TaskStatus InitRecv(Driver* pdrive, int stage);
  TaskStatus SendU(Driver* pdrive, int stage);
  TaskStatus RecvU(Driver* pdrive, int stage);
  TaskStatus ClearSend(Driver* pdrive, int stage);
  TaskStatus ClearRecv(Driver* pdrive, int stage);

  /// Task to restrict the conserved variables into the coarse arrays with
  /// SMR/AMR, needed before they are packed and sent by SendU
  TaskStatus RestrictU(Driver* pdrive, int stage);

  /// Task to prolongate the conserved (or primitive) variables into fine/coarse
  /// boundaries with SMR/AMR
  TaskStatus Prolongate(Driver* pdrive, int stage);

  /// Task to re-apply physical boundary conditions after the chemistry update
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);

  /// Task to update the primitive grid, over all cells including the ghost
  /// cells, from the conserved grid after the ghost cell exchange
  TaskStatus ConsToPrim(Driver* pdrive, int stage);

  /*!
   * \brief Compute the number of required passive scalars that the chemistry
   * module will use
   *
   * \param ppack The MeshBlockPack instance in use
   * \param pin The ParameterInput instance in use
   * \param nscalars The current number of nscalars
   * \param chemistry_constructor Whether or not this is being called within the
   * Chemistry constructor. This should be left to its default value otherwise.
   * \return int The number of passive scalars that the chemistry module
   * requires
   */
  static int SetupGetNumChemistryScalars(
      MeshBlockPack* ppack, ParameterInput* pin, int const& nscalars,
      bool const not_in_chemistry_constructor = true) {
    // Capture the pre-chemistry number of passive scalars so that the Chemistry
    // constructor later knows where to start the indexing for Chemistry passive
    // scalars
    if (not_in_chemistry_constructor) {
      nscalars_pre_chemistry = nscalars;
    }

    const std::string network = pin->GetString("chemistry", "network");
    int num_species;
    // We need to take number of equations minus 1 since neqs includes the
    // internal energy equation
    if (network == "H2") {
      num_species = H2Network::neqs - 1;
    } else if (network == "GOW17") {
      num_species = GOW17Network::neqs - 1;
    }

    return num_species;
  }

  /*!
   * \brief Return the name of the chemical species at the grid field index
   * provided
   *
   * \param scalar_idx The index in the grid to the passive scalar in question
   * \return std::string The name of the chemical species
   */
  std::string GetSpeciesNames(int const& scalar_idx);

  // ===================
  // Getters and Setters
  // ===================
  int get_chemistry_scalars_first_idx() const {
    return chemistry_scalars_first_idx;
  }
  // The index of the final chemistry scalar
  int get_chemistry_scalars_last_idx() const {
    return chemistry_scalars_first_idx + nscalars_chemistry - 1;
  }

 private:
  // ptr to MeshBlockPack containing this chemistry. note that this is a const
  // pointers, the contents can be changed but the pointer address can't
  MeshBlockPack* const pmy_pack;

  ParameterInput* my_pin;

  // Chemistry radiation
  const chemistry::Radiation pchem_rad;

  // These indicate if hydro or MHD is in use
  bool const is_hydro_enabled;
  bool const is_mhd_enabled;

  // The number of regular passive scalars created before the chemistry scalars
  int inline static nscalars_pre_chemistry;

  // The beginning index of passive scalars reserved for chemisty
  int const chemistry_scalars_first_idx;

  // Get the correct conserved or primitive arrays array
  DvceArray5D<Real> GetU0();
  DvceArray5D<Real> GetW0();

  /*!
   * \brief Get loop limits for looping over all the cells, this includes both
   * real and ghost cells
   *
   * \return std::tuple<Kokkos::Array<int, 4>, Kokkos::Array<int, 4>> The
   * MDRangePolicy start and end
   */
  std::tuple<Kokkos::Array<int, 4>, Kokkos::Array<int, 4>> LoopLimitsAllCells();

  // Functions for setting up
  int ComputeChemistryScalarsStartIndex();
};
}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_HPP_
