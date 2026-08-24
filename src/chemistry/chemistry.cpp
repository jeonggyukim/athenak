//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry.cpp
//! \brief implementation of Chemistry class constructor and assorted other
//! functions
#include "chemistry/chemistry.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ode_solvers/ode_solvers.hpp"

namespace chemistry {
//----------------------------------------------------------------------------------------
// Constructor, initializes data structures and parameters
//----------------------------------------------------------------------------------------
Chemistry::Chemistry(MeshBlockPack* ppack, ParameterInput* pin)
    : pmy_pack(ppack),
      is_hydro_enabled(pin->DoesBlockExist("hydro")),
      is_mhd_enabled(pin->DoesBlockExist("mhd")),
      nscalars_chemistry(SetupGetNumChemistryScalars(ppack, pin, -1, false)),
      mu_H(pin->GetOrAddReal("chemistry", "mu_H", 1.4)),
      report_substeps(
          pin->GetOrAddBoolean("chemistry", "report_substeps", false)),
      ode_substeps_total("ode_substeps_total"),
      ode_substeps_max("ode_substeps_max"),
      chemistry_scalars_first_idx(ComputeChemistryScalarsStartIndex()),
      my_pin(pin),
      pchem_rad(ppack, pin) {
  // Verify that units are enables
  if (!pin->DoesBlockExist("units")) {
    std::cerr
        << "### FATAL ERROR: The chemistry module requires that the units "
           "module be enabled. Please enable it in the athinput file."
        << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Read all chemistry inputs so that CheckUnusedParameters won't flag them
  const std::string network = pin->GetString("chemistry", "network");
  const std::string ode_solver = pin->GetString("chemistry", "ode_solver");
  if (network == "H2") {
    H2Network::GetSettings(pin, pmy_pack);
  } else if (network == "GOW17") {
    GOW17Network::GetSettings(pin, pmy_pack);
  }
  if (ode_solver == "forward_euler") {
    ode_solvers::ForwardEuler<H2Network>::GetSettings(pin, "chemistry");
  } else if (ode_solver == "kokkos_BDF") {
    ode_solvers::KokkosBDF<H2Network>::GetSettings(pin, "chemistry");
  }
}

//----------------------------------------------------------------------------------------
// Destructor, primarily frees memory
//----------------------------------------------------------------------------------------
Chemistry::~Chemistry() {}

// ================
// Member Functions
// ================
/*!
 * \brief Updates the chemistry scalars and internal energy
 *
 * \tparam ODE_Solver_t The ODE solver to ues
 * \tparam Network_t The chemistry network to use
 */
template <template <typename> class ODE_Solver_t, typename Network_t>
void Chemistry::UpdateChemistry() {
  // ------ Collect variables that we'll need -----
  // The primitive grid
  auto w0 = GetW0();
  // The time at the beginning of this timestep
  Real const t_start = pmy_pack->pmesh->time;
  // The timestep
  Real const dt = pmy_pack->pmesh->dt;
  // Cell sizes
  auto sizes = pmy_pack->pmb->mb_size;

  // ----- Get the unit conversions and constants we'll need -----
  Real const time_cgs = pmy_pack->punit->time_cgs();
  Real const energy_density_cgs = pmy_pack->punit->pressure_cgs();
  Real const density_cgs = pmy_pack->punit->density_cgs();
  Real const hydrogen_mass_cgs = pmy_pack->punit->hydrogen_mass_cgs;
  Real const gamma = pmy_pack->phydro->peos->eos_data.gamma;
  Real const mu_H_local = mu_H;

  // ----- Get radiation stuff -----
  const auto ir = pchem_rad.ir;

  // ----- Load network and ODE solver settings -----
  // Read the files and cache the results. Static variables have static storage
  // duration and can't be captured by lambdas, like Kokkos parallel regions. To
  // get around this the static cached variables are copied to local variables
  // with automatic duration that can be captured by a lambda
  static auto const ode_settings_cached =
      ODE_Solver_t<Network_t>::GetSettings(my_pin, "chemistry");
  static auto const network_settings_cached =
      Network_t::GetSettings(my_pin, pmy_pack);
  auto const ode_settings = ode_settings_cached;
  auto const network_settings = network_settings_cached;

  // ----- Get all the loop limits and generate the parallel policy ------
  // NOLINTNEXTLINE(whitespace/braces)
  auto const [start_limit, end_limit] = LoopLimitsAllCells();
  int const species_start_idx = chemistry_scalars_first_idx;
  auto const policy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
      DevExeSpace(), start_limit, end_limit);

  // ----- ODE substep diagnostics -----
  // Captured by value into the kernel, so they have to be locals rather than
  // members. Zeroed here because the counts are per cycle, not cumulative.
  bool const count_substeps = report_substeps;
  auto substeps_total = ode_substeps_total;
  auto substeps_max = ode_substeps_max;
  if (count_substeps) {
    Kokkos::deep_copy(substeps_total, 0.0);
    Kokkos::deep_copy(substeps_max, 0.0);
  }

  Kokkos::parallel_for(
      "Chemistry_ODE_Solve", policy,
      KOKKOS_LAMBDA(const int& mb_idx, const int& k, const int& j,
                    const int& i) {
        // Create the chemisty object
        Network_t chem_net(network_settings, mb_idx, k, j, i, w0, sizes, ir,
                           density_cgs, mu_H_local, gamma, hydrogen_mass_cgs,
                           time_cgs, energy_density_cgs);

        // ------ Load cell values ------
        // Chemistry scalars. The loop is based off of the chemical
        // network's number of equations since that's known at compile time,
        // enabling more loop optimizations. The minus 1 is because internal
        // energy occupies the last slot in the array
        int grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs - 1; s_idx++) {
          chem_net.y(s_idx) = w0(mb_idx, grid_idx, k, j, i);
          grid_idx += 1;
        }

        // Load internal energy
        chem_net.y(Network_t::IIE) = w0(mb_idx, IEN, k, j, i);

        // ------ Solve the ODEs ------
        ODE_Solver_t ode_solver(ode_settings, chem_net, t_start, dt);
        // ode_solvers::KokkosBDF solver(chem_net, t_start, dt);
        ode_solver.SolveODE();

        // How many internal steps that macro-step cost. One atomic pair per
        // cell, so this stays behind the input flag.
        if (count_substeps) {
          Real const n = static_cast<Real>(ode_solver.n_substeps);
          Kokkos::atomic_add(&substeps_total(), n);
          Kokkos::atomic_max(&substeps_max(), n);
        }

        // ------ Write cell values back out ------
        // Chemistry scalars
        grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs - 1; s_idx++) {
          w0(mb_idx, grid_idx, k, j, i) = chem_net.y(s_idx);
          grid_idx += 1;
        }

        // Write internal energy
        w0(mb_idx, IEN, k, j, i) = chem_net.y(Network_t::IIE);
      });

  if (count_substeps) {
    Real total = 0.0, max = 0.0;
    Kokkos::deep_copy(total, substeps_total);
    Kokkos::deep_copy(max, substeps_max);
    // Cell count from the same limits the kernel used, ghost zones included.
    std::uint64_t ncells = 1;
    for (int n = 0; n < 4; ++n) {
      ncells *= static_cast<std::uint64_t>(end_limit[n] - start_limit[n]);
    }
    std::cout << "chemistry substeps: total=" << static_cast<std::uint64_t>(total)
              << " mean=" << total / static_cast<Real>(ncells)
              << " max=" << static_cast<std::uint64_t>(max)
              << " cells=" << ncells << std::endl;
  }
}

// Instantiate the different versions of UpdateChemistry
template void
Chemistry::UpdateChemistry<ode_solvers::ForwardEuler, H2Network>();
template void Chemistry::UpdateChemistry<ode_solvers::KokkosBDF, H2Network>();
template void
Chemistry::UpdateChemistry<ode_solvers::ForwardEuler, GOW17Network>();
template void
Chemistry::UpdateChemistry<ode_solvers::KokkosBDF, GOW17Network>();
// The semi-implicit sweep needs CDRates/Edot/RenormalizeElements, which only
// GOW17 provides, so it is not instantiated for H2.
template void
Chemistry::UpdateChemistry<ode_solvers::SemiImplicitSweep, GOW17Network>();

/*!
 * \brief Return the name of the chemical species at scalar_idx
 *
 * \param scalar_idx The index of the chemistry scalar who's name is needed
 * \return std::string The name of the chemistry species
 */
std::string Chemistry::GetSpeciesNames(int const& scalar_idx) {
  // Only the first time this is called create the mapping between species names
  // and grid index
  static std::map<int, std::string> species_names_map;
  if (species_names_map.size() == 0) {
    // std::vector of scalar names
    const std::string network = my_pin->GetString("chemistry", "network");
    std::vector<std::string_view> species_names;
    if (network == "H2") {
      species_names.assign(H2Network::species_names.begin(),
                           H2Network::species_names.end());
    } else if (network == "GOW17") {
      species_names.assign(GOW17Network::species_names.begin(),
                           GOW17Network::species_names.end());
    }

    // Create the mapping
    int name_idx = 0;
    for (size_t i = get_chemistry_scalars_first_idx();
         i < get_chemistry_scalars_last_idx() + 1; i++) {
      species_names_map[i] = "chem_" + std::string(species_names[name_idx]);
      name_idx++;
    }
  }

  // Verify that this is a chemistry scalar
  if (scalar_idx < get_chemistry_scalars_first_idx() ||
      scalar_idx > get_chemistry_scalars_last_idx()) {
    std::stringstream msg;
    msg << "Attempted to output the field at index " << scalar_idx
        << " as a passive scalar for the chemistry module but it is not one of "
           "the scalars managed by the chemistry module.";
    throw std::runtime_error(msg.str());
  }

  // Return the proper name
  return species_names_map[scalar_idx];
}

/*!
 * \brief Get the conserved array. Correctly sources the array from the hydro or
 * MHD classes.
 *
 * \return DvceArray5D<Real> The conserved array
 */
DvceArray5D<Real> Chemistry::GetU0() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->u0;
  } else {  // if (is_mhd_enabled) {
    return pmy_pack->pmhd->u0;
  }
}

/*!
 * \brief Get the primitive array. Correctly sources the array from the hydro or
 * MHD classes.
 *
 * \return DvceArray5D<Real> The primitive array
 */
DvceArray5D<Real> Chemistry::GetW0() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->w0;
  } else {  // if (is_mhd_enabled) {
    return pmy_pack->pmhd->w0;
  }
}

/*!
 * \brief Gets the first index for a chemistry scalar
 *
 * \return int
 */
int Chemistry::ComputeChemistryScalarsStartIndex() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->nhydro + nscalars_pre_chemistry;
  } else if (is_mhd_enabled) {
    return pmy_pack->pmhd->nmhd + nscalars_pre_chemistry;
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }
}

/*!
 * \brief Returns loop limits for the chemistry solver to use with
 * MDRangePolicy.
 *
 * \return std::tuple<Kokkos::Array<int, 4>, Kokkos::Array<int, 4>> The start
 * and end limits in that order
 */
std::tuple<Kokkos::Array<int, 4>, Kokkos::Array<int, 4>>
Chemistry::LoopLimitsAllCells() {
  // Set the start indices
  Kokkos::Array<int, 4> const start = {
      0,                             // meshblock start
      pmy_pack->pmesh->mb_indcs.ks,  // k start
      pmy_pack->pmesh->mb_indcs.js,  // j start
      pmy_pack->pmesh->mb_indcs.is   // i start
  };

  // Check if the dimension is active and if it's not set the upper limit to 1
  Kokkos::Array<int, 4> const end = {
      pmy_pack->nmb_thispack,            // meshblock end
      pmy_pack->pmesh->mb_indcs.ke + 1,  // k end
      pmy_pack->pmesh->mb_indcs.je + 1,  // j end
      pmy_pack->pmesh->mb_indcs.ie + 1   // i end
  };

  return {start, end};
}

}  // namespace chemistry
