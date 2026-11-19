//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry2.cpp
//! \brief Chemistry2 constructor, species names, the update task and the helpers shared
//!        with the problem generators.

#include "chemistry2/chemistry2.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "chemistry2/registry.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "units/units.hpp"

namespace chemistry2 {

namespace {
[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in chemistry2: " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

int FindPair(ParameterInput *pin) {
  const std::string net = pin->GetString("chemistry2", "network");
  const std::string solver = pin->GetString("chemistry2", "ode_solver");
  const int idx = Find(net, solver);
  if (idx < 0) {
    Fatal("<chemistry2> network = " + net + ", ode_solver = " + solver +
          " is not built; available: " + List());
  }
  return idx;
}

int FirstScalarIndex(MeshBlockPack *pp, ParameterInput *pin, bool is_mhd) {
  if (is_mhd) {
    return pp->pmhd->nmhd + pin->GetOrAddInteger("mhd", "nscalars", 0);
  }
  if (pp->phydro == nullptr) Fatal("needs a <hydro> or <mhd> block");
  return pp->phydro->nhydro + pin->GetOrAddInteger("hydro", "nscalars", 0);
}

UnitFactors ReadUnits(MeshBlockPack *pp, ParameterInput *pin, Real mu_h) {
  if (!pin->DoesBlockExist("units")) Fatal("needs a <units> block");
  return MakeUnitFactors(pp->punit->density_cgs(), pp->punit->time_cgs(),
                         pp->punit->pressure_cgs(), mu_h);
}
}  // namespace

Chemistry2::Chemistry2(MeshBlockPack *ppack, ParameterInput *pin)
    : pmy_pack_(ppack),
      is_mhd_(pin->DoesBlockExist("mhd")),
      nscalars_(NumScalars(pin)),
      first_scalar_(FirstScalarIndex(ppack, pin, is_mhd_)),
      mu_h_(pin->GetOrAddReal("chemistry2", "mu_H", 1.4)),
      active_(FindPair(pin)),
      units_(ReadUnits(ppack, pin, mu_h_)),
      semi_implicit_(SemiImplicitSettings::Read(pin, "chemistry2")),
      networks_{H2::ReadSettings(pin, "chemistry2")},
      inputs_(InputSource::Read(pin, "chemistry2")) {
  if (pin->DoesBlockExist("chemistry")) {
    Fatal("<chemistry> and <chemistry2> cannot both be used");
  }
  // The update evolves the internal energy w0(IEN), which only an ideal EOS has
  const bool ideal = is_mhd_ ? ppack->pmhd->peos->eos_data.is_ideal
                             : ppack->phydro->peos->eos_data.is_ideal;
  if (!ideal) Fatal("needs an ideal equation of state");
}

int Chemistry2::NumScalars(ParameterInput *pin) {
  const std::string net = pin->GetString("chemistry2", "network");
  const int n = SpeciesCount(net);
  if (n < 0) Fatal("<chemistry2> network = " + net + " is not built; available: " +
                   List());
  return n;
}

std::string Chemistry2::SpeciesName(const int n) const {
  std::string name;
  Dispatch(active_, [&](auto pair) {
    using Net = typename decltype(pair)::Network;
    name = "chem_" + std::string(Net::rows[n - first_scalar_].name);
  });
  return name;
}

TaskStatus Chemistry2::UpdateTask(Driver *pdrive, int stage) {
  // The networks do not depend on t; integrating from 0 keeps the substep clock exact
  // when t is large compared with the substeps.
  const Real dt = pmy_pack_->pmesh->dt;
  Dispatch(active_, [&](auto pair) { Update<decltype(pair)>(0.0, dt); });
  return TaskStatus::complete;
}

int ChemistryScalarStart(MeshBlockPack *pmbp) {
  if (pmbp->pchemistry2 != nullptr) return pmbp->pchemistry2->FirstScalar();
  return pmbp->pchemistry->get_chemistry_scalars_first_idx();
}

Real ChemistryMuH(MeshBlockPack *pmbp) {
  if (pmbp->pchemistry2 != nullptr) return pmbp->pchemistry2->MuH();
  return pmbp->pchemistry->mu_H;
}

}  // namespace chemistry2
