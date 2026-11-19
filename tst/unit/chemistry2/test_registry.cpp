//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file test_registry.cpp
//! \brief The (network, solver) registry, the unit factors and the constant input source.

#include <string>
#include <type_traits>

#include "athena.hpp"
#include "check.hpp"
#include "chemistry2/inputs/cell_inputs.hpp"
#include "chemistry2/registry.hpp"
#include "chemistry2/unit_factors.hpp"

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  unit::Report report;

  // Registry
  report.Expect(chemistry2::Find("H2", "semi_implicit") == 0, "Find H2 + semi_implicit");
  report.Expect(chemistry2::Find("H2", "kokkos_BDF") == -1, "unknown solver gives -1");
  report.Expect(chemistry2::Find("GOW17", "semi_implicit") == 1,
                "Find GOW17 + semi_implicit");
  report.Expect(chemistry2::Find("NCR", "semi_implicit") == 4,
                "Find NCR + semi_implicit");
  report.Expect(chemistry2::Find("NL99", "semi_implicit") == -1,
                "unknown network gives -1");
  report.Expect(chemistry2::Find("h2", "semi_implicit") == -1,
                "names are case-sensitive");
  report.Expect(chemistry2::Find("GOW17_analytic", "semi_implicit") == 2,
                "Find GOW17_analytic + semi_implicit");
  report.Expect(chemistry2::Find("GOW17", "kokkos_BDF") == 3, "Find GOW17 + kokkos_BDF");
  report.Expect(chemistry2::List() == "H2 + semi_implicit, GOW17 + semi_implicit, "
                                      "GOW17_analytic + semi_implicit, "
                                      "GOW17 + kokkos_BDF, NCR + semi_implicit",
                "List: " + chemistry2::List());
  report.Expect(chemistry2::SpeciesCount("GOW17") == 12 &&
                chemistry2::SpeciesCount("H2") == 2 &&
                chemistry2::SpeciesCount("NCR") == 3 &&
                chemistry2::SpeciesCount("NL99") == -1, "SpeciesCount");

  bool called_h2 = false;
  const bool found = chemistry2::Dispatch(0, [&](auto pair) {
    using P = decltype(pair);
    called_h2 = std::is_same_v<typename P::Network, chemistry2::H2> &&
                std::is_same_v<typename P::Integrator,
                               chemistry2::SemiImplicit<chemistry2::H2>>;
  });
  report.Expect(found && called_h2,
                "Dispatch(0) calls f with the H2 + semi_implicit pair");
  bool called = false;
  report.Expect(!chemistry2::Dispatch(chemistry2::NUM_PAIRS, [&](auto) { called = true; })
                && !called,
                "Dispatch of an unregistered index calls nothing");

  // Unit factors: each factor times its inverse is 1 to rounding
  const chemistry2::UnitFactors u =
      chemistry2::MakeUnitFactors(2.3e-24, 3.15576e13, 1.7e-12, 1.4);
  report.ExpectNear(u.nh_per_density*u.density_per_nh, 1.0, 2.0e-16, 0.0,
                    "nh_per_density times density_per_nh");
  report.ExpectNear(u.energy_cgs*u.energy_per_cgs, 1.0, 2.0e-16, 0.0,
                    "energy_cgs times energy_per_cgs");
  report.ExpectNear(u.nh_per_density, 2.3e-24/(1.4*units::Units::hydrogen_mass_cgs),
                    1.0e-15, 0.0, "nh_per_density = density_cgs/(mu_H m_H)");

  // Constant input source
  const chemistry2::InputSource src{1.5, 3.0e-16};
  const chemistry2::CellInputs in = src.Load(0, 1, 2, 3);
  report.Expect(in.g0 == 1.5 && in.xi_cr == 3.0e-16,
                "constant source returns its values");

  return report.ExitCode();
}
