//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dump_tigris_ncr.cpp
//! \brief Tigris NCR rates and closures on random states, the reference of
//!        test_ncr_tigris.cpp.
//!
//! Not built by CMake: compiled against a staged Tigris tree (tigris-rayt
//! tst/unit/build/ncr/src, made by tigris-rayt tst/unit/run.py --config ncr):
//
//!   T=~/Projects/tigris-rayt; R=$T/tst/unit/build/ncr/src
//!   c++ -std=c++17 -O1 -I $R -I $T/tst/unit -DTIGRIS_SOURCE_DIR="\"$T/\"" \
//!     -o dump dump_tigris_ncr.cpp $R/parameter_input.cpp $R/globals.cpp \
//!     $R/utils/interp_table.cpp $R/microphysics/units.cpp $R/outputs/io_wrapper.cpp \
//!     $R/task_list/task_id.cpp && ./dump > tigris_ncr_states.txt
//
//! Writes, for 500 random states, the inputs and the Tigris NCR rates and closures
//! (interp_mode 0, Tigris defaults): C and D of H2 and H+ (s^-1), x_e after the update,
//! x_C+, x_CO. Radiation as in chemistry2's constant source: every chi = G, xi_diss_H2 =
//! 5.7e-11 G, no ionizing photons; Z_d = Z_g = 1.

#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "microphysics/units.hpp"
#include "parameter_input.hpp"
#include "photchem/ncr_rates.hpp"

int main() {
  ParameterInput *pin = new ParameterInput;
  std::istringstream is(std::string("<hydro>\ngamma = 1.6666666666666667\n"
      "<photchem_ncr>\ninterp_mode = 0\ncool_dust_flag = false\n"
      "temp_hot0 = 20000.0\ntemp_hot1 = 35000.0\ncoolftn_file = ") +
      TIGRIS_SOURCE_DIR + "inputs/tables/tigress_coolftn_ncr.txt\n");
  pin->LoadFromStream(is);
  Units units(pin);
  units.SetDensityConversion(pin, 1.4);
  NCRRates r;
  r.Init(pin);
  std::mt19937_64 rng(41);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::printf("# n_H T x_H2 x_H+ x_e G xi_CR | C_H2 D_H2 C_H+ D_H+ x_e_new x_C+ x_CO\n");
  for (int n = 0; n < 500; ++n) {
    NCRVar v{};
    v.nH = std::pow(10.0, -2.0 + 6.0*u(rng));
    v.temp = std::pow(10.0, 1.0 + 3.2*u(rng));  // 10 K to 1.6e4 K
    v.x_h2 = 0.49*u(rng);
    v.x_hii = (1.0 - 2.0*v.x_h2)*std::pow(10.0, -6.0*u(rng));
    v.x_hi = std::fmax(1.0 - 2.0*v.x_h2 - v.x_hii, 1e-20);
    v.x_e = v.x_hii + 1.6e-4*u(rng);
    const Real G = std::pow(10.0, -3.0 + 5.0*u(rng));
    v.chi_fuv = v.chi_pe = v.chi_lw = v.chi_h2 = v.chi_ci = v.chi_co = G;
    v.xi_ph_hi = v.xi_ph_h2 = 0.0;
    v.xi_diss_h2 = 5.7e-11*G;
    v.xi_cr = std::pow(10.0, -17.0 + 2.0*u(rng));
    v.z_d = v.z_g = 1.0;
    Real c_h2, d_h2, c_hii, d_hii, x_cii;
    r.EvaluateChemistryCD(v, c_h2, d_h2, c_hii, d_hii);
    const Real x_e_new = r.CalculateFreeElectron(v.x_e, v.x_hi, v.x_h2, v.x_hii, v.nH,
                                                 v.temp, v.z_d, v.z_g, v.xi_cr,
                                                 v.chi_fuv, v.chi_ci, &x_cii);
    const Real x_co = r.CalculateCOAbundance(v.x_h2, x_cii, v.x_hii*3.2e-4, v.nH, v.z_d,
                                             v.z_g, v.xi_cr, v.chi_co);
    std::printf("%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e "
                "%.17e %.17e %.17e\n", v.nH, v.temp, v.x_h2, v.x_hii, v.x_e, G, v.xi_cr,
                c_h2, d_h2, c_hii, d_hii, x_e_new, x_cii, x_co);
  }
  return 0;
}
