#ifndef CHEMISTRY2_IONS_RATES_ION_THRESHOLD_HPP_
#define CHEMISTRY2_IONS_RATES_ION_THRESHOLD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion_threshold.hpp
//! \brief Ionization energies of ground-state atoms and ions: the threshold E_th of the
//!        outer-shell photoionization fits of Verner, Ferland, Korista & Yakovlev (1996),
//!        ApJ 465, 487, Table 1 (inputs/chemistry2/ions/verner96_photx.dat; columns Z,
//!        N, E_th [eV], ...). Z is the atomic number and N the electron count before
//!        ionization, so N = Z is the neutral atom.

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "athena.hpp"
#include "units/units.hpp"

namespace chemistry2 {
namespace ions {

//! Ionization thresholds (energies) per ion, read from the ion data directory
class IonThreshold {
 public:
  void Load(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("IonThreshold: cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      int z, n;
      Real e_th;
      if (!(is >> z >> n >> e_th)) continue;
      ev_[{z, n}] = e_th;
    }
    if (ev_.empty()) throw std::runtime_error("IonThreshold: no rows in " + path);
  }

  bool Has(const int z, const int n) const { return ev_.count({z, n}) > 0; }

  //! Ionization energy [eV] of the species with Z protons and N electrons
  Real Ev(const int z, const int n) const {
    const auto it = ev_.find({z, n});
    if (it == ev_.end()) {
      throw std::runtime_error("IonThreshold: no entry for Z = " + std::to_string(z) +
                               ", N = " + std::to_string(n));
    }
    return it->second;
  }

  //! Ionization energy [erg]
  Real Erg(const int z, const int n) const {
    return Ev(z, n)*units::Units::electron_volt_cgs;
  }

 private:
  std::map<std::pair<int, int>, Real> ev_;
};

}  // namespace ions
}  // namespace chemistry2

#endif  // CHEMISTRY2_IONS_RATES_ION_THRESHOLD_HPP_
