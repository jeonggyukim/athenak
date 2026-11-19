#ifndef CHEMISTRY2_IONS_RATES_COLL_ION_RATE_HPP_
#define CHEMISTRY2_IONS_RATES_COLL_ION_RATE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file coll_ion_rate.hpp
//! \brief Collisional ionization rate coefficients.
//!
//!   A free electron carrying more than the ionization energy can remove a
//!   bound one.  The rate is therefore set by how much of the Maxwellian sits
//!   above threshold, which is the exp(-U) with U = dE/kT, and it is
//!   negligible until the electrons are within a factor of a few of the
//!   threshold temperature.  Hydrogen's 13.6 eV is 158000 K, so collisional
//!   ionization is a hot-gas process and photoionization is what ionizes
//!   photoionized gas.
//!
//!   Voronov (1997), ADNDT 65, 1, fits the rate as
//!
//!     k(T) = A (1 + P sqrt(U)) / (X + U) U^K exp(-U),
//!
//!   with five parameters per ion.

// C++ headers
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "athena.hpp"
#include "chemistry2/ions/rates/rate_table.hpp"

namespace Rates {

//! Above this the Maxwellian has nothing left above threshold and the fit is
//! taken as zero, which is where Voronov's own tabulation stops it.  At U = 80
//! the exponential alone is 1.8e-35.
constexpr Real kCollIonUMax = 80.0;

//! Ionization energy in electron volts, as a temperature: eV/k_B in kelvin.
constexpr Real kEvOverBoltzmann = 11604.518121550082;

//----------------------------------------------------------------------------------------
//! \class CollIonRate
//! \brief Electron-impact ionization of an ion with Z protons and N electrons,
//!   N counted before the electron is removed, so atomic hydrogen is
//!   Z = 1, N = 1.
class CollIonRate {
 public:
  //! \brief Read the fit parameters.
  //!
  //!   The file counts both Z and N from zero and puts the electron count
  //!   first, which is the Cloudy convention it is distributed under; the two
  //!   are converted here so that callers use the same (Z, N) as everywhere
  //!   else.  A row of -1 -1 ends the data, after which the file carries prose.
  void Load(const std::string &fname) {
    std::ifstream f(fname.c_str());
    if (!f) {
      std::stringstream msg;
      msg << "CollIonRate: cannot open table '" << fname << "'.";
      throw std::runtime_error(msg.str());
    }
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      int nm1, zm1;
      Fit fit;
      if (!(is >> nm1 >> zm1)) continue;   // the version stamp
      if (nm1 < 0 || zm1 < 0) break;       // the -1 -1 terminator
      if (!(is >> fit.de_ev >> fit.p >> fit.a >> fit.x >> fit.k)) continue;
      fit_[Key(zm1 + 1, nm1 + 1)] = fit;
    }
    if (fit_.empty()) {
      std::stringstream msg;
      msg << "CollIonRate: no fits found in '" << fname << "'.";
      throw std::runtime_error(msg.str());
    }
  }

  bool Has(int z, int n) const { return fit_.find(Key(z, n)) != fit_.end(); }

  //! \brief Ionization energy [eV].
  Real Threshold(int z, int n) const { return Find(z, n).de_ev; }

  //! \brief Rate coefficient [cm^3 s^-1] at temperature T [K].
  Real Rate(int z, int n, Real temp) const {
    const Fit &f = Find(z, n);
    const Real u = f.de_ev*kEvOverBoltzmann/temp;
    if (u > kCollIonUMax) return 0.0;
    return f.a*(1.0 + f.p*std::sqrt(u))/(f.x + u)*std::pow(u, f.k)
        *std::exp(-u);
  }

 private:
  struct Fit {
    Real de_ev = 0.0, p = 0.0, a = 0.0, x = 0.0, k = 0.0;
  };

  static int Key(int z, int n) { return z*128 + n; }

  const Fit &Find(int z, int n) const {
    std::map<int, Fit>::const_iterator it = fit_.find(Key(z, n));
    if (it == fit_.end()) {
      std::stringstream msg;
      msg << "CollIonRate: no fit for Z = " << z << ", N = " << n
          << ". The table must be loaded first, and N counts the electrons"
          << " before ionization, so atomic hydrogen is Z = 1, N = 1.";
      throw std::runtime_error(msg.str());
    }
    return it->second;
  }

  std::map<int, Fit> fit_;
};

}  // namespace Rates

#endif  // CHEMISTRY2_IONS_RATES_COLL_ION_RATE_HPP_
