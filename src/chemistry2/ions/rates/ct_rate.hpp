#ifndef CHEMISTRY2_IONS_RATES_CT_RATE_HPP_
#define CHEMISTRY2_IONS_RATES_CT_RATE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ct_rate.hpp
//! \brief Charge transfer with hydrogen.
//!
//!   An ion and a hydrogen atom exchange an electron without a photon or a
//!   free electron taking part, so the rate does not carry the electron
//!   density that recombination and collisional ionization do.  In gas that is
//!   partly neutral this beats radiative recombination for several ions,
//!   because a neutral hydrogen atom is far more abundant than a free electron
//!   there.  Oxygen is the extreme case: O II and H II are nearly resonant, so
//!   the two charge states lock together.
//!
//!   Both directions are fit by Kingdon & Ferland (1996), ApJS 106, 205, as
//!
//!     k(T) = a 1e-9 (T4)^b [1 + c exp(d T4)],   T4 = T/1e4,
//!
//!   with T clamped to the range each fit was made for.  The ionizing
//!   direction carries an extra Boltzmann factor for reactions that have to
//!   climb an energy barrier.
//!
//!   Hydrogen is the only collider.  The helium and H2 collider tables exist
//!   in the same distribution.
//!
//!   Helium as the transferring ion, rather than as a collider, is incomplete
//!   here.  Cloudy adds a second term on top of the table row for
//!   `He+ + H0 -> He0 + H+` (`atmdat_char_tran.cpp`, in ChargTranEval),
//!   attributing the table to the radiative transfer of Stancil et al. (1998)
//!   and the addition to Kingdon & Ferland.  The two share `b`, `c` and `d`
//!   exactly and differ only in amplitude, 1.87e-6 against 7.47e-6, so
//!   Cloudy's helium rate is about five times what this returns.  Nothing
//!   tracks helium yet.  Resolve it against Kingdon & Ferland before anything
//!   does, rather than by copying either number.

// C++ headers
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "athena.hpp"

namespace Rates {

//! Elements the Cloudy tables carry, hydrogen through zinc.
constexpr int kCtNElement = 30;

//! Charge states per element, `q = 0` to `q = 3`.  Above this Kingdon &
//! Ferland have no fit and Cloudy substitutes Dalgarno's estimate.
constexpr int kCtNStage = 4;

//! Dalgarno's estimate for charge transfer recombination onto an ion too
//! highly charged for the fits, `1.92e-9 (q+1)` cm^3 s^-1.  Cloudy's default
//! for `set charge transfer`.
constexpr Real kCtDalgarno = 1.92e-9;

//----------------------------------------------------------------------------------------
//! \class ChargeTransferRate
//! \brief Charge transfer between an ion and atomic or ionized hydrogen.
//!
//!   Indexed the way every other table in this directory is: `N` is the
//!   electron count of the reacting ion, counted before the reaction.  So the
//!   recombining ion of `CtRec` is the one with `N` electrons, exactly as in
//!   `RecombRate`, and the ionizing ion of `CtIon` is the one with `N`
//!   electrons, exactly as in `CollIonRate`.  Assembling the rates for one
//!   charge state therefore never reaches for a neighbour's index:
//!
//!     destruction by recombination = alpha_rr(Z,N) + alpha_dr(Z,N)
//!                                    + CtRec(Z,N) x_HI/x_e
//!     destruction by ionization    = k_ci(Z,N) + CtIon(Z,N) x_HII/x_e
//!
//!   Cloudy indexes both directions by the lower charge state of the pair
//!   instead, so its recombination rate is labelled by the product.  That is
//!   self-consistent, naming a transition rather than a reactant, but it puts
//!   the two rates a given charge state needs at two different indices.  The
//!   conversion happens once, in Load.
class ChargeTransferRate {
 public:
  //! \brief Read the recombination table, `X^(q+1) + H0 -> X^q + H+`.
  void LoadRecomb(const std::string &fname) { Load(fname, &rec_, 7); }

  //! \brief Read the ionization table, `X^q + H+ -> X^(q+1) + H0`.
  void LoadIon(const std::string &fname) { Load(fname, &ion_, 8); }

  //! \brief Whether a fit exists for the ion with Z protons and N electrons
  //!        recombining by charge transfer.
  bool HasRec(int z, int n) const { return Lookup(rec_, z, n) != nullptr; }

  //! \brief Whether a fit exists for that ion being ionized by charge transfer.
  bool HasIon(int z, int n) const { return Lookup(ion_, z, n) != nullptr; }

  //! \brief Rate coefficient [cm^3 s^-1] for `X^(q+1) + H0 -> X^q + H+`, where
  //!        `X^(q+1)` is the ion with Z protons and N electrons.
  //!
  //!   Above the four charge states the fits cover, Dalgarno's estimate stands
  //!   in, which is what Cloudy does rather than returning zero.  A charge
  //!   transfer rate of order 1e-9 cm^3 s^-1 is close to the geometric limit
  //!   for these reactions, so the estimate is a plausible number rather than
  //!   a placeholder.
  Real CtRec(int z, int n, Real temp) const {
    const int q = z - n - 1;    // charge of the recombining ion, X^(q+1)
    if (q < 0) return 0.0;
    if (q >= kCtNStage) return kCtDalgarno*(q + 1);
    const Fit *fit = Lookup(rec_, z, n);
    if (fit == nullptr) return 0.0;
    return Evaluate(*fit, temp);
  }

  //! \brief Rate coefficient [cm^3 s^-1] for `X^q + H+ -> X^(q+1) + H0`, where
  //!        `X^q` is the ion with Z protons and N electrons.
  //!
  //!   Zero outside the four charge states the fits cover.  There is no
  //!   counterpart to Dalgarno's estimate in this direction, and Cloudy has
  //!   none either.
  Real CtIon(int z, int n, Real temp) const {
    const Fit *fit = Lookup(ion_, z, n);
    if (fit == nullptr) return 0.0;
    return Evaluate(*fit, temp);
  }

  //! \brief Energy released [eV] by the recombining reaction, for the heating
  //!        term.  Negative where the reaction takes energy from the gas.
  Real RecEnergy(int z, int n) const {
    const Fit *fit = Lookup(rec_, z, n);
    return (fit == nullptr) ? 0.0 : fit->de_ev;
  }

  //! \brief Energy released [eV] by the ionizing reaction.
  Real IonEnergy(int z, int n) const {
    const Fit *fit = Lookup(ion_, z, n);
    return (fit == nullptr) ? 0.0 : fit->de_ev;
  }

 private:
  struct Fit {
    Real a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    Real t_min = 0.0, t_max = 0.0;
    Real barrier = 0.0;   //!< [1e4 K]; zero in the recombination table
    Real de_ev = 0.0;
  };

  static int Key(int z, int n) { return z*128 + n; }

  //! \brief Evaluate the fit.
  //!
  //!   The power law is held constant outside the range the fit covers, which
  //!   is what Cloudy's `HCTIon` and `HCTRecom` do, while the Boltzmann factor
  //!   keeps the unclamped temperature so that a reaction with a barrier still
  //!   switches off as the gas gets cold instead of freezing at its value at
  //!   `t_min`.
  static Real Evaluate(const Fit &f, Real temp) {
    Real t = std::max(temp, f.t_min);
    t = std::min(t, f.t_max)*1.0e-4;
    Real k = f.a*1.0e-9*std::pow(t, f.b)*(1.0 + f.c*std::exp(f.d*t));
    if (f.barrier > 0.0) k *= std::exp(-f.barrier*1.0e4/temp);
    return k;
  }

  const Fit *Lookup(const std::map<int, Fit> &tab, int z, int n) const {
    std::map<int, Fit>::const_iterator it = tab.find(Key(z, n));
    return (it == tab.end()) ? nullptr : &it->second;
  }

  //! \brief Read one of the two tables.
  //!
  //!   The rows carry no labels.  They run over thirty elements from hydrogen,
  //!   four charge states each, `q` being the lower member of the pair in both
  //!   files, and are converted here to the electron count of the reacting
  //!   ion: `X^(q+1)` for recombination, `X^q` for ionization.  Rows whose
  //!   first coefficient is zero carry no fit and are dropped rather than
  //!   stored, so a lookup that misses says so.
  void Load(const std::string &fname, std::map<int, Fit> *tab, int ncol) {
    std::ifstream f(fname.c_str());
    if (!f) {
      std::stringstream msg;
      msg << "ChargeTransferRate: cannot open table '" << fname << "'.";
      throw std::runtime_error(msg.str());
    }
    const bool is_ion = (ncol == 8);
    std::string line;
    int row = -1;   // -1 while the version stamp is still ahead
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      Real v[8];
      if (!(is >> v[0])) continue;
      if (row < 0) { row = 0; continue; }   // the version stamp, one number
      bool ok = true;
      for (int i = 1; i < ncol && ok; ++i) ok = static_cast<bool>(is >> v[i]);
      if (!ok) continue;

      const int z = row/kCtNStage + 1;
      const int q = row%kCtNStage;
      ++row;
      if (v[0] <= 0.0) continue;            // no fit for this charge state

      Fit fit;
      fit.a = v[0]; fit.b = v[1]; fit.c = v[2]; fit.d = v[3];
      fit.t_min = v[4]; fit.t_max = v[5];
      fit.barrier = is_ion ? v[6] : 0.0;
      fit.de_ev = is_ion ? v[7] : v[6];

      // The reacting ion: X^q when it is being ionized, X^(q+1) when it is
      // recombining.
      const int n = is_ion ? (z - q) : (z - q - 1);
      if (n >= 0) (*tab)[Key(z, n)] = fit;
    }
    if (tab->empty()) {
      std::stringstream msg;
      msg << "ChargeTransferRate: no fits found in '" << fname << "'.";
      throw std::runtime_error(msg.str());
    }
  }

  std::map<int, Fit> rec_, ion_;
};

}  // namespace Rates

#endif  // CHEMISTRY2_IONS_RATES_CT_RATE_HPP_
