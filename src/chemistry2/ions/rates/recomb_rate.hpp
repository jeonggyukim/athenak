#ifndef CHEMISTRY2_IONS_RATES_RECOMB_RATE_HPP_
#define CHEMISTRY2_IONS_RATES_RECOMB_RATE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recomb_rate.hpp
//! \brief Radiative and dielectronic recombination rate coefficients.
//!
//!   An electron recombines with an ion two ways.  Radiative recombination
//!   emits the excess energy as a photon and proceeds at any energy.
//!   Dielectronic recombination captures the electron into a doubly excited
//!   state that then stabilizes radiatively, which requires the electron
//!   energy to match the capture resonance, so its rate rises steeply through
//!   the resonance region and falls away on either side.  That difference in
//!   shape is what decides how finely a table has to resolve temperature:
//!   radiative recombination is close to a power law and interpolates in
//!   log-log to round-off, while dielectronic recombination does not.
//!
//!   Both come from Badnell's fits, whose method is Badnell (2006), ApJS 167,
//!   334 for the radiative part and A&A 447, 1165 for the dielectronic, and
//!   whose coefficients are the TAMOC compilation carried in
//!   `inputs/tables/rates/badnell_*_2023.dat`.

// C++ headers
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "chemistry2/ions/rates/rate_table.hpp"

namespace Rates {

//! Most channels a dielectronic fit uses. The tables carry up to nine.
constexpr int kMaxDRChannels = 9;

//----------------------------------------------------------------------------------------
//! \class RecombRate
//! \brief Recombination of an ion with Z protons and N electrons, N counted
//!   before the electron is captured.
//!
//!   Rows are keyed by (Z, N, M), with M the initial metastable level; M = 1 is
//!   the ground state and is what a chemistry network wants.
class RecombRate {
 public:
  //! \brief Read the radiative fits and the two halves of the dielectronic
  //!        ones.  The coefficient and energy files carry the same rows in the
  //!        same order, so they are read together and paired.
  void Load(const std::string &fname_rr, const std::string &fname_dr_c,
            const std::string &fname_dr_e) {
    LoadRR(fname_rr);
    LoadDR(fname_dr_c, fname_dr_e);
  }

  bool HasRR(int z, int n, int m = 1) const {
    return rr_.find(Key(z, n, m)) != rr_.end();
  }
  bool HasDR(int z, int n, int m = 1) const {
    return dr_.find(Key(z, n, m)) != dr_.end();
  }

  //! \brief Radiative recombination coefficient [cm^3 s^-1] at temperature T
  //!        [K].
  //!
  //!   alpha = A / [ sqrt(T/T0) (1 + sqrt(T/T0))^(1-B)
  //!                             (1 + sqrt(T/T1))^(1+B) ],
  //!
  //!   with B replaced by B + C exp(-T2/T) for the low-charge ions whose rows
  //!   carry those two extra numbers.
  Real RadiativeRate(int z, int n, Real temp, int m = 1) const {
    const RRFit &f = FindRR(z, n, m);
    const Real s0 = std::sqrt(temp/f.t0);
    const Real s1 = std::sqrt(temp/f.t1);
    const Real b = f.modified ? f.b + f.c*std::exp(-f.t2/temp) : f.b;
    return f.a/(s0*std::pow(1.0 + s0, 1.0 - b)*std::pow(1.0 + s1, 1.0 + b));
  }

  //! \brief Dielectronic recombination coefficient [cm^3 s^-1] at temperature
  //!        T [K], as alpha = T^(-3/2) sum_m c_m exp(-E_m/T).
  Real DielectronicRate(int z, int n, Real temp, int m = 1) const {
    const DRFit &f = FindDR(z, n, m);
    Real sum = 0.0;
    for (int i = 0; i < f.nchan; ++i) sum += f.c[i]*std::exp(-f.e[i]/temp);
    return sum*std::pow(temp, -1.5);
  }

  //! \brief Case-B recombination coefficient of hydrogen [cm^3 s^-1], the fit
  //!        to Ferland that the NCR network uses.
  //!
  //!   Case B counts recombination to every level except the ground state.  A
  //!   photon from recombination to n = 1 is reabsorbed where it was emitted
  //!   and leaves the ionization unchanged, so an H II region calculation
  //!   wants case B, while the Badnell fit is case A and counts that channel.
  //!   The difference is not small: case A exceeds case B by 24 per cent at
  //!   10 K and by a factor of two near 3e5 K.
  static Real CaseBFerland(Real temp) {
    const Real tinv = 1.0/temp;
    const Real bb = 315614.0*tinv;   // 2 x 13.6 eV / k_B
    const Real cc = 115188.0*tinv;
    return 2.753e-14*std::pow(bb, 1.5)*std::pow(1.0 + std::pow(cc, 0.407),
                                                -2.242);
  }

  //! \brief Case-B recombination coefficient of hydrogen [cm^3 s^-1], Table
  //!        14.1 of Draine (2011).
  static Real CaseBDraine11(Real temp) {
    const Real t4 = temp*1.0e-4;
    return 2.54e-13*std::pow(t4, -0.8163 - 0.0208*std::log(t4));
  }

  //! Which hydrogen rate Rate returns.
  //!
  //!   kCaseBFerland is the default because the NCR network already uses it,
  //!   and a multi-ion network that extends NCR has to agree with it where the
  //!   two overlap; two hydrogen recombination rates in one run would be a
  //!   defect no test would show.  kCaseA returns the Badnell fit itself, for
  //!   a calculation that resolves the Lyman continuum instead of assuming it
  //!   is reabsorbed.
  enum class HydrogenCase { kCaseBFerland = 0, kCaseBDraine11 = 1, kCaseA = 2 };

  void SetHydrogenCase(HydrogenCase c) { hcase_ = c; }
  HydrogenCase GetHydrogenCase() const { return hcase_; }

  //! \brief Total recombination coefficient [cm^3 s^-1].
  //!
  //!   Hydrogen takes whichever case is selected.  A fully stripped ion of any
  //!   element has no bound electron to excite alongside the captured one, so
  //!   it has no dielectronic channel and the total is the radiative rate
  //!   alone; an ion absent from the dielectronic tables is treated the same
  //!   way.
  Real Rate(int z, int n, Real temp, int m = 1) const {
    if (z == 1) {
      switch (hcase_) {
        case HydrogenCase::kCaseBFerland:  return CaseBFerland(temp);
        case HydrogenCase::kCaseBDraine11: return CaseBDraine11(temp);
        case HydrogenCase::kCaseA:         break;
      }
    }
    const Real rr = RadiativeRate(z, n, temp, m);
    if (z == 1 || n == 0 || !HasDR(z, n, m)) return rr;
    return rr + DielectronicRate(z, n, temp, m);
  }

 private:
  struct RRFit {
    Real a = 0.0, b = 0.0, t0 = 0.0, t1 = 0.0, c = 0.0, t2 = 0.0;
    bool modified = false;
  };
  struct DRFit {
    Real c[kMaxDRChannels] = {};   // NOLINT (runtime/arrays)
    Real e[kMaxDRChannels] = {};   // NOLINT (runtime/arrays)
    int nchan = 0;
  };

  //! (Z, N, M) packed into one integer, the counts being small.
  static int Key(int z, int n, int m) { return (z*128 + n)*128 + m; }

  void LoadRR(const std::string &fname) {
    std::ifstream f(fname.c_str());
    if (!f) {
      std::stringstream msg;
      msg << "RecombRate: cannot open radiative table '" << fname << "'.";
      throw std::runtime_error(msg.str());
    }
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      int z, n, m, w;
      RRFit fit;
      if (!(is >> z >> n >> m >> w >> fit.a >> fit.b >> fit.t0 >> fit.t1)) {
        continue;   // the file's own header lines
      }
      // The low-charge rows carry two more numbers; the rest end here.
      if (is >> fit.c >> fit.t2) fit.modified = true;
      rr_[Key(z, n, m)] = fit;
    }
    if (rr_.empty()) {
      std::stringstream msg;
      msg << "RecombRate: no radiative fits found in '" << fname << "'.";
      throw std::runtime_error(msg.str());
    }
  }

  void LoadDR(const std::string &fname_c, const std::string &fname_e) {
    std::ifstream fc(fname_c.c_str()), fe(fname_e.c_str());
    if (!fc || !fe) {
      std::stringstream msg;
      msg << "RecombRate: cannot open dielectronic tables '" << fname_c
          << "' and '" << fname_e << "'.";
      throw std::runtime_error(msg.str());
    }
    std::string lc, le;
    while (std::getline(fc, lc)) {
      if (lc.empty() || lc[0] == '#') continue;
      // Advance the energy file to its next data line, so the two stay paired.
      do {
        if (!std::getline(fe, le)) {
          throw std::runtime_error(
              "RecombRate: the dielectronic energy table ran out before the "
              "coefficient table; the two must carry the same rows.");
        }
      } while (le.empty() || le[0] == '#');

      std::istringstream ic(lc), ie(le);
      int zc, nc, mc, wc, ze, ne, me, we;
      if (!(ic >> zc >> nc >> mc >> wc)) continue;
      if (!(ie >> ze >> ne >> me >> we)) continue;
      if (zc != ze || nc != ne || mc != me) {
        std::stringstream msg;
        msg << "RecombRate: the dielectronic tables disagree at Z = " << zc
            << ", N = " << nc << ", M = " << mc
            << "; they must carry the same rows in the same order.";
        throw std::runtime_error(msg.str());
      }
      DRFit fit;
      Real cval, eval;
      while (fit.nchan < kMaxDRChannels && (ic >> cval) && (ie >> eval)) {
        fit.c[fit.nchan] = cval;
        fit.e[fit.nchan] = eval;
        ++fit.nchan;
      }
      dr_[Key(zc, nc, mc)] = fit;
    }
    if (dr_.empty()) {
      std::stringstream msg;
      msg << "RecombRate: no dielectronic fits found in '" << fname_c << "'.";
      throw std::runtime_error(msg.str());
    }
  }

  const RRFit &FindRR(int z, int n, int m) const {
    std::map<int, RRFit>::const_iterator it = rr_.find(Key(z, n, m));
    if (it == rr_.end()) {
      std::stringstream msg;
      msg << "RecombRate: no radiative fit for Z = " << z << ", N = " << n
          << ", M = " << m << ". The table must be loaded first, and N counts"
          << " the electrons before recombination.";
      throw std::runtime_error(msg.str());
    }
    return it->second;
  }

  const DRFit &FindDR(int z, int n, int m) const {
    std::map<int, DRFit>::const_iterator it = dr_.find(Key(z, n, m));
    if (it == dr_.end()) {
      std::stringstream msg;
      msg << "RecombRate: no dielectronic fit for Z = " << z << ", N = " << n
          << ", M = " << m << ".";
      throw std::runtime_error(msg.str());
    }
    return it->second;
  }

 private:
  std::map<int, RRFit> rr_;
  std::map<int, DRFit> dr_;
  HydrogenCase hcase_ = HydrogenCase::kCaseBFerland;
};

}  // namespace Rates

#endif  // CHEMISTRY2_IONS_RATES_RECOMB_RATE_HPP_
