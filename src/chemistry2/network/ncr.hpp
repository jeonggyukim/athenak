#ifndef CHEMISTRY2_NETWORK_NCR_HPP_
#define CHEMISTRY2_NETWORK_NCR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ncr.hpp
//! \brief The NCR chemical network of Kim et al. (2023), ported from Tigris photchem
//!        (src/photchem/ncr_rates.hpp, ncr_solver.hpp): H2, H+ and e stored; H I the
//!        hydrogen remainder; C+, CO, C I and O I closures per substep.
//!
//!   H2 and H+ are advanced together by backward Euler, linear in (x_H2, x_H+), as a
//!   2x2 system (Tigris UpdateChemistry), followed by the RAMSES-style closure and the
//!   free-electron update with the new abundances. The substep is limited by the
//!   absolute rates of change of x_H2 and x_H+, as in Tigris. Heating and cooling use
//!   the terms shared with GOW17 (thermo.hpp); the NCR-only terms come in P5b.

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "chemistry2/data_path.hpp"
#include "chemistry2/inputs/cell_inputs.hpp"
#include "chemistry2/network/ncr_rates.hpp"
#include "chemistry2/network/ncr_table.hpp"
#include "chemistry2/rates/row_table.hpp"
#include "chemistry2/network/network_config.hpp"
#include "chemistry2/network/species_table.hpp"
#include "chemistry2/thermo/dust.hpp"
#include "chemistry2/thermo/hot_gas.hpp"
#include "chemistry2/thermo/h2_cooling_moseley.hpp"
#include "chemistry2/thermo/ionized.hpp"
#include "chemistry2/thermo/thermo.hpp"
#include "chemistry2/unit_factors.hpp"
#include "units/units.hpp"

namespace chemistry2 {

//! TABLE: temperature-only coefficients from the NCR rate table (the default, "NCR"), or
//! from the fits ("NCR_analytic", for comparisons with Tigris)
template <bool TABLE>
struct NCRNetwork {
  static constexpr std::string_view NAME = TABLE ? "NCR" : "NCR_analytic";
  static constexpr std::array<SpeciesRow, 3> rows = {
      Species("H2").H(2), Species("H+").Charge(1).H(1), Species("e")};
  static constexpr bool STORES_ELECTRONS = true;
  static constexpr int ROW_EL = 2;
  static constexpr int HYDROGEN_REMAINDER_CHARGE = 0;
  enum : int {IH2, IH_PLUS, IE};
  static constexpr int NSPEC = NumSpecies<NCRNetwork>();
  using UpdateOrder = std::integer_sequence<int, IH2>;  // the H2/H+ block, then e
  using StepLimitingSpecies = std::integer_sequence<int>;  // MaxStep instead
  static constexpr bool MAX_STEP = true;

  struct Settings {
    Real zd, zg, x_he;
    Real kgr_h2;                 // H2 formation on grains at Z_d = 1 [cm^3 s^-1]
    bool kgr_h2_t;               // Hollenbach & McKee (1979) temperature factor
    bool cr_phot_c;              // C ionization by CR-induced H2 emission
    Real temp_hot1;              // no H2 above [K]
    Real xi_diss_h2_isrf;        // H2 dissociation in the unattenuated ISRF [s^-1]
    Real leff_co_max;            // [cm]
    bool h2_rovib_cooling;
    bool dust_cooling;
    Real temp_dust, alpha_gd;
    RowTable table;              // NCR rate table (TABLE only)
    bool case_a = false;         // H+ recombination: case A instead of B (as Cloudy CIE)
    bool nebular = true;         // nebular line fit; off when ion ladders cool by line
    Real temp_hot0 = ncr::TEMP_HOT0;  // hot-gas window [temp_hot0, temp_hot1] [K]
    hot::CIETable cie;           // CIE helium and metal terms (analytic path only; the
                                 // table path reads them from gow17::C_CIE columns)
    // H I collisional ionization: Janev et al. (1987) fit as in Gong et al. (2017) and
    // Tigris NCR (default), or Voronov (1997) from inputs/chemistry2/ions
    bool coll_ion_hi_voronov = false;
    Rates::CollIonFit ci_h;
    // above temp_hot0 the hydrogen cooling hands over to the CIE n_H Lambda_H(T) with the
    // hot-gas weight (Tigris cool_hyd_cie_flag)
    bool cool_hyd_cie = false;
  };

  static Settings ReadSettings(ParameterInput *pin, const std::string &b) {
    Settings s = ReadValues(pin, b);
    s.temp_hot0 = pin->GetOrAddReal(b, "NCR_temp_hot0", ncr::TEMP_HOT0);
    s.cool_hyd_cie = pin->GetOrAddBoolean(b, "NCR_cool_hyd_cie", false);
    const std::string ci = pin->GetOrAddString(b, "NCR_coll_ion_HI", "janev");
    if (ci != "janev" && ci != "voronov") {
      throw std::runtime_error("NCR_coll_ion_HI must be janev or voronov, not " + ci);
    }
    s.coll_ion_hi_voronov = (ci == "voronov");
    // the data files are read only when an NCR chemical network is the one selected
    const bool selected = pin->GetOrAddString(b, "network", "").rfind("NCR", 0) == 0;
    hot::CIESource src;
    if (selected) {
      src = hot::ReadCIESource(pin->GetOrAddString(
          b, "cie_table", DataPath("inputs/chemistry2/cie/tigress_coolftn_ncr.txt")));
      if (s.coll_ion_hi_voronov) {
        Rates::CollIonRate rates;
        rates.Load(pin->GetOrAddString(b, "ion_data_dir",
                                       DataPath("inputs/chemistry2/ions/")) +
                   "voronov97_coll_ion.dat");
        s.ci_h = rates.Get(1, 1);
      }
    }
    if constexpr (TABLE) {
      s.table = ncr::BuildNCRTable(gow17::ROWS_PER_OCTAVE, selected ? &src : nullptr,
                                   s.coll_ion_hi_voronov ? &s.ci_h : nullptr);
    } else {
      if (selected) s.cie = hot::MakeCIETable(src);
    }
    return s;
  }

  static Settings ReadValues(ParameterInput *pin, const std::string &b) {
    return Settings{pin->GetOrAddReal(b, "NCR_Z_d", 1.0),
                    pin->GetOrAddReal(b, "NCR_Z_g", 1.0),
                    pin->GetOrAddReal(b, "NCR_xHe", 0.1),
                    pin->GetOrAddReal(b, "NCR_kgr_H2", 3.0e-17),
                    pin->GetOrAddBoolean(b, "NCR_kgr_H2_flag", true),
                    pin->GetOrAddBoolean(b, "NCR_CRPhotC_flag", true),
                    pin->GetOrAddReal(b, "NCR_temp_hot1", ncr::TEMP_HOT1),
                    pin->GetOrAddReal(b, "NCR_xi_diss_h2_isrf", 5.7e-11),
                    pin->GetOrAddReal(b, "NCR_Leff_CO_max", 3.086e20),
                    pin->GetOrAddBoolean(b, "NCR_H2_rovib_cooling", true),
                    pin->GetOrAddBoolean(b, "NCR_dust_cooling", false),
                    pin->GetOrAddReal(b, "NCR_dust_temperature", 10.0),
                    pin->GetOrAddReal(b, "NCR_alpha_gd", 0.0), RowTable(),
                    pin->GetOrAddBoolean(b, "NCR_case_A", false)};
  }

  struct Cell {
    Real n_h, inv_n_h, gamma;
    Real u_time, u_energy, inv_u_energy;
    Settings s;
    Real chi, xi_cr, grad_v;  // FUV field [Draine] for PE, C I, CO, H2; CR rate; |grad v|
    Real xi_diss_h2;          // [s^-1]
  };

  KOKKOS_INLINE_FUNCTION static Cell MakeCell(const Real rho, const Real gamma,
                                              const Settings &s, const UnitFactors &u,
                                              const CellInputs &in) {
    Cell c;
    c.n_h = rho*u.nh_per_density;
    c.inv_n_h = 1.0/c.n_h;
    c.gamma = gamma;
    c.u_time = u.time_cgs;
    c.u_energy = u.energy_cgs;
    c.inv_u_energy = u.energy_per_cgs;
    c.s = s;
    c.chi = in.g0;
    c.xi_cr = in.xi_cr;
    c.grad_v = in.grad_v;
    c.xi_diss_h2 = in.g0*s.xi_diss_h2_isrf;
    return c;
  }

  //! Rate coefficients that depend on T alone
  struct TCoef {
    Real alpha_b, k_coll, alpha_cii, k_ch2p, kgr_fac;
    Real l9l, l9h, l10l, l10h, ncr_h2, ncr_hi, k1619;
    Real xe_cie;  // CIE helium and metal electrons, xe_He + Z_g xe_metal
  };

  KOKKOS_INLINE_FUNCTION static TCoef Coefficients(const Real T, const Cell &c,
                                                   Slot *slot) {
    TCoef k;
    if constexpr (TABLE) {
      const RowTable &t = c.s.table;
      *slot = t.grid.Locate(T);
      const Slot &s = *slot;
      k.alpha_b = t.At(s, c.s.case_a ? ncr::C_ALPHA_A : ncr::C_ALPHA_B);
      k.k_coll = t.At(s, ncr::C_K_COLL_HI);
      k.alpha_cii = t.At(s, ncr::C_ALPHA_CII);
      k.k_ch2p = t.At(s, ncr::C_K_CH2P);
      k.kgr_fac = t.At(s, ncr::C_KGR_FAC);
      k.l9l = t.At(s, ncr::C_L9L);
      k.l9h = t.At(s, ncr::C_L9H);
      k.l10l = t.At(s, ncr::C_L10L);
      k.l10h = t.At(s, ncr::C_L10H);
      k.ncr_h2 = t.At(s, ncr::C_NCR_H2);
      k.ncr_hi = t.At(s, ncr::C_NCR_HI);
      k.k1619 = t.At(s, ncr::C_K1619);
      k.xe_cie = t.At(s, gow17::C_CIE + hot::XE_HE) +
                 c.s.zg*t.At(s, gow17::C_CIE + hot::XE_METAL);
    } else {
      k.alpha_b = c.s.case_a ? ncr::AlphaAHII(T) : ncr::AlphaBHII(T);
      k.k_coll = c.s.coll_ion_hi_voronov ? c.s.ci_h.Rate(T) : ncr::CollIonHI(T);
      k.alpha_cii = ncr::AlphaCII(T);
      k.k_ch2p = 3.3e-13*Kokkos::pow(T, -1.3)*Kokkos::exp(-23.0/T);
      const Real T2 = T*1e-2;
      k.kgr_fac = Kokkos::sqrt(T2)*2.0/(1.0 + 0.4*Kokkos::sqrt(T2) + 0.2*T2 +
                                        0.08*T2*T2);
      const Real Tc = Kokkos::fmax(T, 7.0e2), inv_T = 1.0/Tc;
      const Real lt4 = Kokkos::log10(Tc*1e-4);
      k.l9l = Kokkos::log10(6.67e-12*Kokkos::sqrt(Tc)*
                            Kokkos::exp(-(1.0 + 63590.0*inv_T)));
      k.l9h = Kokkos::log10(3.52e-9*Kokkos::exp(-43900.0*inv_T));
      k.l10l = Kokkos::log10(5.996e-30*Kokkos::pow(Tc, 4.1881)/
                             Kokkos::pow(1.0 + 6.761e-6*Tc, 5.6881)*
                             Kokkos::exp(-54657.4*inv_T));
      k.l10h = Kokkos::log10(1.3e-9*Kokkos::exp(-53300.0*inv_T));
      k.ncr_h2 = Kokkos::pow(10.0, 4.845 - 1.3*lt4 + 1.62*lt4*lt4);
      k.ncr_hi = Kokkos::pow(10.0, 3.0 - 0.416*lt4 - 0.327*lt4*lt4);
      k.k1619 = 1.0e-7*Kokkos::pow(T*1e-2, -0.5);
      k.xe_cie = 0.0;
      if (c.s.cie.n > 0 && T > c.s.temp_hot0) {
        Real v[hot::NCOL];
        c.s.cie.Eval(T, v);
        k.xe_cie = v[hot::XE_HE] + c.s.zg*v[hot::XE_METAL];
      }
    }
    return k;
  }

  //! One substep: temperature, closures, and the rates [s^-1] at the substep start
  struct Substep {
    Real T, x_hi, x_e;
    Real x_cii, x_co, x_ci, x_oi;
    Real c_hii, d_hii, c_h2, d_h2;  // dx/dt = c x_HI - d x (Tigris form)
    Real k_coll_hi, kgr, k_h2_h, k_h2_h2;
    TCoef tc;
    Slot slot;  // table rows of T (TABLE only)
  };

  KOKKOS_INLINE_FUNCTION static Real Temperature(const Real *y, const Real e,
                                                 const Cell &c) {
    return e*c.u_energy*c.inv_n_h/
           Thermo::CvCold(y[IH2], c.s.x_he, y[IE], c.gamma);
  }

  //! x_C+ in equilibrium (Tigris GetxCII), with the stored (lagged) x_e
  KOKKOS_INLINE_FUNCTION static Real XCII(const Real x_e, const Real x_h2, const Real T,
                                          const Real lnT, const Cell &c,
                                          const TCoef &tc) {
    const Real n_h = c.n_h;
    const Real k_cr = 3.85*c.xi_cr;
    Real k_ph = 3.5e-10*c.chi;
    if (c.s.cr_phot_c) k_ph += 520.0*2.0*x_h2*c.xi_cr;
    const Real psi = 1.7*c.chi*Kokkos::sqrt(T)/(n_h*x_e + 1e-50) + 1e-50;
    constexpr Real CCP[7] = {45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4};
    const Real k_gr = ncr::GrainRec(CCP, T, lnT, psi)*c.s.zd;
    const Real cc = (k_cr + k_ph)/n_h;
    return ncr::X_C_STD*c.s.zg*cc/(tc.alpha_cii*x_e + k_gr + tc.k_ch2p*x_h2 + cc);
  }

  //! Electrons from molecular ions (Tigris GetxeMol)
  KOKKOS_INLINE_FUNCTION static Real XeMol(const Real x_e, const Real x_h2, const Cell &c,
                                           const TCoef &tc) {
    const Real k1620 = 1e-14*c.s.zd, k1622 = 1e-14*c.s.zd, k1621 = 1e-9;
    const Real k1619 = tc.k1619;
    const Real phi_s = (1.0 - x_e/ncr::X_EL_MAX)*0.67/(1.0 + x_e/0.05);
    const Real x_s = 5.3e-6*c.s.zg;
    const Real a = k1619*(1.0 + k1621/k1622*x_s);
    const Real b = k1620 + k1621*x_s;
    return 2.0*x_h2*(Kokkos::sqrt(b*b + 4.0*a*c.xi_cr*(1.0 + phi_s)*c.inv_n_h) - b)/
           (2.0*k1619);
  }

  //! x_CO (Tigris CalculateCOAbundance), chi_CO = chi
  KOKKOS_INLINE_FUNCTION static Real XCO(const Real x_h2, const Real x_cii,
                                         const Real x_oii, const Cell &c) {
    const Real k16 = c.xi_cr*1.0e16;
    const Real t1 = Kokkos::fmax(4.0e3*c.s.zd/(k16*k16), 1.0);
    const Real ncrit = Kokkos::pow(t1, Kokkos::cbrt(c.chi))*
                       (50.0*k16/Kokkos::pow(c.s.zd, 1.4));
    const Real f = 2.0*x_h2/(1.0 + ncrit*ncrit/(c.n_h*c.n_h));
    return f*Kokkos::fmin(ncr::X_O_STD*c.s.zg - x_oii, ncr::X_C_STD*c.s.zg - x_cii);
  }

  //! The closures at T: x_C+, x_CO, x_C I, x_O I
  KOKKOS_INLINE_FUNCTION static void SetClosures(const Real *y, const Real T,
                                                 const Real lnT, const Cell &c,
                                                 Substep &sub) {
    sub.x_cii = XCII(y[IE], y[IH2], T, lnT, c, sub.tc);
    const Real x_oii = y[IH_PLUS]*ncr::X_O_STD*c.s.zg;
    sub.x_co = XCO(y[IH2], sub.x_cii, x_oii, c);
    sub.x_ci = Kokkos::fmax(ncr::X_C_STD*c.s.zg - sub.x_cii - sub.x_co, ncr::TINY);
    sub.x_oi = Kokkos::fmax(ncr::X_O_STD*c.s.zg - x_oii - sub.x_co, ncr::TINY);
  }

  KOKKOS_INLINE_FUNCTION static Substep Prepare(const Real *y, const Real e,
                                                const Cell &c) {
    Substep sub;
    const Real T = Kokkos::fmax(Temperature(y, e, c), 1.0);
    const Real lnT = Kokkos::log(T);
    sub.T = T;
    sub.x_e = y[IE];
    sub.x_hi = Kokkos::fmax(1.0 - 2.0*y[IH2] - y[IH_PLUS], ncr::TINY);
    sub.tc = Coefficients(T, c, &sub.slot);
    const TCoef &tc = sub.tc;
    SetClosures(y, T, lnT, c, sub);
    const Real n_h = c.n_h, n_e = y[IE]*n_h;
    // H+
    sub.k_coll_hi = tc.k_coll;
    Real a_gr = 0.0;
    if (T < c.s.temp_hot1) {
      constexpr Real CHP[7] = {12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723,
                               1.102e-5};
      const Real psi = 1.7*c.chi*Kokkos::sqrt(T)/(n_e + 1e-50) + 1e-50;
      a_gr = ncr::GrainRec(CHP, T, lnT, psi)*c.s.zd;
      if (T > c.s.temp_hot0) {
        a_gr *= 1.0 - hot::Weight<Real>(T, c.s.temp_hot0, c.s.temp_hot1);
      }
    }
    const Real xi_h = c.xi_cr*(2.3*y[IH2] + 1.5*sub.x_hi);
    sub.c_hii = sub.k_coll_hi*n_e + xi_h;
    sub.d_hii = n_e*tc.alpha_b + n_h*a_gr;
    // H2
    sub.k_h2_h = 0.0;
    sub.k_h2_h2 = 0.0;
    if (T < c.s.temp_hot1) {
      sub.kgr = c.s.kgr_h2_t ? c.s.kgr_h2*c.s.zd*tc.kgr_fac : c.s.kgr_h2*c.s.zd;
      if (T > 7.0e2) {
        const Real inv_ncr = Kokkos::fmax(sub.x_hi/tc.ncr_hi + 2.0*y[IH2]/tc.ncr_h2,
                                          1e-50);
        const Real n2 = n_h*inv_ncr, w = n2/(1.0 + n2), w0 = 1.0/(1.0 + n2);
        sub.k_h2_h = Kokkos::pow(10.0, tc.l9h*w + tc.l9l*w0);
        sub.k_h2_h2 = Kokkos::pow(10.0, tc.l10h*w + tc.l10l*w0);
      }
      const Real xi_coll = (sub.k_h2_h2*y[IH2] + sub.k_h2_h*sub.x_hi)*n_h;
      sub.c_h2 = sub.kgr*n_h;
      sub.d_h2 = 1.65*2.0*xi_h + c.xi_diss_h2 + xi_coll;
    } else {
      sub.kgr = 0.0;
      sub.c_h2 = 0.0;
      sub.d_h2 = 0.0;
    }
    return sub;
  }

  //! Free electrons from H+, C+, O+ and molecular ions (Tigris CalculateFreeElectron);
  //! above temp_hot0 the metal and molecular part hands over to the CIE helium and
  //! metal electrons with the hot-gas weight
  KOKKOS_INLINE_FUNCTION static Real FreeElectrons(const Real x_e_old, const Real x_h2,
                                                   const Real x_hii, const Real T,
                                                   const Cell &c, const TCoef &tc) {
    const Real lnT = Kokkos::log(T);
    const Real x_cii = XCII(x_e_old, x_h2, T, lnT, c, tc);
    if (T <= c.s.temp_hot0) {
      return x_hii + x_cii + x_hii*ncr::X_O_STD*c.s.zg + XeMol(x_e_old, x_h2, c, tc);
    }
    const Real w = hot::Weight<Real>(T, c.s.temp_hot0, c.s.temp_hot1);
    const Real cold = x_cii + x_hii*ncr::X_O_STD*c.s.zg + XeMol(x_e_old, x_h2, c, tc);
    return x_hii + (1.0 - w)*cold + w*tc.xe_cie;
  }

  //! Creation c [per H per code time] and destruction d [per code time]
  template <int K>
  KOKKOS_INLINE_FUNCTION static CD Rates(const Real *, Real, const Cell &cl,
                                         const Substep &sub) {
    if constexpr (K == IH2) {
      return CD{cl.u_time*sub.c_h2*sub.x_hi, cl.u_time*sub.d_h2};
    } else if constexpr (K == IH_PLUS) {
      return CD{cl.u_time*sub.c_hii*sub.x_hi, cl.u_time*sub.d_hii};
    } else {
      return CD{0.0, 0.0};  // electrons follow the closure in Update<IH2>
    }
  }

  template <int K>
  static constexpr bool CUSTOM_UPDATE = (K == IH2);

  //! H2 and H+ by backward Euler, linear in both (x_HI = 1 - 2 x_H2 - x_H+), Cramer's
  //! rule; then the RAMSES-style closure and the free electrons (Tigris UpdateChemistry)
  template <int K>
  KOKKOS_INLINE_FUNCTION static void Update(Real *y, const Real *y_n, const Real,
                                            const Cell &cl, const Substep &sub,
                                            const Real h) {
    static_assert(K == IH2, "NCR's only custom update is the H2/H+ block");
    const Real dt = h*cl.u_time;
    const Real a = 1.0 + (2.0*sub.c_h2 + sub.d_h2)*dt;
    const Real b = sub.c_h2*dt;
    const Real c = 2.0*sub.c_hii*dt;
    const Real d = 1.0 + (sub.c_hii + sub.d_hii)*dt;
    const Real e = y_n[IH2] + sub.c_h2*dt;
    const Real f = y_n[IH_PLUS] + sub.c_hii*dt;
    const Real inv_det = 1.0/(a*d - b*c);
    Real x_hii = Kokkos::fmax(Kokkos::fmin((a*f - c*e)*inv_det, 1.0), ncr::TINY);
    Real x_h2 = (sub.T < cl.s.temp_hot1) ? (d*e - b*f)*inv_det : 0.0;
    const Real sum = x_hii + 2.0*x_h2;
    if (sum > 1.0) {
      x_hii /= sum;
      x_h2 /= sum;
    }
    if (x_h2 < 0.25) {
      x_h2 = Kokkos::fmin(Kokkos::fmax(x_h2, ncr::TINY), 0.5);
      x_hii = Kokkos::fmin(Kokkos::fmax(x_hii, ncr::TINY), 1.0 - 2.0*x_h2);
    } else {
      x_hii = Kokkos::fmin(Kokkos::fmax(x_hii, ncr::TINY), 1.0);
      x_h2 = Kokkos::fmin(Kokkos::fmax(x_h2, ncr::TINY), 1.0 - x_hii);
    }
    y[IH2] = x_h2;
    y[IH_PLUS] = x_hii;
    y[IE] = FreeElectrons(y_n[IE], x_h2, x_hii, sub.T, cl, sub.tc);
  }

  //! Largest substep [code time]: frac over the absolute rates of change of x_H+ and x_H2
  KOKKOS_INLINE_FUNCTION static Real MaxStep(const Real frac, const Real *y, Real,
                                             const Cell &cl, const Substep &sub) {
    const Real xdot_hii = Kokkos::fabs(sub.c_hii*sub.x_hi - sub.d_hii*y[IH_PLUS]);
    const Real xdot_h2 = Kokkos::fabs(sub.c_h2*sub.x_hi - sub.d_h2*y[IH2]);
    const Real xdot = Kokkos::fmax(xdot_hii, xdot_h2)*cl.u_time;
    return (xdot > 0.0) ? frac/xdot : Kokkos::Experimental::infinity_v<Real>;
  }

  //! Heating and cooling rates [erg s^-1 per H]
  struct HeatCool {
    Real heating, cooling;
  };

  //! de/dt in code units
  KOKKOS_INLINE_FUNCTION static Real Edot(const Real *y, const Real e, const Cell &c,
                                          const Substep &sub) {
    const HeatCool hc = HeatingCooling(y, e, c, sub);
    return c.u_time*((hc.heating - hc.cooling)*c.n_h*c.inv_u_energy);
  }

  //! The heating and cooling terms shared with GOW17
  KOKKOS_INLINE_FUNCTION static HeatCool HeatingCooling(const Real *y, const Real e,
                                                        const Cell &c,
                                                        const Substep &sub) {
    const Real T = Kokkos::fmax(Temperature(y, e, c), 1.0);
    if constexpr (TABLE) {
      const HeatCoolT<Real> hc = HeatCoolTable<Real>(y, T, 1.0, c, sub);
      return HeatCool{hc.heating, hc.cooling};
    }
    const Real n_h = c.n_h, x_e = y[IE], x_h2 = y[IH2], x_hi = sub.x_hi;
    const Real n_hi = n_h*x_hi, n_h2 = n_h*x_h2, n_e = n_h*x_e;
    const Real k_ph_h2 = c.xi_diss_h2;
    Real heating = Thermo::HeatingCR(x_e, n_h, x_hi, x_h2, c.xi_cr);
    heating += Thermo::HeatingPE(c.chi, c.s.zd, T, n_e);
    heating += Thermo::HeatingH2gr(x_hi, x_h2, n_h, T, sub.kgr*n_h, k_ph_h2);
    heating += Thermo::HeatingH2pump(x_hi, x_h2, n_h, T, k_ph_h2);
    heating += Thermo::HeatingH2diss(k_ph_h2, x_h2);
    // non-hydrogen cold-gas cooling (fine-structure, CO, nebular, grain recombination,
    // gas-dust), handed to CIE above temp_hot0
    Real other = Thermo::CoolingCII(sub.x_cii, n_hi, n_h2, n_e, T);
    other += Thermo::CoolingCI(sub.x_ci, n_hi, n_h2, n_e, T);
    other += Thermo::CoolingOI(sub.x_oi, n_hi, n_h2, n_e, T);
    if (sub.x_co > 0.0) {
      const Real vth = Kokkos::sqrt(2.*units::Units::k_boltzmann_cgs*T/
                                    units::Units::CO_mass_cgs);
      const Real grad_eff = Kokkos::fmax(c.grad_v, vth/c.s.leff_co_max);
      other += Thermo::CoolingCOR(sub.x_co, n_hi, n_h2, n_e, T, n_h*sub.x_co/grad_eff);
    }
    if (c.s.nebular && T < c.s.temp_hot1) {
      other += ionized::CoolingNebular(x_e, y[IH_PLUS], n_h, T, c.s.zg);
    }
    other += Thermo::CoolingRec(c.s.zd, T, n_e, c.chi);
    if (c.s.dust_cooling) {
      other += dust::GasToDust(n_h, T, c.s.temp_dust, c.s.zd, c.s.alpha_gd);
    }
    Real cooling = 0.0;
    if (T > 3.0e3) cooling += Thermo::CoolingLya(x_hi, n_e, T);
    if (c.s.h2_rovib_cooling && T < c.s.temp_hot1) {
      cooling += h2_moseley::Cooling(x_h2, x_hi, n_h, T);
    }
    cooling += ionized::CoolingHII(x_e, y[IH_PLUS], n_h, T, ncr::AlphaBHII(T));
    cooling += Thermo::CoolingH2diss(x_hi, x_h2, sub.k_h2_h*n_h, sub.k_h2_h2*n_h);
    cooling += Thermo::CoolingHIion(x_hi, x_e, sub.k_coll_hi*n_h);
    if (T > c.s.temp_hot0) {
      Real lam = 0.0;
      if (c.s.cie.n > 0) {
        Real v[hot::NCOL];
        c.s.cie.Eval(T, v);
        lam = v[hot::LAMBDA_HE] + c.s.zg*v[hot::LAMBDA_METAL];
        if (c.s.cool_hyd_cie) HotHydrogen<Real>(T, n_h*v[hot::LAMBDA_H], c, &cooling);
      }
      HotBlend<Real>(T, n_h*lam, c, &heating, &other);
    }
    return HeatCool{heating, cooling + other};
  }

  //! Above temp_hot0: heating and the non-hydrogen cold-gas cooling times 1 - w, plus
  //! w times the CIE helium and metal cooling cie [erg s^-1 per H]
  template <class S>
  KOKKOS_INLINE_FUNCTION static void HotBlend(const S &T, const S &cie, const Cell &c,
                                              S *heating, S *other) {
    if (dual::Value(T) <= c.s.temp_hot0) return;
    const S w = hot::Weight<S>(T, c.s.temp_hot0, c.s.temp_hot1);
    *heating *= 1.0 - w;
    *other = (1.0 - w)*(*other) + w*cie;
  }

  //! Above temp_hot0: the hydrogen cooling times 1 - w plus w times the CIE hydrogen
  //! cooling cie [erg s^-1 per H]
  template <class S>
  KOKKOS_INLINE_FUNCTION static void HotHydrogen(const S &T, const S &cie, const Cell &c,
                                                 S *cooling) {
    const S w = hot::Weight<S>(T, c.s.temp_hot0, c.s.temp_hot1);
    *cooling = (1.0 - w)*(*cooling) + w*cie;
  }

  static constexpr bool EDOT_DERIVATIVE = TABLE;  // EdotWithDerivative, dual numbers

  //! edot in code units, and d(edot)/de at fixed species and closures, from one pass
  KOKKOS_INLINE_FUNCTION static Real EdotWithDerivative(const Real *y, const Real e,
                                                        const Cell &c,
                                                        const Substep &sub,
                                                        Real *dedot_de) {
    const Real dT_de = c.u_energy*c.inv_n_h/
                       Thermo::CvCold(y[IH2], c.s.x_he, y[IE], c.gamma);
    const Real T_raw = e*dT_de;
    const Real T = Kokkos::fmax(T_raw, 1.0);
    const Real seed = (T_raw >= 1.0) ? 1.0 : 0.0;
    const auto hc = HeatCoolTable<dual::Dual>(y, T, seed, c, sub);
    const Real fac = c.u_time*c.n_h*c.inv_u_energy;
    *dedot_de = fac*(hc.heating.d - hc.cooling.d)*dT_de;
    return fac*(hc.heating.v - hc.cooling.v);
  }

  template <class S>
  struct HeatCoolT {
    S heating, cooling;
  };

  //! Heating and cooling with the temperature-only factors from the NCR table; with
  //! S = dual::Dual the derivative part is d/dT at fixed species and closures
  template <class S>
  KOKKOS_INLINE_FUNCTION static HeatCoolT<S> HeatCoolTable(const Real *y, const Real T,
                                                           const Real seed,
                                                           const Cell &c,
                                                           const Substep &sub) {
    constexpr bool DERIV = std::is_same_v<S, dual::Dual>;
    const RowTable &tab = c.s.table;
    Real dw = 0.0;
    Slot s_t;
    if constexpr (DERIV) {
      s_t = tab.grid.Locate(T, &dw);
      dw *= seed;
    } else {
      s_t = (T == sub.T) ? sub.slot : tab.grid.Locate(T);
    }
    S Ts;
    if constexpr (DERIV) {
      Ts = dual::Dual(T, seed);
    } else {
      Ts = T;
    }
    auto col = [&](const int k) { return gow17::RowCoef<S>(tab, s_t, dw, k); };
    const Real n_h = c.n_h, x_e = y[IE], x_h2 = y[IH2], x_hi = sub.x_hi;
    const Real x_hii = y[IH_PLUS];
    const Real n_hi = n_h*x_hi, n_h2 = n_h*x_h2, n_e = n_h*x_e;
    const Real k_ph_h2 = c.xi_diss_h2;
    // heating
    const Thermo::H2GeffRateCoefficientsT<S> geff{col(gow17::C_H2GEFF),
                                                  col(gow17::C_H2GEFF + 1)};
    S kgr_n = sub.kgr*n_h;
    if constexpr (DERIV) {
      if (c.s.kgr_h2_t && T < c.s.temp_hot1) {
        kgr_n.d = c.s.kgr_h2*c.s.zd*n_h*col(ncr::C_KGR_FAC).d;
      }
    }
    S heating = Thermo::HeatingCRFrom(x_e, x_hi, x_h2, c.xi_cr,
                                      Thermo::CRHeatPerH2(n_h));
    heating += Thermo::HeatingPE(c.chi, c.s.zd, Ts, n_e);
    heating += Thermo::HeatingH2grFrom(x_hi, x_h2, n_h, kgr_n, k_ph_h2, geff);
    heating += Thermo::HeatingH2pumpFrom(x_hi, x_h2, n_h, k_ph_h2, geff);
    heating += Thermo::HeatingH2diss(k_ph_h2, x_h2);
    // cooling: fine-structure lines from GOW17's columns
    const gow17::GOW17ThermoCoefficients<S> k = gow17::LookupThermo<S>(tab, s_t, dw);
    // non-hydrogen cold-gas cooling, handed to CIE above temp_hot0 (HotBlend)
    S other = Thermo::CoolingCIIFrom(sub.x_cii, n_hi, n_h2, n_e, k.cii, k.cii_boltz);
    other += Thermo::CoolingCIFrom(sub.x_ci, n_hi, n_h2, n_e, k.ci, k.ci_boltz[0],
                                   k.ci_boltz[1], k.ci_boltz[2]);
    other += Thermo::CoolingOIFrom(sub.x_oi, n_hi, n_h2, n_e, k.oi, k.oi_boltz[0],
                                   k.oi_boltz[1], k.oi_boltz[2]);
    if (sub.x_co > 0.0) {
      const S vth = dual::sqrt((2.*units::Units::k_boltzmann_cgs/
                                units::Units::CO_mass_cgs)*Ts);
      const S grad_eff = dual::fmax(vth*(1.0/c.s.leff_co_max), c.grad_v);
      const S n_co_eff = (n_h*sub.x_co)/grad_eff;
      other += Thermo::CoolingCOR(sub.x_co, n_hi, n_h2, n_e, Ts, n_co_eff);
    }
    if (c.s.nebular && T < c.s.temp_hot1) {  // nebular lines
      const S l = dual::log(Ts*1e-4);
      const S f_red = 1.0/(1.0 + 0.12*dual::pow(x_e*n_h*1e-2, 0.38 - 0.12*l));
      other += col(ncr::C_NEB)*f_red*(3.677602203699553e-21*c.s.zg*x_hii*x_e*n_h);
    }
    other += Thermo::CoolingRec(c.s.zd, Ts, n_e, c.chi);
    if (c.s.dust_cooling) {
      other += dust::GasToDust(n_h, Ts, c.s.temp_dust, c.s.zd, c.s.alpha_gd);
    }
    S cooling = 0.0;
    if (T > 3.0e3) {
      cooling += Thermo::CoolingLyaFrom(x_hi, n_e, col(gow17::C_LYA),
                                        col(gow17::C_LYA + 1));
    }
    if (c.s.h2_rovib_cooling && T < c.s.temp_hot1) {  // Moseley et al. (2021)
      Real d[4];
      h2_moseley::DensityFactors(n_hi, n_h2, d);
      cooling += (col(ncr::C_MOS)*d[0] + col(ncr::C_MOS + 1)*d[1] +
                  col(ncr::C_MOS + 2)*d[2] + col(ncr::C_MOS + 3)*d[3])*x_h2;
    }
    cooling += (col(gow17::C_HII_FF) + col(gow17::C_HII_REC))*(n_h*x_e*x_hii);
    // collisional H2 dissociation and H ionization: values from Prepare, derivative
    // of their temperature dependence from the table
    S k_h2_h = sub.k_h2_h*n_h, k_h2_h2 = sub.k_h2_h2*n_h, k_hi = sub.k_coll_hi*n_h;
    if constexpr (DERIV) {
      constexpr Real LN10 = 2.30258509299404590e0;
      if (T > 7.0e2 && T < c.s.temp_hot1) {
        const S inv_ncr = x_hi/col(ncr::C_NCR_HI) + 2.0*x_h2/col(ncr::C_NCR_H2);
        const S n2 = n_h*inv_ncr;
        const S inv = 1.0/(1.0 + n2);
        const S e9 = (col(ncr::C_L9H)*n2 + col(ncr::C_L9L))*inv;
        const S e10 = (col(ncr::C_L10H)*n2 + col(ncr::C_L10L))*inv;
        k_h2_h.d = k_h2_h.v*LN10*e9.d;
        k_h2_h2.d = k_h2_h2.v*LN10*e10.d;
      }
      k_hi.d = n_h*col(ncr::C_K_COLL_HI).d;
    }
    cooling += Thermo::CoolingH2diss(x_hi, x_h2, k_h2_h, k_h2_h2);
    cooling += Thermo::CoolingHIion(x_hi, x_e, k_hi);
    if (T > c.s.temp_hot0) {
      const S cie = n_h*(col(gow17::C_CIE + hot::LAMBDA_HE) +
                         c.s.zg*col(gow17::C_CIE + hot::LAMBDA_METAL));
      if (c.s.cool_hyd_cie) {
        HotHydrogen<S>(Ts, n_h*col(gow17::C_CIE + hot::LAMBDA_H), c, &cooling);
      }
      HotBlend<S>(Ts, cie, c, &heating, &other);
    }
    return HeatCoolT<S>{heating, cooling + other};
  }

  KOKKOS_INLINE_FUNCTION static void Closures(Real *, const Real *, const Cell &) {}
};

using NCR = NCRNetwork<true>;
using NCRAnalytic = NCRNetwork<false>;

}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_NCR_HPP_
