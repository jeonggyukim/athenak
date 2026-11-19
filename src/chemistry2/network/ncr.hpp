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
#include <string>
#include <string_view>
#include <utility>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "chemistry2/inputs/cell_inputs.hpp"
#include "chemistry2/network/ncr_rates.hpp"
#include "chemistry2/network/network_config.hpp"
#include "chemistry2/network/species_table.hpp"
#include "chemistry2/thermo/dust.hpp"
#include "chemistry2/thermo/ncr_terms.hpp"
#include "chemistry2/thermo/thermo.hpp"
#include "chemistry2/unit_factors.hpp"
#include "units/units.hpp"

namespace chemistry2 {

struct NCR {
  static constexpr std::string_view NAME = "NCR";  // <chemistry2>/network
  static constexpr std::array<SpeciesRow, 3> rows = {
      Species("H2").H(2), Species("H+").Charge(1).H(1), Species("e")};
  static constexpr bool STORES_ELECTRONS = true;
  static constexpr int ROW_EL = 2;
  static constexpr int HYDROGEN_REMAINDER_CHARGE = 0;
  enum : int {IH2, IH_PLUS, IE};
  static constexpr int NSPEC = NumSpecies<NCR>();
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
  };

  static Settings ReadSettings(ParameterInput *pin, const std::string &b) {
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
                    pin->GetOrAddReal(b, "NCR_alpha_gd", 0.0)};
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

  //! One substep: temperature, closures, and the rates [s^-1] at the substep start
  struct Substep {
    Real T, x_hi, x_e;
    Real x_cii, x_co, x_ci, x_oi;
    Real c_hii, d_hii, c_h2, d_h2;  // dx/dt = c x_HI - d x (Tigris form)
    Real k_coll_hi, kgr, k_h2_h, k_h2_h2;
  };

  KOKKOS_INLINE_FUNCTION static Real Temperature(const Real *y, const Real e,
                                                 const Cell &c) {
    return e*c.u_energy*c.inv_n_h/
           Thermo::CvCold(y[IH2], c.s.x_he, y[IE], c.gamma);
  }

  //! x_C+ in equilibrium (Tigris GetxCII), with the stored (lagged) x_e
  KOKKOS_INLINE_FUNCTION static Real XCII(const Real x_e, const Real x_h2, const Real T,
                                          const Real lnT, const Cell &c) {
    const Real n_h = c.n_h;
    const Real k_cr = 3.85*c.xi_cr;
    Real k_ph = 3.5e-10*c.chi;
    if (c.s.cr_phot_c) k_ph += 520.0*2.0*x_h2*c.xi_cr;
    const Real psi = 1.7*c.chi*Kokkos::sqrt(T)/(n_h*x_e + 1e-50) + 1e-50;
    constexpr Real CCP[7] = {45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4};
    const Real k_gr = ncr::GrainRec(CCP, T, lnT, psi)*c.s.zd;
    const Real k_h2 = 3.3e-13*Kokkos::pow(T, -1.3)*Kokkos::exp(-23.0/T);
    const Real cc = (k_cr + k_ph)/n_h;
    return ncr::X_C_STD*c.s.zg*cc/(ncr::AlphaCII(T)*x_e + k_gr + k_h2*x_h2 + cc);
  }

  //! Electrons from molecular ions (Tigris GetxeMol)
  KOKKOS_INLINE_FUNCTION static Real XeMol(const Real x_e, const Real x_h2, const Real T,
                                           const Cell &c) {
    const Real k1620 = 1e-14*c.s.zd, k1622 = 1e-14*c.s.zd, k1621 = 1e-9;
    const Real k1619 = 1.0e-7*Kokkos::pow(T*1e-2, -0.5);
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
    sub.x_cii = XCII(y[IE], y[IH2], T, lnT, c);
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
    SetClosures(y, T, lnT, c, sub);
    const Real n_h = c.n_h, n_e = y[IE]*n_h;
    // H+
    sub.k_coll_hi = ncr::CollIonHI(T);
    Real a_gr = 0.0;
    if (T < ncr::TEMP_HOT1) {
      constexpr Real CHP[7] = {12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723,
                               1.102e-5};
      const Real psi = 1.7*c.chi*Kokkos::sqrt(T)/(n_e + 1e-50) + 1e-50;
      a_gr = ncr::GrainRec(CHP, T, lnT, psi)*c.s.zd;
      if (T > ncr::TEMP_HOT0) {
        a_gr *= 1.0 - 1.0/(1.0 + Kokkos::exp(-10.0*(T - 0.5*(ncr::TEMP_HOT0 +
                                                              ncr::TEMP_HOT1))/
                                             (ncr::TEMP_HOT1 - ncr::TEMP_HOT0)));
      }
    }
    const Real xi_h = c.xi_cr*(2.3*y[IH2] + 1.5*sub.x_hi);
    sub.c_hii = sub.k_coll_hi*n_e + xi_h;
    sub.d_hii = n_e*ncr::AlphaBHII(T) + n_h*a_gr;
    // H2
    sub.k_h2_h = 0.0;
    sub.k_h2_h2 = 0.0;
    if (T < c.s.temp_hot1) {
      const Real T2 = T*1e-2;
      sub.kgr = c.s.kgr_h2_t
                    ? c.s.kgr_h2*c.s.zd*Kokkos::sqrt(T2)*2.0/
                          (1.0 + 0.4*Kokkos::sqrt(T2) + 0.2*T2 + 0.08*T2*T2)
                    : c.s.kgr_h2*c.s.zd;
      if (T > 7.0e2) {
        const Real inv_T = 1.0/T;
        const Real lt4 = Kokkos::log10(T*1e-4);
        const Real l9l = Kokkos::log10(6.67e-12*Kokkos::sqrt(T)*
                                       Kokkos::exp(-(1.0 + 63590.0*inv_T)));
        const Real l9h = Kokkos::log10(3.52e-9*Kokkos::exp(-43900.0*inv_T));
        const Real l10l = Kokkos::log10(5.996e-30*Kokkos::pow(T, 4.1881)/
                                        Kokkos::pow(1.0 + 6.761e-6*T, 5.6881)*
                                        Kokkos::exp(-54657.4*inv_T));
        const Real l10h = Kokkos::log10(1.3e-9*Kokkos::exp(-53300.0*inv_T));
        const Real ncr_h2 = Kokkos::pow(10.0, 4.845 - 1.3*lt4 + 1.62*lt4*lt4);
        const Real ncr_hi = Kokkos::pow(10.0, 3.0 - 0.416*lt4 - 0.327*lt4*lt4);
        const Real inv_ncr = Kokkos::fmax(sub.x_hi/ncr_hi + 2.0*y[IH2]/ncr_h2, 1e-50);
        const Real n2 = n_h*inv_ncr, w = n2/(1.0 + n2), w0 = 1.0/(1.0 + n2);
        sub.k_h2_h = Kokkos::pow(10.0, l9h*w + l9l*w0);
        sub.k_h2_h2 = Kokkos::pow(10.0, l10h*w + l10l*w0);
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

  //! Free electrons from H+, C+, O+ and molecular ions (Tigris CalculateFreeElectron,
  //! below TEMP_HOT0; the hot branch with CIE comes with P4)
  KOKKOS_INLINE_FUNCTION static Real FreeElectrons(const Real x_e_old, const Real x_h2,
                                                   const Real x_hii, const Real T,
                                                   const Cell &c) {
    const Real lnT = Kokkos::log(T);
    const Real x_cii = XCII(x_e_old, x_h2, T, lnT, c);
    return x_hii + x_cii + x_hii*ncr::X_O_STD*c.s.zg + XeMol(x_e_old, x_h2, T, c);
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
    y[IE] = FreeElectrons(y_n[IE], x_h2, x_hii, sub.T, cl);
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
    const Real n_h = c.n_h, x_e = y[IE], x_h2 = y[IH2], x_hi = sub.x_hi;
    const Real n_hi = n_h*x_hi, n_h2 = n_h*x_h2, n_e = n_h*x_e;
    const Real k_ph_h2 = c.xi_diss_h2;
    Real heating = Thermo::HeatingCR(x_e, n_h, x_hi, x_h2, c.xi_cr);
    heating += Thermo::HeatingPE(c.chi, c.s.zd, T, n_e);
    heating += Thermo::HeatingH2gr(x_hi, x_h2, n_h, T, sub.kgr*n_h, k_ph_h2);
    heating += Thermo::HeatingH2pump(x_hi, x_h2, n_h, T, k_ph_h2);
    heating += Thermo::HeatingH2diss(k_ph_h2, x_h2);
    Real cooling = Thermo::CoolingCII(sub.x_cii, n_hi, n_h2, n_e, T);
    cooling += Thermo::CoolingCI(sub.x_ci, n_hi, n_h2, n_e, T);
    cooling += Thermo::CoolingOI(sub.x_oi, n_hi, n_h2, n_e, T);
    if (T > 3.0e3) cooling += Thermo::CoolingLya(x_hi, n_e, T);
    if (sub.x_co > 0.0) {
      const Real vth = Kokkos::sqrt(2.*units::Units::k_boltzmann_cgs*T/
                                    units::Units::CO_mass_cgs);
      const Real grad_eff = Kokkos::fmax(c.grad_v, vth/c.s.leff_co_max);
      cooling += Thermo::CoolingCOR(sub.x_co, n_hi, n_h2, n_e, T,
                                    n_h*sub.x_co/grad_eff);
    }
    if (c.s.h2_rovib_cooling && T < c.s.temp_hot1) {
      cooling += ncr_terms::CoolingH2Moseley(x_h2, x_hi, n_h, T);
    }
    cooling += ncr_terms::CoolingHII(x_e, y[IH_PLUS], n_h, T, ncr::AlphaBHII(T));
    if (T < c.s.temp_hot1) {
      cooling += ncr_terms::CoolingNebular(x_e, y[IH_PLUS], n_h, T, c.s.zg);
    }
    cooling += Thermo::CoolingRec(c.s.zd, T, n_e, c.chi);
    cooling += Thermo::CoolingH2diss(x_hi, x_h2, sub.k_h2_h*n_h, sub.k_h2_h2*n_h);
    cooling += Thermo::CoolingHIion(x_hi, x_e, sub.k_coll_hi*n_h);
    if (c.s.dust_cooling) {
      cooling += dust::GasToDust(n_h, T, c.s.temp_dust, c.s.zd, c.s.alpha_gd);
    }
    return HeatCool{heating, cooling};
  }

  KOKKOS_INLINE_FUNCTION static void Closures(Real *, const Real *, const Cell &) {}
};

}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_NCR_HPP_
