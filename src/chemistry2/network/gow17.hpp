#ifndef CHEMISTRY2_NETWORK_GOW17_HPP_
#define CHEMISTRY2_NETWORK_GOW17_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gow17.hpp
//! \brief The GOW17 chemical network (Gong, Ostriker & Wolfire 2017): 12 stored species,
//!        neutral H, He, C, O, Si and electrons from element and charge conservation.
//!        Rates and heating and cooling as in src/chemistry/network/gow17.hpp.
//!
//!   Ions made by ionizing an unstored neutral are written in reservoir form,
//!   dx+/dt = a (x0 + x+) + C' - (a + D) x+, so that backward Euler also steps the
//!   neutral by backward Euler. CO and HCO+ are updated together (Update<ICO>).

#include <array>
#include <string>
#include <string_view>
#include <utility>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "chemistry2/inputs/cell_inputs.hpp"
#include "chemistry2/network/gow17_rates.hpp"
#include "chemistry2/network/gow17_table.hpp"
#include "chemistry2/network/network_config.hpp"
#include "chemistry2/network/species_table.hpp"
#include "chemistry2/thermo/dust.hpp"
#include "chemistry2/thermo/thermo.hpp"
#include "chemistry2/unit_factors.hpp"
#include "units/units.hpp"

namespace chemistry2 {

//! TABLE: temperature-only coefficients from the rate table (the default, "GOW17"), or
//! from the analytic fits ("GOW17_analytic", for comparisons with src/chemistry)
template <bool TABLE>
struct GOW17Network {
  static constexpr std::string_view NAME = TABLE ? "GOW17" : "GOW17_analytic";
  static constexpr std::array<SpeciesRow, 12> rows = {
      Species("He+").Charge(1).He(1),
      Species("OHx").H(1).O(1),
      Species("CHx").H(1).C(1),
      Species("CO").C(1).O(1),
      Species("C+").Charge(1).C(1),
      Species("HCO+").Charge(1).H(1).C(1).O(1),
      Species("H2").H(2),
      Species("H+").Charge(1).H(1),
      Species("H3+").Charge(1).H(3),
      Species("H2+").Charge(1).H(2),
      Species("O+").Charge(1).O(1),
      Species("Si+").Charge(1).Si(1)};
  static constexpr bool STORES_ELECTRONS = false;
  static constexpr bool EDOT_DERIVATIVE = TABLE;  // EdotWithDerivative, dual numbers
  static constexpr int ROW_EL = -1;
  static constexpr int HYDROGEN_REMAINDER_CHARGE = 0;
  enum : int {IHE_PLUS, IOHX, ICHX, ICO, IC_PLUS, IHCO_PLUS, IH2, IH_PLUS, IH3_PLUS,
              IH2_PLUS, IO_PLUS, ISI_PLUS};
  static constexpr int NSPEC = NumSpecies<GOW17Network>();
  static_assert(SpeciesIndex(rows, "HCO+") == IHCO_PLUS);
  static_assert(SpeciesIndex(rows, "Si+") == ISI_PLUS);

  // He+ feeds H+, C+, H2+ and O+; H2+ -> H3+ before the ions it creates; CHx and OHx
  // before CO; H2 last. HCO+ is updated with CO.
  using UpdateOrder = std::integer_sequence<int, IHE_PLUS, ISI_PLUS, IH2_PLUS, IH3_PLUS,
                                            IH_PLUS, IO_PLUS, IC_PLUS, ICHX, IOHX, ICO,
                                            IH2>;
  using StepLimitingSpecies = std::integer_sequence<int, IH2, ICO>;

  struct Settings {
    Real zd, x_he, x_c, x_o, x_si;
    Real t_min_rates, t_max_rates, t_max_heating, t_min_cooling, t_max_cooling_nm;
    Real leff_co_max;     // maximum effective length for CO cooling [cm]
    bool h2_rovib_cooling;
    bool kgr_h2_const;
    RowTable table;       // temperature-only coefficients (TABLE only)
    // gas-dust collisional exchange, at a fixed dust temperature [K]; alpha_gd <= 0
    // selects Hollenbach & McKee (1989)
    bool dust_cooling = false;
    Real temp_dust = 10.0;
    Real alpha_gd = 0.0;
    // He I + e -> He+ + 2e (Tigris, not in GOW17); off for parity with src/chemistry
    bool he_coll_ion = true;
  };

  static Settings ReadSettings(ParameterInput *pin, const std::string &b) {
    Settings s = ReadValues(pin, b);
    s.he_coll_ion = pin->GetOrAddBoolean(b, "GOW17_he_coll_ion", true);
    if constexpr (TABLE) s.table = gow17::BuildGOW17Table(s.kgr_h2_const);
    return s;
  }

  static Settings ReadValues(ParameterInput *pin, const std::string &b) {
    const Real inf = Kokkos::Experimental::infinity_v<Real>;
    const Real zg = pin->GetOrAddReal(b, "GOW17_Z_g", 1.0);
    return Settings{pin->GetOrAddReal(b, "GOW17_Z_d", 1.0),
                    pin->GetOrAddReal(b, "GOW17_xHe", 0.1),
                    zg*pin->GetOrAddReal(b, "GOW17_xC", 1.6e-4),
                    zg*pin->GetOrAddReal(b, "GOW17_xO", 3.2e-4),
                    zg*pin->GetOrAddReal(b, "GOW17_xSi", 1.7e-6),
                    pin->GetOrAddReal(b, "GOW17_temperature_min_rates", 1.0),
                    pin->GetOrAddReal(b, "GOW17_temperature_max_rates", inf),
                    pin->GetOrAddReal(b, "GOW17_temperature_max_heating", inf),
                    pin->GetOrAddReal(b, "GOW17_temperature_min_cooling", 1.0),
                    pin->GetOrAddReal(b, "GOW17_temperature_max_cooling_nm", 1.0e9),
                    pin->GetOrAddReal(b, "GOW17_Leff_CO_max", 3.0e20),
                    pin->GetOrAddBoolean(b, "GOW17_H2_rovib_cooling", true),
                    pin->GetOrAddBoolean(b, "is_kgrH2_const", false),
                    RowTable(),
                    pin->GetOrAddBoolean(b, "GOW17_dust_cooling", false),
                    pin->GetOrAddReal(b, "GOW17_dust_temperature", 10.0),
                    pin->GetOrAddReal(b, "GOW17_alpha_gd", 0.0)};
  }

  //! Constants of one cell over the hydro step
  struct Cell {
    Real n_h, inv_n_h, gamma;
    Real u_time, u_energy, inv_u_energy;
    Settings s;
    Real g0, xi_cr, grad_v;  // FUV field [Draine], CR rate [s^-1], |grad v| [s^-1]
    Real kph[gow17::N_PH];   // photo rates [s^-1]
    Real cr_heat_h2;         // heat per cosmic-ray ionization in molecular gas [erg]
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
    c.g0 = in.g0;
    c.xi_cr = in.xi_cr;
    c.grad_v = in.grad_v;
    constexpr Real kph_base[gow17::N_PH] = {3.5e-10, 9.1e-10, 2.4e-10,
                                            3.8e-10, 5.7e-11, 4.5e-9};
    for (int p = 0; p < gow17::N_PH; ++p) c.kph[p] = kph_base[p]*in.g0;
    c.cr_heat_h2 = Thermo::CRHeatPerH2(c.n_h);
    return c;
  }

  struct Ghosts {
    Real si, c, o, he, e, h;
  };

  //! One substep: ghosts at its start, T, and the rate coefficients
  struct Substep {
    Ghosts g;
    Real T;
    Slot slot;  // table rows of T (TABLE only)
    gow17::GOW17Rates r;
  };

  KOKKOS_INLINE_FUNCTION static Ghosts MakeGhosts(const Real *y, const Settings &s) {
    Ghosts g;
    g.si = ElementBudget<GOW17Network, ELEM_SI>(y, s.x_si);
    g.c = ElementBudget<GOW17Network, ELEM_C>(y, s.x_c);
    g.o = ElementBudget<GOW17Network, ELEM_O>(y, s.x_o);
    g.he = ElementBudget<GOW17Network, ELEM_HE>(y, s.x_he);
    g.e = Electrons<GOW17Network>(y);
    g.h = ElementBudget<GOW17Network, ELEM_H>(y, 1.0);
    return g;
  }

  //! Temperature [K] from e, before the rate clamps
  KOKKOS_INLINE_FUNCTION static Real RawTemperature(const Real *y, const Real e,
                                                    const Real x_e, const Cell &c) {
    return e*c.u_energy*c.inv_n_h/Thermo::CvCold(y[IH2], c.s.x_he, x_e, c.gamma);
  }

  KOKKOS_INLINE_FUNCTION static Substep Prepare(const Real *y, const Real e,
                                                const Cell &c) {
    Substep sub;
    sub.g = MakeGhosts(y, c.s);
    Real T = RawTemperature(y, e, sub.g.e, c);
    Real T_coll = T;
    if (T < c.s.t_min_rates) {
      T = c.s.t_min_rates;
      T_coll = T;
    } else if (T > c.s.t_max_rates) {
      T = c.s.t_max_rates;
    }
    sub.T = T;
    if constexpr (TABLE) {
      sub.slot = c.s.table.grid.Locate(T);
      const gow17::GOW17TemperatureRates tr =
          gow17::LookupRates(c.s.table, sub.slot, T, T_coll);
      sub.r = gow17::MakeGOW17Rates(tr, y[IH2], sub.g.h, sub.g.e, c.n_h, c.s.zd,
                                    c.xi_cr, c.g0);
    } else {
      const gow17::GOW17TemperatureRates tr(T, T_coll, c.s.kgr_h2_const);
      sub.r = gow17::MakeGOW17Rates(tr, y[IH2], sub.g.h, sub.g.e, c.n_h, c.s.zd,
                                    c.xi_cr, c.g0);
    }
    return sub;
  }

  //! Heating and cooling rates [erg s^-1 per H]
  struct HeatCool {
    Real heating, cooling;
  };

  //! de/dt in code units, with the substep's ghosts and rate coefficients
  KOKKOS_INLINE_FUNCTION static Real Edot(const Real *y, const Real e, const Cell &c,
                                          const Substep &sub) {
    const HeatCool hc = HeatingCooling(y, e, c, sub);
    return c.u_time*((hc.heating - hc.cooling)*c.n_h*c.inv_u_energy);
  }

  KOKKOS_INLINE_FUNCTION static HeatCool HeatingCooling(const Real *y, const Real e,
                                                        const Cell &c,
                                                        const Substep &sub) {
    const Ghosts &g = sub.g;
    const gow17::GOW17Rates &r = sub.r;
    Real T = RawTemperature(y, e, g.e, c);
    if (T < c.s.t_min_rates) T = c.s.t_min_rates;
    constexpr Real T_FLOOR = 1.0;
    if (T < T_FLOOR) return HeatCool{0.0, 0.0};
    const Real n_h = c.n_h;
    if constexpr (TABLE) {
      const HeatCoolT<Real> hc = HeatCoolTable<Real>(y, T, 1.0, c, sub);
      return HeatCool{hc.heating, hc.cooling};
    }
    // heating
    Real heating = 0.0;
    if (T <= c.s.t_max_heating) {
      heating = Thermo::HeatingCR(g.e, n_h, g.h, y[IH2], c.xi_cr);
      heating += Thermo::HeatingPE(c.g0, c.s.zd, T, n_h*g.e);
      const Real k_h2_photo = c.kph[gow17::IPH_H2];
      heating += Thermo::HeatingH2gr(g.h, y[IH2], n_h, T, r.kgr[gow17::IGR_H2],
                                     k_h2_photo);
      heating += Thermo::HeatingH2pump(g.h, y[IH2], n_h, T, k_h2_photo);
      heating += Thermo::HeatingH2diss(k_h2_photo, y[IH2]);
    }
    // cooling
    Real cooling = 0.0;
    if (T >= c.s.t_min_cooling) {
      const Real T_cap = Kokkos::fmin(T, c.s.t_max_cooling_nm);
      const Real n_hi = n_h*g.h, n_h2 = n_h*y[IH2], n_e = n_h*g.e;
      cooling = Thermo::CoolingCII(y[IC_PLUS], n_hi, n_h2, n_e, T_cap);
      cooling += Thermo::CoolingCI(g.c, n_hi, n_h2, n_e, T_cap);
      cooling += Thermo::CoolingOI(g.o, n_hi, n_h2, n_e, T_cap);
      cooling += Thermo::CoolingLya(g.h, n_e, T);
      const Real vth = Kokkos::sqrt(2.*units::Units::k_boltzmann_cgs*T_cap/
                                    units::Units::CO_mass_cgs);
      const Real grad_eff = Kokkos::fmax(c.grad_v, vth/c.s.leff_co_max);
      const Real n_co_eff = n_h*y[ICO]/grad_eff;
      cooling += Thermo::CoolingCOR(y[ICO], n_hi, n_h2, n_e, T_cap, n_co_eff);
      if (c.s.h2_rovib_cooling) {
        cooling += Thermo::CoolingH2(y[IH2], n_hi, n_h2, n_h*g.he, n_h*y[IH_PLUS], n_e,
                                     T_cap);
      }
      cooling += Thermo::CoolingRec(c.s.zd, T_cap, n_e, c.g0);
      cooling += Thermo::CoolingH2diss(g.h, y[IH2], r.k2[gow17::I2_H2_H],
                                       r.k2[gow17::I2_H2_H2]);
      cooling += Thermo::CoolingHIion(g.h, g.e, r.k2[gow17::I2_H_E]);
      if (c.s.he_coll_ion) {  // 24.587 eV per He I ionization
        cooling += gow17::CHI_HEI_ERG*r.k2[gow17::I2_HE_E]*g.e*g.he;
      }
      if (c.s.dust_cooling) {
        cooling += dust::GasToDust(n_h, T, c.s.temp_dust, c.s.zd, c.s.alpha_gd);
      }
    }
    return HeatCool{heating, cooling};
  }

  //! edot in code units, and d(edot)/de at fixed species, from one dual-number pass
  KOKKOS_INLINE_FUNCTION static Real EdotWithDerivative(const Real *y, const Real e,
                                                        const Cell &c,
                                                        const Substep &sub,
                                                        Real *dedot_de) {
    const Real dT_de = c.u_energy*c.inv_n_h/
                       Thermo::CvCold(y[IH2], c.s.x_he, sub.g.e, c.gamma);
    const Real T_raw = e*dT_de;
    const Real T = Kokkos::fmax(T_raw, c.s.t_min_rates);
    constexpr Real T_FLOOR = 1.0;
    *dedot_de = 0.0;
    if (T < T_FLOOR) return 0.0;
    const Real seed = (T_raw >= c.s.t_min_rates) ? 1.0 : 0.0;
    const auto hc = HeatCoolTable<dual::Dual>(y, T, seed, c, sub);
    const Real fac = c.u_time*c.n_h*c.inv_u_energy;
    *dedot_de = fac*(hc.heating.d - hc.cooling.d)*dT_de;
    const Real edot = fac*(hc.heating.v - hc.cooling.v);
    // Below t_min_rates the rates are frozen at t_min_rates and the derivative is 0, so
    // net cooling would shrink e geometrically, one substep each, up to nsub_max: no
    // cooling below the rate floor
    if (seed == 0.0 && edot < 0.0) return 0.0;
    return edot;
  }

  template <class S>
  struct HeatCoolT {
    S heating, cooling;
  };

  //! Heating and cooling with the temperature-only coefficients from the table. With
  //! S = dual::Dual the derivative part is d/dT at fixed species, seeded by seed (0
  //! where T is clamped at t_min_rates)
  template <class S>
  KOKKOS_INLINE_FUNCTION static HeatCoolT<S> HeatCoolTable(const Real *y, const Real T,
                                                           const Real seed,
                                                           const Cell &c,
                                                           const Substep &sub) {
    constexpr bool DERIV = std::is_same_v<S, dual::Dual>;
    using gow17::RowCoef;
    const Ghosts &g = sub.g;
    const gow17::GOW17Rates &r = sub.r;
    const RowTable &tab = c.s.table;
    const Real n_h = c.n_h;
    Real dw_t = 0.0;
    Slot s_t;
    if constexpr (DERIV) {
      s_t = tab.grid.Locate(T, &dw_t);
      dw_t *= seed;
    } else {
      s_t = (T == sub.T) ? sub.slot : tab.grid.Locate(T);
    }
    const bool capped = T > c.s.t_max_cooling_nm;
    const Real T_cap = capped ? c.s.t_max_cooling_nm : T;
    const Slot s_cap = capped ? tab.grid.Locate(T_cap) : s_t;
    const Real dw_cap = capped ? 0.0 : dw_t;
    S Ts, Ts_cap;
    if constexpr (DERIV) {
      Ts = dual::Dual(T, seed);
      Ts_cap = dual::Dual(T_cap, capped ? 0.0 : seed);
    } else {
      Ts = T;
      Ts_cap = T_cap;
    }
    auto col = [&](const int k) { return RowCoef<S>(tab, s_t, dw_t, k); };
    S heating = 0.0;
    if (T <= c.s.t_max_heating) {
      const Thermo::H2GeffRateCoefficientsT<S> geff{col(gow17::C_H2GEFF),
                                                    col(gow17::C_H2GEFF + 1)};
      S kgr = r.kgr[gow17::IGR_H2];
      if constexpr (DERIV) {
        if (T <= c.s.t_max_rates) {
          kgr.d = n_h*c.s.zd*col(gow17::C_KGR_H2).d;
        }
      }
      heating = Thermo::HeatingCRFrom(g.e, g.h, y[IH2], c.xi_cr, c.cr_heat_h2);
      heating += Thermo::HeatingPE(c.g0, c.s.zd, Ts, n_h*g.e);
      const Real k_h2_photo = c.kph[gow17::IPH_H2];
      heating += Thermo::HeatingH2grFrom(g.h, y[IH2], n_h, kgr, k_h2_photo, geff);
      heating += Thermo::HeatingH2pumpFrom(g.h, y[IH2], n_h, k_h2_photo, geff);
      heating += Thermo::HeatingH2diss(k_h2_photo, y[IH2]);
    }
    S cooling = 0.0;
    if (T >= c.s.t_min_cooling) {
      const gow17::GOW17ThermoCoefficients<S> k =
          gow17::LookupThermo<S>(tab, s_cap, dw_cap);
      const Real n_hi = n_h*g.h, n_h2 = n_h*y[IH2], n_e = n_h*g.e;
      cooling = Thermo::CoolingCIIFrom(y[IC_PLUS], n_hi, n_h2, n_e, k.cii, k.cii_boltz);
      if (T_cap <= 1.0e6) {
        cooling += Thermo::CoolingCIFrom(g.c, n_hi, n_h2, n_e, k.ci, k.ci_boltz[0],
                                         k.ci_boltz[1], k.ci_boltz[2]);
      }
      cooling += Thermo::CoolingOIFrom(g.o, n_hi, n_h2, n_e, k.oi, k.oi_boltz[0],
                                       k.oi_boltz[1], k.oi_boltz[2]);
      cooling += Thermo::CoolingLyaFrom(g.h, n_e, col(gow17::C_LYA),
                                        col(gow17::C_LYA + 1));
      const S vth = dual::sqrt((2.*units::Units::k_boltzmann_cgs/
                                units::Units::CO_mass_cgs)*Ts_cap);
      const S grad_eff = dual::fmax(vth*(1.0/c.s.leff_co_max), c.grad_v);
      const S n_co_eff = (n_h*y[ICO])/grad_eff;
      cooling += Thermo::CoolingCOR(y[ICO], n_hi, n_h2, n_e, Ts_cap, n_co_eff);
      if (c.s.h2_rovib_cooling) {
        cooling += Thermo::CoolingH2From(y[IH2], n_hi, n_h2, n_h*g.he, n_h*y[IH_PLUS],
                                         n_e, k.h2cool);
      }
      cooling += Thermo::CoolingRec(c.s.zd, Ts_cap, n_e, c.g0);
      S k_h2_h = r.k2[gow17::I2_H2_H], k_h2_h2 = r.k2[gow17::I2_H2_H2];
      S k_h_e = r.k2[gow17::I2_H_E];
      if constexpr (DERIV) {
        if (T > 7.0e2) {  // collisional terms active (T here is Prepare's T_coll)
          // derivative of the exponent only; the values stay those of Prepare
          constexpr Real LN10 = 2.30258509299404590e0;
          constexpr Real SMALL = gow17::SMALL_REAL;
          const S div = g.h/(col(gow17::C_NCR_H) + SMALL) +
                        y[IH2]/(col(gow17::C_NCR_H2) + SMALL);
          const S n2ncr = (div < SMALL) ? S(n_h*SMALL) : n_h*div;
          const S inv = 1.0/(1.0 + n2ncr);
          const S e9 = (col(gow17::C_LOG10_K9H)*n2ncr + col(gow17::C_LOG10_K9L))*inv;
          const S e10 = (col(gow17::C_LOG10_K10H)*n2ncr + col(gow17::C_LOG10_K10L))*inv;
          k_h2_h.d = k_h2_h.v*LN10*e9.d;
          k_h2_h2.d = k_h2_h2.v*LN10*e10.d;
          k_h_e.d = n_h*col(gow17::C_K2 + gow17::I2_H_E).d;  // sub.r.k2 carries n_H
        }
      }
      cooling += Thermo::CoolingH2diss(g.h, y[IH2], k_h2_h, k_h2_h2);
      cooling += Thermo::CoolingHIion(g.h, g.e, k_h_e);
      if (c.s.he_coll_ion) {  // 24.587 eV per He I ionization; value from Prepare
        S k_he_e = r.k2[gow17::I2_HE_E];
        if constexpr (DERIV) k_he_e.d = n_h*col(gow17::C_K2 + gow17::I2_HE_E).d;
        cooling += gow17::CHI_HEI_ERG*k_he_e*(g.e*g.he);
      }
      if (c.s.dust_cooling) {
        cooling += dust::GasToDust(n_h, Ts, c.s.temp_dust, c.s.zd, c.s.alpha_gd);
      }
    }
    return HeatCoolT<S>{heating, cooling};
  }

  //! Creation c [per H per code time] and destruction d [per code time] of species K
  template <int K>
  KOKKOS_INLINE_FUNCTION static CD Rates(const Real *y, Real, const Cell &cl,
                                         const Substep &sub) {
    using namespace gow17;  // NOLINT(build/namespaces)
    const Ghosts &g = sub.g;
    const Real *k = sub.r.k2;
    const Real *kcr = sub.r.kcr;
    const Real *kgr = sub.r.kgr;
    const Real *kph = cl.kph;
    Real c, d;
    if constexpr (K == IHE_PLUS) {
      const Real a = kcr[ICR_HE] + (cl.s.he_coll_ion ? k[I2_HE_E]*g.e : 0.0);
      c = a*(g.he + y[IHE_PLUS]);
      d = k[I2_HEP_H2]*y[IH2] + k[I2_HEP_CO]*y[ICO] + k[I2_HEP_E]*g.e +
          k[I2_HEP_H2_H2P]*y[IH2] + k[I2_HEP_OH]*y[IOHX] + kgr[IGR_HEP] + a;
    } else if constexpr (K == ISI_PLUS) {
      const Real a = kcr[ICR_SI] + kph[IPH_SI];
      c = a*(g.si + y[ISI_PLUS]);
      d = k[I2_SIP_E]*g.e + kgr[IGR_SIP] + a;
    } else if constexpr (K == IH2_PLUS) {
      c = kcr[ICR_H2]*y[IH2] + k[I2_HEP_H2_H2P]*y[IHE_PLUS]*y[IH2];
      d = k[I2_H2P_H2]*y[IH2] + k[I2_H2P_H]*g.h;
    } else if constexpr (K == IH3_PLUS) {
      c = k[I2_H2P_H2]*y[IH2_PLUS]*y[IH2];
      d = k[I2_H3P_C]*g.c + k[I2_H3P_O]*g.o + k[I2_H3P_CO]*y[ICO] + k[I2_H3P_E]*g.e +
          k[I2_H3P_E_3H]*g.e + k[I2_H3P_O_H2]*g.o;
    } else if constexpr (K == IH_PLUS) {
      const Real a = kcr[ICR_H] + k[I2_H_E]*g.e;
      c = a*(g.h + y[IH_PLUS]) + k[I2_HEP_H2]*y[IHE_PLUS]*y[IH2] +
          k[I2_H2P_H]*y[IH2_PLUS]*g.h + k[I2_OP_H]*y[IO_PLUS]*g.h;
      d = k[I2_HP_E]*g.e + k[I2_HP_O]*g.o + kgr[IGR_HP] + a;
    } else if constexpr (K == IO_PLUS) {
      const Real a = k[I2_HP_O]*y[IH_PLUS];
      c = k[I2_HEP_OH]*y[IHE_PLUS]*y[IOHX] + a*(g.o + y[IO_PLUS]);
      d = k[I2_OP_H]*g.h + k[I2_OP_H2_OH]*y[IH2] + k[I2_OP_H2]*y[IH2] + a;
    } else if constexpr (K == IC_PLUS) {
      const Real a = kcr[ICR_C] + kph[IPH_C];
      c = a*(g.c + y[IC_PLUS]) + k[I2_HEP_CO]*y[IHE_PLUS]*y[ICO];
      d = k[I2_CP_H2]*y[IH2] + k[I2_CP_OH]*y[IOHX] + k[I2_CP_E]*g.e +
          k[I2_CP_H2_E]*y[IH2] + kgr[IGR_CP] + a;
    } else if constexpr (K == ICHX) {
      c = k[I2_H3P_C]*y[IH3_PLUS]*g.c + k[I2_CP_H2]*y[IC_PLUS]*y[IH2];
      d = k[I2_CH_O]*g.o + k[I2_CH_H]*g.h + kph[IPH_CHX];
    } else if constexpr (K == IOHX) {
      c = k[I2_H3P_O]*y[IH3_PLUS]*g.o + k[I2_OP_H2_OH]*y[IO_PLUS]*y[IH2];
      d = k[I2_CP_OH]*y[IC_PLUS] + k[I2_OH_C]*g.c + k[I2_OH_O]*g.o +
          k[I2_HEP_OH]*y[IHE_PLUS] + kph[IPH_OHX];
    } else if constexpr (K == ICO) {
      // CO's own terms, for the step limit; the update is Update<ICO>
      c = k[I2_CH_O]*y[ICHX]*g.o + k[I2_OH_C]*y[IOHX]*g.c + k[I2_HCOP_E]*y[IHCO_PLUS]*g.e;
      d = kcr[ICR_CO_PHOTO] + kcr[ICR_CO] + k[I2_H3P_CO]*y[IH3_PLUS] +
          k[I2_HEP_CO]*y[IHE_PLUS] + kph[IPH_CO];
    } else if constexpr (K == IHCO_PLUS) {
      // HCO+'s own terms; its update is part of Update<ICO>
      c = k[I2_CP_OH]*y[IC_PLUS]*y[IOHX] + kcr[ICR_CO]*y[ICO] +
          k[I2_H3P_CO]*y[IH3_PLUS]*y[ICO];
      d = k[I2_HCOP_E]*g.e;
    } else if constexpr (K == IH2) {
      c = k[I2_H3P_C]*y[IH3_PLUS]*g.c + k[I2_H3P_O]*y[IH3_PLUS]*g.o +
          k[I2_H3P_CO]*y[IH3_PLUS]*y[ICO] + k[I2_H3P_E]*y[IH3_PLUS]*g.e +
          k[I2_H2_H2]*y[IH2]*y[IH2] + k[I2_CH_H]*y[ICHX]*g.h +
          k[I2_H3P_O_H2]*y[IH3_PLUS]*g.o + k[I2_H2P_H]*y[IH2_PLUS]*g.h +
          kgr[IGR_H2]*g.h;
      d = kcr[ICR_H2] + k[I2_HEP_H2]*y[IHE_PLUS] + k[I2_CP_H2]*y[IC_PLUS] +
          k[I2_H2P_H2]*y[IH2_PLUS] + k[I2_H2_H]*g.h + 2.0*k[I2_H2_H2]*y[IH2] +
          k[I2_HEP_H2_H2P]*y[IHE_PLUS] + k[I2_CP_H2_E]*y[IC_PLUS] +
          k[I2_OP_H2_OH]*y[IO_PLUS] + k[I2_OP_H2]*y[IO_PLUS] + kph[IPH_H2];
    } else {
      static_assert(K < 0, "GOW17 has no rates for this species");
    }
    return CD{cl.u_time*c, cl.u_time*d};
  }

  //! Updates that are not one backward Euler step of one species
  template <int K>
  static constexpr bool CUSTOM_UPDATE = (K == ICO);

  template <int K>
  KOKKOS_INLINE_FUNCTION static void Update(Real *y, const Real *y_n, const Real,
                                            const Cell &cl, const Substep &sub,
                                            const Real h) {
    using namespace gow17;  // NOLINT(build/namespaces)
    const Ghosts &g = sub.g;
    const Real *k = sub.r.k2;
    const Real *kcr = sub.r.kcr;
    const Real u = cl.u_time;
    if constexpr (K == ICO) {
      // CO and HCO+ by backward Euler on the 2x2 system
      //   (1 + d1 h) CO - b h HCO+ = CO^n + a h
      //   -cc h CO + (1 + d2 h) HCO+ = HCO+^n + f h
      // b <= d2 and cc <= d1, so det >= 1 + (d1 + d2) h > 0.
      const Real a = u*(k[I2_CH_O]*y[ICHX]*g.o + k[I2_OH_C]*y[IOHX]*g.c);
      const Real b = u*k[I2_HCOP_E]*g.e;
      const Real d1 = u*(kcr[ICR_CO_PHOTO] + kcr[ICR_CO] + k[I2_H3P_CO]*y[IH3_PLUS] +
                         k[I2_HEP_CO]*y[IHE_PLUS] + cl.kph[IPH_CO]);
      const Real f = u*k[I2_CP_OH]*y[IC_PLUS]*y[IOHX];
      const Real cc = u*(kcr[ICR_CO] + k[I2_H3P_CO]*y[IH3_PLUS]);
      const Real d2 = u*k[I2_HCOP_E]*g.e;
      const Real A = 1.0 + d1*h;
      const Real B = 1.0 + d2*h;
      const Real det = A*B - b*cc*h*h;
      y[ICO] = (B*(y_n[ICO] + a*h) + b*h*(y_n[IHCO_PLUS] + f*h))/det;
      y[IHCO_PLUS] = (y_n[IHCO_PLUS] + f*h + cc*h*y[ICO])/B;
    }
  }

  KOKKOS_INLINE_FUNCTION static void Closures(Real *, const Real *, const Cell &) {}
};

using GOW17 = GOW17Network<true>;
using GOW17Analytic = GOW17Network<false>;

}  // namespace chemistry2

#endif  // CHEMISTRY2_NETWORK_GOW17_HPP_
