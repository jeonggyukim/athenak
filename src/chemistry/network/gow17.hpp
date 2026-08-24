#ifndef CHEMISTRY_NETWORK_GOW17_HPP_
#define CHEMISTRY_NETWORK_GOW17_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gow17.hpp
//  \brief The implementation for the struct for the GOW17 chemistry network

#include <Kokkos_NumericTraits.hpp>

#include "athena.hpp"
#include "chemistry/chemistry_utils.hpp"
#include "chemistry/thermo/thermo.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
struct GOW17Settings {
  /// If we're using an isothermal equation of state
  bool isothermal;
  /// The temperature to use for an isothermal EOS
  Real isothermal_temperature;
  /// Dust metallicity
  Real zd;
  /// He abundance per H
  Real xHe;
  /// C abundance at Z=1
  Real xC;
  /// O abundance at Z=1
  Real xO;
  /// Si abundance at Z=1
  Real xSi;
  /// Build the Jacobian with the temperature-only rates hoisted out of the
  /// species columns, restoring the thermal coupling as a rank-1 update.
  /// Off by default so the two paths can be compared in one binary.
  bool jacobian_hoist;
  /// Minimum temperature for reaction rates, also applied to energy equation
  Real temperature_min_rates;
  /// Maximum temperature for reaction rates
  Real temperature_max_rates;
  /// Temperature above which heating is turned off
  Real temperature_max_heating;
  /// Temperature below which cooling is turned off
  Real temperature_min_cooling;
  /// Cooling for neutral medium is capped at this temperature
  Real temperature_max_cooling_nm;
  /// maximum effective length for CO cooling in cm
  Real Leff_CO_max;
  /// Whether or not to use H2 rovibrational cooling
  bool H2_rovib_cooling;
  // H2 formation rate on grains
  bool is_kgrH2_const;

  Real velocity_cgs;
  Real length_cgs;
  bool three_d;
  bool multi_d;
};

/*!
 * \brief The class for the GOW17 network.
 *
 * \details All chemistry networks are required to have a handful of specific
 * things so that the ODE solvers have a common interface to work with. They
 * need:
 *   - A `neqs` variable that specifies the number of equations. That should be
 *     the number of species plus 1 for the internal energy .
 *   - `y` and `f` RegisterArray variables to hold the current state and the
 *     result of evaluating the equations respectively.
 *   - An `evaluate_function` method that computes `f` from `y`
 *
 * General notes:
 *   - enums indicting positive ions (like H+) use either "p" or "_plus" instead
 * of "+" since
 *     "+" isn't a legal character in that case
 */
class GOW17Network {
 public:
  KOKKOS_FUNCTION GOW17Network(GOW17Settings const settings, const int mb_idx,
                               const int k, const int j, const int i,
                               const DvceArray5D<Real> w0,
                               const DualArray1D<RegionSize> sizes,
                               const DvceArray1D<Real> ir,
                               Real const density_cgs, Real const mu_H,
                               Real const gamma, Real const hydrogen_mass_cgs,
                               Real const units_time_cgs,
                               Real const units_energy_density_cgs)
      : n_H(w0(mb_idx, IDN, k, j, i) * density_cgs /
            (mu_H * hydrogen_mass_cgs)),
        gamma(gamma),
        units_time_cgs(units_time_cgs),
        units_energy_density_cgs(units_energy_density_cgs),
        isothermal(settings.isothermal),
        zd(settings.zd),
        xHe(settings.xHe),
        xC(settings.xC),
        xO(settings.xO),
        xSi(settings.xSi),
        jacobian_hoist(settings.jacobian_hoist),
        temperature_min_rates(settings.temperature_min_rates),
        temperature_max_rates(settings.temperature_max_rates),
        temperature_max_heating(settings.temperature_max_heating),
        temperature_min_cooling(settings.temperature_min_cooling),
        temperature_max_cooling_nm(settings.temperature_max_cooling_nm),
        Leff_CO_max(settings.Leff_CO_max),
        H2_rovib_cooling(settings.H2_rovib_cooling),
        isothermal_temperature_(settings.isothermal_temperature),
        rad_(ir),
        is_kgrH2_const(settings.is_kgrH2_const),
        gradv_(SetGradv_(mb_idx, k, j, i, w0, sizes, settings.velocity_cgs,
                         settings.length_cgs, settings.multi_d,
                         settings.three_d)),
        y(y_buffer_, neqs),
        y_new(y_new_buffer_, neqs) {}

  // ----- Number of equations -----
  static constexpr int neqs = 13;

  /// If the network is using an isothermal equation of state
  const bool isothermal;

  // ----- Views to store ODE state -----
  // The current state
  Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> y;
  // The results of evaluating the ODEs
  Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> y_new;

  // ----- Species indices within the ODE system ------
  // The order must be: real species then internal energy
  enum : size_t {
    IHE_plus,   // He+
    IOHx,       // OHx
    ICHx,       // CHx
    ICO,        // CO
    IC_plus,    // C+
    IHCO_plus,  // HCO+
    IH2,        // H2
    IH_plus,    // H+
    IH3_plus,   // H3+
    IH2_plus,   // H2+
    IO_plus,    // O+
    ISi_plus,   // Si+
    IIE,        // internal energy, must be last
  };

  // ----- Names, used for output, must be the same order as the enum -----
  static constexpr std::array<std::string_view, neqs - 1> species_names = {
      "He+", "OHx", "CHx", "CO",  "C+", "HCO+",
      "H2",  "H+",  "H3+", "H2+", "O+", "Si+"};

  /*!
   * \brief Abundances of the ghost species
   *
   * \details The ghost species aren't integrated by the ODE solver; they're
   * algebraic functions of the real species through the conservation laws and
   * are recomputed from the current state on every function evaluation.
   */
  struct GhostSpecies {
    Real Si, C, O, He, e, H;
  };

  // ----- cell values -----
  Real const n_H;  // The number density of hydrogen
  Real const gamma;

  // ----- unit conversion factors -----
  Real const units_time_cgs;
  Real const units_energy_density_cgs;

  // ----- Metallicity factors -----
  const Real zd;   /// Dust metallicity
  const Real xHe;  /// He abundance per H
  const Real xC;   /// C abundance at Z=1
  const Real xO;   /// O abundance at Z=1
  const Real xSi;  /// Si abundance at Z=1

  // ----- Temperature Variables -----
  /// Hoist the temperature-only rates out of the Jacobian's species columns
  const bool jacobian_hoist;
  /// Minimum temperature for reaction rates, also applied to energy equation
  const Real temperature_min_rates;
  /// Maximum temperature for reaction rates. Does not apply for collisional
  /// dissociation reactions and energy equation
  const Real temperature_max_rates;
  /// Temperature above which heating is turned off
  const Real temperature_max_heating;
  /// Temperature below which cooling is turned off
  const Real temperature_min_cooling;
  /// Cooling for neutral medium is capped at this temperature
  const Real temperature_max_cooling_nm;

  // ----- Chemical Settings -----
  /// maximum effective length for CO cooling in cm
  const Real Leff_CO_max;
  /// Whether or not to use H2 rovibrational cooling
  const bool H2_rovib_cooling;
  // H2 formation rate on grains
  const bool is_kgrH2_const;

  // ----- Radiation Constants -----
  static constexpr int n_ph = 6;
  static constexpr int n_freq = n_ph + 2;

  // ----- Member Functions -----
  /*!
   * \brief Get the settings for the GOW17 network from the input file
   *
   * \param pin The ParameterInput object
   * \return GOW17Settings The settings for the GOW17 network
   */
  static GOW17Settings GetSettings(ParameterInput* pin, MeshBlockPack* ppack) {
    // Get the parameters from input file
    GOW17Settings output;

    // Dust metallicity
    output.zd = pin->GetOrAddReal("chemistry", "GOW17_Z_d", 1.0);
    // Gas metallicity
    Real zg = pin->GetOrAddReal("chemistry", "GOW17_Z_g", 1.0);
    // He abundance per H
    output.xHe = pin->GetOrAddReal("chemistry", "GOW17_xHe", 0.1);
    // C abundance at Z=1
    output.xC = zg * pin->GetOrAddReal("chemistry", "GOW17_xC", 1.6e-4);
    // O abundance at Z=1
    output.xO = zg * pin->GetOrAddReal("chemistry", "GOW17_xO", 3.2e-4);
    // Si abundance at Z=1
    output.xSi = zg * pin->GetOrAddReal("chemistry", "GOW17_xSi", 1.7e-6);

    // Isothermal EOS?
    output.isothermal =
        pin->GetOrAddBoolean("chemistry", "GOW17_isothermal", false);
    if (output.isothermal) {
      const Real mu_iso = pin->GetReal("chemistry", "GOW17_mu_iso");
      const Real cs = pin->GetReal("hydro", "iso_sound_speed");
      const Real temperature_mu_cgs =
          ppack->punit->pressure_cgs() / ppack->punit->density_cgs() *
          units::Units::hydrogen_mass_cgs / units::Units::k_boltzmann_cgs;
      output.isothermal_temperature = cs * cs * mu_iso * temperature_mu_cgs;
    } else {
      output.isothermal_temperature =
          Kokkos::Experimental::signaling_NaN_v<Real>;
    }

    Real inf = Kokkos::Experimental::infinity_v<Real>;
    output.jacobian_hoist =
        pin->GetOrAddBoolean("chemistry", "GOW17_jacobian_hoist", false);
    output.temperature_min_rates =
        pin->GetOrAddReal("chemistry", "GOW17_temperature_min_rates", 1.0);
    output.temperature_max_rates =
        pin->GetOrAddReal("chemistry", "GOW17_temperature_max_rates", inf);
    output.temperature_max_heating =
        pin->GetOrAddReal("chemistry", "GOW17_temperature_max_heating", inf);
    output.temperature_min_cooling =
        pin->GetOrAddReal("chemistry", "GOW17_temperature_min_cooling", 1.);
    output.temperature_max_cooling_nm = pin->GetOrAddReal(
        "chemistry", "GOW17_temperature_max_cooling_nm", 1.0e9);

    // Chemical Settings
    output.Leff_CO_max =
        pin->GetOrAddReal("chemistry", "GOW17_Leff_CO_max", 3.0e20);
    output.H2_rovib_cooling =
        pin->GetOrAddBoolean("chemistry", "GOW17_H2_rovib_cooling", true);
    // H2 formation rate on grains
    output.is_kgrH2_const =
        pin->GetOrAddBoolean("chemistry", "is_kgrH2_const", false);

    output.velocity_cgs = ppack->punit->velocity_cgs();
    output.length_cgs = ppack->punit->length_cgs();
    output.multi_d = ppack->pmesh->multi_d;
    output.three_d = ppack->pmesh->three_d;
    return output;
  }

  /*!
   * \brief Compute the temperature in the cell
   *
   * \param y_in The current state to compute the temperature from
   * \param ghosts The ghost species abundances
   * \return Real The temperature in the cell
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real Temperature(const vec_type& y_in,
                                   const GhostSpecies& ghosts) const {
    // energy per hydrogen atom
    const Real E_ergs = y_in(IIE) * units_energy_density_cgs / n_H;

    // Temperature
    Real T = E_ergs / Thermo::CvCold(y_in[IH2], xHe, ghosts.e, gamma);

    // apply temperature floor, incase of very small or negative energy
    if (T < temperature_min_rates) {
      T = temperature_min_rates;
    }

    return T;
  }

  /*!
   * \brief Compute the cooling term from the temperature
   *
   * \param y_in The current state to compute the cooling from
   * \param ghosts The ghost species abundances
   * \param T The temperature in the cell
   * \return Real The cooling term, i.e. how much the energy decreases. Note
   * that this is a positive value so the energy update should look like `E =
   * HeatingTerm() - CoolingTerm();`
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real CoolingTerm(const vec_type& y_in,
                                   const GhostSpecies& ghosts, Real T) const {
    // Check that the temperature is below the maximum allowed for neutral
    // medium. If above then set it to T_max_NM
    Real const T_capped = Kokkos::fmin(T, temperature_max_cooling_nm);

    // cut-off cooling at low temperature
    if (T < temperature_min_cooling) {
      return 0;
    }

    // C+ fine structure line
    Real cooling =
        Thermo::CoolingCII(y_in[IC_plus], n_H * ghosts.H, n_H * y_in[IH2],
                           n_H * ghosts.e, T_capped);
    // CI fine structure line
    cooling += Thermo::CoolingCI(ghosts.C, n_H * ghosts.H, n_H * y_in[IH2],
                                 n_H * ghosts.e, T_capped);
    // OI fine structure line
    cooling += Thermo::CoolingOI(ghosts.O, n_H * ghosts.H, n_H * y_in[IH2],
                                 n_H * ghosts.e, T_capped);
    // cooling of hot gas: radiative cooling, free-free.
    cooling += Thermo::CoolingLya(ghosts.H, n_H * ghosts.e, T);
    //  CO rotational lines
    //  Calculate effective CO column density
    const Real vth = Kokkos::sqrt(2. * units::Units::k_boltzmann_cgs *
                                  T_capped / units::Units::CO_mass_cgs);
    const Real nCO = n_H * y_in[ICO];
    const Real grad_small_ = vth / Leff_CO_max;
    const Real gradeff = Kokkos::fmax(gradv_, grad_small_);
    const Real NCOeff = nCO / gradeff;
    cooling += Thermo::CoolingCOR(y_in[ICO], n_H * ghosts.H, n_H * y_in[IH2],
                                  n_H * ghosts.e, T_capped, NCOeff);
    // H2 vibration and rotation lines
    if (H2_rovib_cooling) {
      cooling += Thermo::CoolingH2(y_in[IH2], n_H * ghosts.H, n_H * y_in[IH2],
                                   n_H * ghosts.He, n_H * y_in[IH_plus],
                                   n_H * ghosts.e, T_capped);
    }
    // dust thermo emission. Disabled because our simulation does not go to high
    // enough density (>~ 10^5 cm-3) for dust cooling to matter.
    // cooling += 0.;  // Thermo::CoolingDustTd(zd,  n_H, T, 10.);

    // recombination of e on PAHs
    cooling += Thermo::CoolingRec(zd, T_capped, n_H * ghosts.e, rad_(irad_GPE));
    // collisional dissociation of H2
    cooling += Thermo::CoolingH2diss(ghosts.H, y_in[IH2], k2body_[i2body_H2_H],
                                     k2body_[i2body_H2_H2]);
    // collisional ionization of HI
    cooling += Thermo::CoolingHIion(ghosts.H, ghosts.e, k2body_[i2body_H_e]);

    return cooling;
  }

  /*!
   * \brief Compute the heating term from the temperature.
   *
   * \param y_in The current state to compute the heating from
   * \param ghosts The ghost species abundances
   * \param T The temperature in the cell
   * \return Real The heating term, i.e. how much the energy increases. The
   * energy update should look like `E = HeatingTerm() - CoolingTerm();`
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real HeatingTerm(const vec_type& y_in,
                                   const GhostSpecies& ghosts,
                                   Real const& T) const {
    if (T > temperature_max_heating) {
      return 0.0;
    }

    // Cosmic ray heating
    Real heating =
        Thermo::HeatingCr(ghosts.e, n_H, ghosts.H, y_in[IH2], rad_(irad_CR));

    // photo electric effect on dust
    heating += Thermo::HeatingPE(rad_(irad_GPE), zd, T, n_H * ghosts.e);

    // H2 formation on dust grains
    const Real k_xH2_photo = kph_[iph_H2];
    heating += Thermo::HeatingH2gr(ghosts.H, y_in[IH2], n_H, T, kgr_[igr_H],
                                   k_xH2_photo);

    // H2 UV pumping
    heating += Thermo::HeatingH2pump(ghosts.H, y_in[IH2], n_H, T, k_xH2_photo);

    // H2 Photodissociation
    heating += Thermo::HeatingH2diss(k_xH2_photo, y_in[IH2]);

    return heating;
  }

  /*!
   * \brief Evaluate the internal energy equation
   *
   * \param y_in The current state to evaluate the internal energy equation from
   * \param ghosts The ghost species abundances
   * \return Real The result of evaluating the internal energy equation
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real Edot(const vec_type& y_in,
                            const GhostSpecies& ghosts) const {
    if (isothermal) {
      return 0.0;
    }

    const Real T = Temperature(y_in, ghosts);

    static constexpr Real T_floor = 1.0;  // temperature floor for cooling
    if (T < T_floor) {
      return 0;
    } else {
      const Real dEdt =
          HeatingTerm(y_in, ghosts, T) - CoolingTerm(y_in, ghosts, T);
      // convert to code units
      return units_time_cgs * (dEdt * n_H / units_energy_density_cgs);
    }
  }

  /*!
   * \brief Compute the creation and destruction rates. These are used like
   * `f(i) = rates.creation(i) - y_in(i) * rates.destruction(i);`, i.e. the
   * destruction rates do not include the abundance of the species being
   * destroyed.
   *
   * \details Creation of ghost species isn't tracked since the ghosts are
   * recomputed from the real species on every function evaluation. Ghost
   * reactants contribute their abundance to the reaction rate but accumulate
   * no destruction of their own.
   *
   * \param y_in The current state to compute the rates from
   * \param ghosts The ghost species abundances
   * \return CDRates_t A struct containing the creation and destruction rate
   * arrays.
   */
  template <class vec_type>
  KOKKOS_FUNCTION CDRates_t<neqs - 1> CDRates(
      const vec_type& y_in, const GhostSpecies& ghosts) const {
    // Create and zero out rate arrays
    CDRates_t<neqs - 1> rates;
    for (size_t i = 0; i < rates.creation.size(); i++) {
      rates.creation(i) = 0.0;
      rates.destruction(i) = 0.0;
    }

    // ----- cosmic ray reactions -----
    // (0) cr + H2 -> H2+ + *e
    rates.destruction[IH2] += kcr_[0];
    rates.creation[IH2_plus] += kcr_[0] * y_in[IH2];
    // (1) cr + *He -> He+ + *e
    rates.creation[IHE_plus] += kcr_[1] * ghosts.He;
    // (2) cr + *H -> H+ + *e
    rates.creation[IH_plus] += kcr_[2] * ghosts.H;
    // (3) cr + *C -> C+ + *e
    rates.creation[IC_plus] += kcr_[3] * ghosts.C;
    // (4) crphoto + CO -> *O + *C
    rates.destruction[ICO] += kcr_[4];
    // (5) cr + CO -> HCO+ + *e
    rates.destruction[ICO] += kcr_[5];
    rates.creation[IHCO_plus] += kcr_[5] * y_in[ICO];
    // (6) cr + *Si -> Si+ + *e
    rates.creation[ISi_plus] += kcr_[6] * ghosts.Si;

    // ----- 2 body reactions -----
    // (0) H3+ + *C -> CH + H2
    {
      const Real k = k2body_[i2body_H3p_C];
      const Real rate = k * y_in[IH3_plus] * ghosts.C;
      rates.destruction[IH3_plus] += k * ghosts.C;
      rates.creation[ICHx] += rate;
      rates.creation[IH2] += rate;
    }
    // (1) H3+ + *O -> OH + H2
    {
      const Real k = k2body_[i2body_H3p_O];
      const Real rate = k * y_in[IH3_plus] * ghosts.O;
      rates.destruction[IH3_plus] += k * ghosts.O;
      rates.creation[IOHx] += rate;
      rates.creation[IH2] += rate;
    }
    // (2) H3+ + CO -> HCO+ + H2
    {
      const Real k = k2body_[i2body_H3p_CO];
      const Real rate = k * y_in[IH3_plus] * y_in[ICO];
      rates.destruction[IH3_plus] += k * y_in[ICO];
      rates.destruction[ICO] += k * y_in[IH3_plus];
      rates.creation[IHCO_plus] += rate;
      rates.creation[IH2] += rate;
    }
    // (3) He+ + H2 -> H+ + *He + *H
    {
      const Real k = k2body_[i2body_Hep_H2];
      rates.destruction[IHE_plus] += k * y_in[IH2];
      rates.destruction[IH2] += k * y_in[IHE_plus];
      rates.creation[IH_plus] += k * y_in[IHE_plus] * y_in[IH2];
    }
    // (4) He+ + CO -> C+ + *O + *He
    {
      const Real k = k2body_[i2body_Hep_CO];
      rates.destruction[IHE_plus] += k * y_in[ICO];
      rates.destruction[ICO] += k * y_in[IHE_plus];
      rates.creation[IC_plus] += k * y_in[IHE_plus] * y_in[ICO];
    }
    // (5) C+ + H2 -> CH + *H
    {
      const Real k = k2body_[i2body_Cp_H2];
      rates.destruction[IC_plus] += k * y_in[IH2];
      rates.destruction[IH2] += k * y_in[IC_plus];
      rates.creation[ICHx] += k * y_in[IC_plus] * y_in[IH2];
    }
    // (6) C+ + OH -> HCO+
    {
      const Real k = k2body_[i2body_Cp_OH];
      rates.destruction[IC_plus] += k * y_in[IOHx];
      rates.destruction[IOHx] += k * y_in[IC_plus];
      rates.creation[IHCO_plus] += k * y_in[IC_plus] * y_in[IOHx];
    }
    // (7) CH + *O -> CO + *H
    {
      const Real k = k2body_[i2body_CH_O];
      rates.destruction[ICHx] += k * ghosts.O;
      rates.creation[ICO] += k * y_in[ICHx] * ghosts.O;
    }
    // (8) OH + *C -> CO + *H
    {
      const Real k = k2body_[i2body_OH_C];
      rates.destruction[IOHx] += k * ghosts.C;
      rates.creation[ICO] += k * y_in[IOHx] * ghosts.C;
    }
    // (9) He+ + *e -> *He
    rates.destruction[IHE_plus] += k2body_[i2body_Hep_e] * ghosts.e;
    // (10) H3+ + *e -> H2 + *H
    {
      const Real k = k2body_[i2body_H3p_e];
      rates.destruction[IH3_plus] += k * ghosts.e;
      rates.creation[IH2] += k * y_in[IH3_plus] * ghosts.e;
    }
    // (11) C+ + *e -> *C
    rates.destruction[IC_plus] += k2body_[i2body_Cp_e] * ghosts.e;
    // (12) HCO+ + *e -> CO + *H
    {
      const Real k = k2body_[i2body_HCOp_e];
      rates.destruction[IHCO_plus] += k * ghosts.e;
      rates.creation[ICO] += k * y_in[IHCO_plus] * ghosts.e;
    }
    // (13) H2+ + H2 -> H3+ + *H
    {
      const Real k = k2body_[i2body_H2p_H2];
      rates.destruction[IH2_plus] += k * y_in[IH2];
      rates.destruction[IH2] += k * y_in[IH2_plus];
      rates.creation[IH3_plus] += k * y_in[IH2_plus] * y_in[IH2];
    }
    // (14) H+ + *e -> *H
    rates.destruction[IH_plus] += k2body_[i2body_Hp_e] * ghosts.e;
    // (15) H2 + *H -> 3 *H
    rates.destruction[IH2] += k2body_[i2body_H2_H] * ghosts.H;
    // (16) H2 + H2 -> H2 + 2 *H. Both reactants are H2 so it's destroyed at
    // twice the rate, but one H2 survives as a product
    {
      const Real k = k2body_[i2body_H2_H2];
      rates.destruction[IH2] += 2.0 * k * y_in[IH2];
      rates.creation[IH2] += k * y_in[IH2] * y_in[IH2];
    }
    // (17) *H + *e -> H+ + 2 *e
    rates.creation[IH_plus] += k2body_[i2body_H_e] * ghosts.H * ghosts.e;
    // (18) H3+ + *e -> *3H
    rates.destruction[IH3_plus] += k2body_[i2body_H3p_e_3H] * ghosts.e;
    // (19) He+ + H2 -> H2+ + *He
    {
      const Real k = k2body_[i2body_Hep_H2_H2p];
      rates.destruction[IHE_plus] += k * y_in[IH2];
      rates.destruction[IH2] += k * y_in[IHE_plus];
      rates.creation[IH2_plus] += k * y_in[IHE_plus] * y_in[IH2];
    }
    // (20) CH + *H -> H2 + *C
    {
      const Real k = k2body_[i2body_CH_H];
      rates.destruction[ICHx] += k * ghosts.H;
      rates.creation[IH2] += k * y_in[ICHx] * ghosts.H;
    }
    // (21) OH + *O -> *O + *O + *H
    rates.destruction[IOHx] += k2body_[i2body_OH_O] * ghosts.O;
    // (22) C+ + H2 + *e -> *C + *H + *H
    {
      const Real k = k2body_[i2body_Cp_H2_e];
      rates.destruction[IC_plus] += k * y_in[IH2];
      rates.destruction[IH2] += k * y_in[IC_plus];
    }
    // (23) Si+ + *e -> *Si
    rates.destruction[ISi_plus] += k2body_[i2body_Sip_e] * ghosts.e;
    // (24) H3+ + *O + *e -> H2 + *O + *H
    {
      const Real k = k2body_[i2body_H3p_O_H2];
      rates.destruction[IH3_plus] += k * ghosts.O;
      rates.creation[IH2] += k * y_in[IH3_plus] * ghosts.O;
    }
    // (25) He+ + OH -> O+ + *He + *H
    {
      const Real k = k2body_[i2body_Hep_OH];
      rates.destruction[IHE_plus] += k * y_in[IOHx];
      rates.destruction[IOHx] += k * y_in[IHE_plus];
      rates.creation[IO_plus] += k * y_in[IHE_plus] * y_in[IOHx];
    }
    // (26) H2+ + *H -> H+ + H2
    {
      const Real k = k2body_[i2body_H2p_H];
      const Real rate = k * y_in[IH2_plus] * ghosts.H;
      rates.destruction[IH2_plus] += k * ghosts.H;
      rates.creation[IH_plus] += rate;
      rates.creation[IH2] += rate;
    }
    // (27) H+ + *O -> O+ + *H
    {
      const Real k = k2body_[i2body_Hp_O];
      rates.destruction[IH_plus] += k * ghosts.O;
      rates.creation[IO_plus] += k * y_in[IH_plus] * ghosts.O;
    }
    // (28) O+ + *H -> H+ + *O
    {
      const Real k = k2body_[i2body_Op_H];
      rates.destruction[IO_plus] += k * ghosts.H;
      rates.creation[IH_plus] += k * y_in[IO_plus] * ghosts.H;
    }
    // (29) O+ + H2 -> OH + *H
    {
      const Real k = k2body_[i2body_Op_H2_OH];
      rates.destruction[IO_plus] += k * y_in[IH2];
      rates.destruction[IH2] += k * y_in[IO_plus];
      rates.creation[IOHx] += k * y_in[IO_plus] * y_in[IH2];
    }
    // (30) O+ + H2 -> *O + *H + *H
    {
      const Real k = k2body_[i2body_Op_H2];
      rates.destruction[IO_plus] += k * y_in[IH2];
      rates.destruction[IH2] += k * y_in[IO_plus];
    }

    // ----- photo reactions -----
    // (0) h nu + *C -> C+ + *e
    rates.creation[IC_plus] += kph_[iph_C] * ghosts.C;
    // (1) h nu + CH -> *C + *H
    rates.destruction[ICHx] += kph_[iph_CHx];
    // (2) h nu + CO -> *C + *O
    rates.destruction[ICO] += kph_[iph_CO];
    // (3) h nu + OH -> *O + *H
    rates.destruction[IOHx] += kph_[iph_OHx];
    // (4) h nu + H2 -> *H + *H
    rates.destruction[IH2] += kph_[iph_H2];
    // (5) h nu + *Si -> Si+
    rates.creation[ISi_plus] += kph_[iph_Si] * ghosts.Si;

    // ----- grain assisted reactions -----
    // (0) *H + *H + gr -> H2 + gr
    rates.creation[IH2] += kgr_[igr_H] * ghosts.H;
    // (1) H+ + *e + gr -> *H + gr
    rates.destruction[IH_plus] += kgr_[igr_Hp];
    // (2) C+ + *e + gr -> *C + gr
    rates.destruction[IC_plus] += kgr_[igr_Cp];
    // (3) He+ + *e + gr -> *He + gr
    rates.destruction[IHE_plus] += kgr_[igr_Hep];
    // (4) Si+ + *e + gr -> *Si + gr
    rates.destruction[ISi_plus] += kgr_[igr_Sip];

    // convert to code units
    for (size_t i = 0; i < rates.creation.size(); i++) {
      rates.creation(i) *= units_time_cgs;
      rates.destruction(i) *= units_time_cgs;
    }

    return rates;
  }

  /*!
   * \brief Setup the network for the next iteration of the ODE solver
   *
   * \details Floors the state at zero and checks it for NaNs/Infs, then
   * computes the ghost species and updates the reaction rates.
   *
   * \param y_in The current state
   * \return GhostSpecies The ghost species abundances
   */
  // Called by every evaluate_function, and so by all 14 Jacobian evaluations.
  // Recomputes the ghost species and the entire rate table unconditionally; the
  // hoist described on UpdateRates_ adds a species-only mode here.
  template <class vec_type>
  KOKKOS_FUNCTION GhostSpecies SetupNextStep(
      const vec_type& y_in, const bool species_only = false) const {
    // Verify abundances are positive, finite, and not NaN valued
    for (size_t i = 0; i < neqs; i++) {
      // Verify positivity
      y_in(i) = Kokkos::fmax(y_in(i), 0.0);

      // Check if inf or NaN valued and throw abort if that's the case
      if (Kokkos::isinf(y_in(i)) || Kokkos::isnan(y_in(i))) {
        Kokkos::abort("Error: NaN or Inf value found in GOW17 `y` array\n");
      }
    }

    // Set the ghost species
    const GhostSpecies ghosts = ComputeGhostSpecies_(y_in);

    // Compute rates
    UpdateRates_(y_in, ghosts, species_only);

    return ghosts;
  }

  /*!
   * \brief phi(x) = (1 - exp(-x)) / x, the entire function with phi(0) = 1.
   *
   * \details Evaluated through expm1 so it stays accurate as x -> 0, where the
   * quotient is 0/0, and switched to its series below the point where the
   * subtraction loses significance.
   */
  static KOKKOS_INLINE_FUNCTION Real PhiOne_(const Real x) {
    if (Kokkos::abs(x) < 1.0e-8) {
      return 1.0 - 0.5 * x;
    }
    return -Kokkos::expm1(-x) / x;
  }

  /*!
   * \brief Advance one species across a substep with C and D held frozen.
   *
   * \details Two forms of the same sub-problem, dy/dt = C - D y.
   *
   * The backward-Euler form (y^n + C h) / (1 + D h) is first order: right in
   * both limits -- forward Euler as D h -> 0, the equilibrium C/D as
   * D h -> infinity -- but only first-order accurate in between, which is what
   * sets the error of the sweep as a whole.
   *
   * The exact form is the analytic solution of that frozen-coefficient problem,
   *
   *   y = C/D + (y^n - C/D) exp(-D h),
   *
   * following the exact-map construction of Inoue & Inutsuka 2008 (ApJ 687,
   * 303), section 3.2.1. Writing it as y^n exp(-x) + C h phi(x) with
   * phi(x) = (1 - exp(-x))/x avoids dividing by D when a species has no
   * destruction channel, and evaluating phi through expm1 keeps it accurate as
   * x -> 0, where it tends to 1 and the whole expression collapses to forward
   * Euler. It removes the single-species truncation error entirely, leaving
   * only the error from freezing C and D across the substep.
   *
   * \param y_n Species abundance at the start of the substep
   * \param creation Creation rate C, independent of this species
   * \param destruction Destruction coefficient D, so that the loss is D*y
   * \param h Substep size
   * \param exact_map Use the exponential solution rather than backward Euler
   */
  static KOKKOS_INLINE_FUNCTION Real BEStep_(const Real y_n, const Real creation,
                                             const Real destruction,
                                             const Real h,
                                             const bool exact_map) {
    const Real x = destruction * h;
    if (!exact_map) {
      return (y_n + creation * h) / (1.0 + x);
    }
    // phi(x) = (1 - exp(-x))/x, expanded near zero where the quotient is 0/0
    return y_n * Kokkos::exp(-x) + creation * h * PhiOne_(x);
  }

  /*!
   * \brief He+ backward-Euler step.
   *
   * \details Factored out because its position in the sweep is the one genuinely
   * open ordering choice. Its creation is cosmic-ray ionization of neutral He
   * and depends on no other integrated species, and it feeds the creation terms
   * of H+, C+, H2+ and O+ -- so a source-first argument puts it at the head of
   * the sweep. But its destruction reads CO and OHx, which are updated last, and
   * at high density the He+ + CO channel is not negligible against electron
   * recombination. `sweep_hep_first` selects between the two placements.
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real HePlusStep_(const vec_type& y, const Real* y_n,
                                   const GhostSpecies& g, const Real h,
                                   const bool exact_map) const {
    const Real c = kcr_[1] * g.He;
    const Real d = k2body_[i2body_Hep_H2] * y[IH2] +
                   k2body_[i2body_Hep_CO] * y[ICO] +
                   k2body_[i2body_Hep_e] * g.e +
                   k2body_[i2body_Hep_H2_H2p] * y[IH2] +
                   k2body_[i2body_Hep_OH] * y[IOHx] + kgr_[igr_Hep];
    return BEStep_(y_n[IHE_plus], units_time_cgs * c, units_time_cgs * d, h,
                   exact_map);
  }

  /*!
   * \brief H2 backward-Euler / exact step.
   *
   * \details Its position in the sweep is the second open ordering choice, and
   * a consequential one: H2 sets the destruction rate of most ions and appears
   * in the creation term of H2+, H3+, CHx and H+, so three of the network's six
   * mutually-creating pairs involve it -- more than any other species.
   *
   * Placing it last treats it as a slowly varying background, which is right
   * once the gas is molecular. It is wrong during H2 formation: in the uniform
   * test problem x(H2) climbs from 1e-6 to 3.6e-2 inside 0.6 Myr, four and a
   * half orders of magnitude, making it among the fastest-changing quantities
   * rather than the slowest. `sweep_h2_first` selects between the two.
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real StepH2_(const vec_type& y, const Real* y_n,
                               const GhostSpecies& g, const Real h,
                               const bool exact_map) const {
    const Real u = units_time_cgs;
      const Real c = k2body_[i2body_H3p_C] * y[IH3_plus] * g.C +
                     k2body_[i2body_H3p_O] * y[IH3_plus] * g.O +
                     k2body_[i2body_H3p_CO] * y[IH3_plus] * y[ICO] +
                     k2body_[i2body_H3p_e] * y[IH3_plus] * g.e +
                     k2body_[i2body_H2_H2] * y_n[IH2] * y_n[IH2] +
                     k2body_[i2body_CH_H] * y[ICHx] * g.H +
                     k2body_[i2body_H3p_O_H2] * y[IH3_plus] * g.O +
                     k2body_[i2body_H2p_H] * y[IH2_plus] * g.H +
                     kgr_[igr_H] * g.H;
      const Real d = kcr_[0] + k2body_[i2body_Hep_H2] * y[IHE_plus] +
                     k2body_[i2body_Cp_H2] * y[IC_plus] +
                     k2body_[i2body_H2p_H2] * y[IH2_plus] +
                     k2body_[i2body_H2_H] * g.H +
                     2.0 * k2body_[i2body_H2_H2] * y_n[IH2] +
                     k2body_[i2body_Hep_H2_H2p] * y[IHE_plus] +
                     k2body_[i2body_Cp_H2_e] * y[IC_plus] +
                     k2body_[i2body_Op_H2_OH] * y[IO_plus] +
                     k2body_[i2body_Op_H2] * y[IO_plus] + kph_[iph_H2];
      y[IH2] = BEStep_(y_n[IH2], u * c, u * d, h, exact_map);
    return y[IH2];
  }

  /*!
   * \brief One substep of the sweep with a hand-chosen species order.
   *
   * \details Replaces the index-order Jacobi loop with a Gauss-Seidel sweep
   * ordered by the reaction graph, so that each species reads the updated value
   * of the species it is created from. The order is
   *
   *   He+, Si+, H2+, H3+, H+, O+, C+, CHx, OHx, {CO, HCO+}, H2
   *
   * derived as follows. He+ is the only species whose creation term involves no
   * other integrated species (cosmic rays on neutral He) while feeding four of
   * them, so it heads the sweep. Si+ is isolated -- creation from cosmic rays
   * and photons on neutral Si, destruction by electrons and grains -- and could
   * sit anywhere. H2+ then H3+ is the one clean chain in the network
   * (cr + H2 -> H2+, H2+ + H2 -> H3+). H+ and O+ follow because their creation
   * reads He+ and H2+. C+ reads He+; CHx and OHx read H3+, C+, He+ and O+. H2 is
   * the hub every species couples to but it is slow and abundant, so it is the
   * lagged background and is updated last, when every other value is fresh.
   *
   * CO and HCO+ sit in each other's creation term (HCO+ + e -> CO and both
   * cr + CO -> HCO+ and H3+ + CO -> HCO+), a two-cycle that no ordering can
   * break. They are solved simultaneously in closed form instead. Writing
   * A = 1 + D_CO h and B = 1 + D_HCO h, the pair reduces to a 2x2 whose
   * determinant A B - b c h^2 is bounded below by 1 + (D_CO + D_HCO) h, because
   * b <= D_HCO and c <= D_CO by construction -- so the solve never divides by
   * zero and, every numerator being a sum of non-negative terms, it cannot
   * produce a negative abundance.
   *
   * The couplings that remain lagged after this ordering are He+ <- CO, OHx;
   * H3+ <- CO; H+ <- O+; C+ <- CO, OHx; and O+ <- OHx. Each is a destruction-side
   * or minor-channel dependence rather than a dominant creation term.
   *
   * Ghost species are evaluated once per substep by the caller and held fixed.
   *
   * \param y     Current state, read and written in place. A species not yet
   *              reached in the sweep still holds its start-of-substep value,
   *              which is what makes this Gauss-Seidel rather than Jacobi.
   * \param y_n   Start-of-substep state, the y^n of the backward-Euler formula
   * \param g     Ghost species, held fixed across the substep
   * \param h     Substep size
   * \param hep_first  Place He+ at the head of the sweep rather than after CO
   * \param h2_first   Update H2 at the head of the sweep rather than the tail
   * \param exact_map  Use the exponential map rather than backward Euler for
   *              the scalar species steps. The CO/HCO+ pair keeps its backward-
   *              Euler 2x2, whose exact analogue is a 2x2 matrix exponential.
   */
  template <class vec_type>
  KOKKOS_FUNCTION void OrderedSweepUpdate(const vec_type& y, const Real* y_n,
                                          const GhostSpecies& g, const Real h,
                                          const bool hep_first,
                                          const bool exact_map,
                                          const bool exact_block,
                                          const bool h2_first) const {
    const Real u = units_time_cgs;

    // ----- H2, early placement: every ion then reads a fresh H2 -----
    if (h2_first) {
      y[IH2] = StepH2_(y, y_n, g, h, exact_map);
    }

    // ----- He+ : source-driven, feeds H+, C+, H2+, O+ -----
    if (hep_first) {
      y[IHE_plus] = HePlusStep_(y, y_n, g, h, exact_map);
    }

    // ----- Si+ : isolated, couples only through the electron abundance -----
    {
      const Real c = kcr_[6] * g.Si + kph_[iph_Si] * g.Si;
      const Real d = k2body_[i2body_Sip_e] * g.e + kgr_[igr_Sip];
      y[ISi_plus] = BEStep_(y_n[ISi_plus], u * c, u * d, h, exact_map);
    }

    // ----- H2+ : cr + H2, and He+ + H2 -----
    {
      const Real c = kcr_[0] * y[IH2] +
                     k2body_[i2body_Hep_H2_H2p] * y[IHE_plus] * y[IH2];
      const Real d = k2body_[i2body_H2p_H2] * y[IH2] +
                     k2body_[i2body_H2p_H] * g.H;
      y[IH2_plus] = BEStep_(y_n[IH2_plus], u * c, u * d, h, exact_map);
    }

    // ----- H3+ : the downstream half of the H2+ chain -----
    {
      const Real c = k2body_[i2body_H2p_H2] * y[IH2_plus] * y[IH2];
      const Real d = k2body_[i2body_H3p_C] * g.C +
                     k2body_[i2body_H3p_O] * g.O +
                     k2body_[i2body_H3p_CO] * y[ICO] +
                     k2body_[i2body_H3p_e] * g.e +
                     k2body_[i2body_H3p_e_3H] * g.e +
                     k2body_[i2body_H3p_O_H2] * g.O;
      y[IH3_plus] = BEStep_(y_n[IH3_plus], u * c, u * d, h, exact_map);
    }

    // ----- H+ : reads fresh He+ and H2+, lags O+ -----
    {
      const Real c = kcr_[2] * g.H +
                     k2body_[i2body_Hep_H2] * y[IHE_plus] * y[IH2] +
                     k2body_[i2body_H_e] * g.H * g.e +
                     k2body_[i2body_H2p_H] * y[IH2_plus] * g.H +
                     k2body_[i2body_Op_H] * y[IO_plus] * g.H;
      const Real d = k2body_[i2body_Hp_e] * g.e +
                     k2body_[i2body_Hp_O] * g.O + kgr_[igr_Hp];
      y[IH_plus] = BEStep_(y_n[IH_plus], u * c, u * d, h, exact_map);
    }

    // ----- O+ : reads fresh He+ and H+, lags OHx -----
    {
      const Real c = k2body_[i2body_Hep_OH] * y[IHE_plus] * y[IOHx] +
                     k2body_[i2body_Hp_O] * y[IH_plus] * g.O;
      const Real d = k2body_[i2body_Op_H] * g.H +
                     k2body_[i2body_Op_H2_OH] * y[IH2] +
                     k2body_[i2body_Op_H2] * y[IH2];
      y[IO_plus] = BEStep_(y_n[IO_plus], u * c, u * d, h, exact_map);
    }

    // ----- C+ : reads fresh He+, lags CO and OHx -----
    {
      const Real c = kcr_[3] * g.C + kph_[iph_C] * g.C +
                     k2body_[i2body_Hep_CO] * y[IHE_plus] * y[ICO];
      const Real d = k2body_[i2body_Cp_H2] * y[IH2] +
                     k2body_[i2body_Cp_OH] * y[IOHx] +
                     k2body_[i2body_Cp_e] * g.e +
                     k2body_[i2body_Cp_H2_e] * y[IH2] + kgr_[igr_Cp];
      y[IC_plus] = BEStep_(y_n[IC_plus], u * c, u * d, h, exact_map);
    }

    // ----- CHx : reads fresh H3+ and C+ -----
    {
      const Real c = k2body_[i2body_H3p_C] * y[IH3_plus] * g.C +
                     k2body_[i2body_Cp_H2] * y[IC_plus] * y[IH2];
      const Real d = k2body_[i2body_CH_O] * g.O +
                     k2body_[i2body_CH_H] * g.H + kph_[iph_CHx];
      y[ICHx] = BEStep_(y_n[ICHx], u * c, u * d, h, exact_map);
    }

    // ----- OHx : reads fresh H3+, O+, C+ and He+ -----
    {
      const Real c = k2body_[i2body_H3p_O] * y[IH3_plus] * g.O +
                     k2body_[i2body_Op_H2_OH] * y[IO_plus] * y[IH2];
      const Real d = k2body_[i2body_Cp_OH] * y[IC_plus] +
                     k2body_[i2body_OH_C] * g.C +
                     k2body_[i2body_OH_O] * g.O +
                     k2body_[i2body_Hep_OH] * y[IHE_plus] + kph_[iph_OHx];
      y[IOHx] = BEStep_(y_n[IOHx], u * c, u * d, h, exact_map);
    }

    // ----- {CO, HCO+} : mutually creating, solved simultaneously -----
    {
      const Real a = u * (k2body_[i2body_CH_O] * y[ICHx] * g.O +
                          k2body_[i2body_OH_C] * y[IOHx] * g.C);
      const Real b = u * k2body_[i2body_HCOp_e] * g.e;
      const Real d1 = u * (kcr_[4] + kcr_[5] +
                           k2body_[i2body_H3p_CO] * y[IH3_plus] +
                           k2body_[i2body_Hep_CO] * y[IHE_plus] + kph_[iph_CO]);
      const Real f = u * k2body_[i2body_Cp_OH] * y[IC_plus] * y[IOHx];
      const Real c = u * (kcr_[5] + k2body_[i2body_H3p_CO] * y[IH3_plus]);
      const Real d2 = u * k2body_[i2body_HCOp_e] * g.e;

      if (!exact_block) {
        // Backward-Euler 2x2. det = A B - b c h^2 >= 1 + (d1 + d2) h, because
        // b <= d2 and c <= d1 by construction, so it never vanishes and every
        // numerator is a sum of non-negative terms.
        const Real A = 1.0 + d1 * h;
        const Real B = 1.0 + d2 * h;
        const Real det = A * B - b * c * h * h;
        const Real co =
            (B * (y_n[ICO] + a * h) + b * h * (y_n[IHCO_plus] + f * h)) / det;
        y[ICO] = co;
        y[IHCO_plus] = (y_n[IHCO_plus] + f * h + c * h * co) / B;
      } else {
        // Exact map for the pair: dy/dt = s - M y with
        //   M = [[d1, -b], [-c, d2]],  s = (a, f),
        // solved as y(h) = exp(-M h) y_n + h phi1(-M h) s. For a 2x2 any matrix
        // function is alpha I + beta M, with alpha and beta the divided
        // differences of the scalar function over the two eigenvalues. The
        // discriminant below cannot go negative because b and c are both
        // non-negative rates, so the eigenvalues are always real and no complex
        // arithmetic is needed.
        //
        // Note b == d2 identically here: HCO+ + e -> CO is HCO+'s only sink, so
        // every HCO+ destroyed becomes a CO. det M = d2 (d1 - c) is therefore
        // the rate at which carbon leaks out of the pair, and it reaches zero
        // when the pair is closed. The equilibrium M^-1 s diverges there, which
        // is exactly why the phi1 form is used instead.
        const Real half_sum = 0.5 * (d1 + d2);
        const Real half_diff = 0.5 * (d1 - d2);
        const Real disc = Kokkos::sqrt(half_diff * half_diff + b * c);
        const Real lam_p = half_sum + disc;
        const Real lam_m = half_sum - disc;

        const Real e_p = Kokkos::exp(-lam_p * h);
        const Real e_m = Kokkos::exp(-lam_m * h);
        const Real phi_p = PhiOne_(lam_p * h);
        const Real phi_m = PhiOne_(lam_m * h);

        // Divided differences, replaced by the derivative when the eigenvalues
        // collide and the quotient becomes 0/0.
        const Real gap = lam_p - lam_m;  // = 2 disc, non-negative
        Real beta, beta_phi;
        if (gap > 1.0e-12 * (1.0 + half_sum)) {
          beta = (e_p - e_m) / gap;
          beta_phi = (phi_p - phi_m) / gap;
        } else {
          // d/dlam exp(-lam h) = -h exp(-lam h); d/dlam phi(lam h) from the
          // series, phi(x) = 1 - x/2 + x^2/6 - ...
          beta = -h * e_m;
          beta_phi = h * (-0.5 + lam_m * h / 3.0);
        }
        const Real alpha = e_m - beta * lam_m;
        const Real alpha_phi = phi_m - beta_phi * lam_m;

        // (alpha I + beta M) y_n
        const Real co_n = y_n[ICO];
        const Real hco_n = y_n[IHCO_plus];
        const Real hom_co = alpha * co_n + beta * (d1 * co_n - b * hco_n);
        const Real hom_hco = alpha * hco_n + beta * (-c * co_n + d2 * hco_n);
        // h (alpha_phi I + beta_phi M) s
        const Real src_co = h * (alpha_phi * a + beta_phi * (d1 * a - b * f));
        const Real src_hco = h * (alpha_phi * f + beta_phi * (-c * a + d2 * f));

        y[ICO] = Kokkos::fmax(hom_co + src_co, 0.0);
        y[IHCO_plus] = Kokkos::fmax(hom_hco + src_hco, 0.0);
      }
    }

    // ----- He+, late placement: reads fresh CO and OHx instead -----
    if (!hep_first) {
      y[IHE_plus] = HePlusStep_(y, y_n, g, h, exact_map);
    }

    // ----- H2 -----
    if (!h2_first) {
      y[IH2] = StepH2_(y, y_n, g, h, exact_map);
    }
  }

  /*!
   * \brief Rescale each element back onto its conservation law.
   *
   * \details The semi-implicit sweep updates every species independently, so
   * nothing enforces the element budgets that ComputeGhostSpecies_ reads back
   * out. Left alone an overshoot is absorbed silently by the fmax(..., 0.0)
   * clamps there, and the element quietly stops being conserved. Any group
   * that exceeds its total is scaled uniformly back onto it.
   *
   * The groups share members -- HCO+ carries C, O and H at once -- so the
   * passes are not independent. They run in order of increasing overlap, with
   * hydrogen last, which leaves hydrogen exactly on its budget and the others
   * at or below theirs.
   *
   * Not needed by the implicit solvers, whose Newton iterate keeps the
   * conservation laws satisfied to the tolerance of the solve.
   *
   * \param y_in The state to renormalize, modified in place
   */
  template <class vec_type>
  KOKKOS_FUNCTION void RenormalizeElements(const vec_type& y_in) const {
    // Non-negativity first: a negative member would mask an overshoot.
    for (size_t i = 0; i < neqs - 1; i++) {
      y_in(i) = Kokkos::fmax(y_in(i), 0.0);
    }

    // Single-species budgets
    y_in[ISi_plus] = Kokkos::fmin(y_in[ISi_plus], xSi);
    y_in[IHE_plus] = Kokkos::fmin(y_in[IHE_plus], xHe);

    // Carbon: HCO+ + CHx + CO + C+ <= xC
    {
      const Real total =
          y_in[IHCO_plus] + y_in[ICHx] + y_in[ICO] + y_in[IC_plus];
      if (total > xC) {
        const Real scale = xC / total;
        y_in[IHCO_plus] *= scale;
        y_in[ICHx] *= scale;
        y_in[ICO] *= scale;
        y_in[IC_plus] *= scale;
      }
    }

    // Oxygen: HCO+ + OHx + CO + O+ <= xO
    {
      const Real total =
          y_in[IHCO_plus] + y_in[IOHx] + y_in[ICO] + y_in[IO_plus];
      if (total > xO) {
        const Real scale = xO / total;
        y_in[IHCO_plus] *= scale;
        y_in[IOHx] *= scale;
        y_in[ICO] *= scale;
        y_in[IO_plus] *= scale;
      }
    }

    // Hydrogen: OHx + CHx + HCO+ + 3 H3+ + 2 H2+ + H+ + 2 H2 <= 1
    {
      const Real total = y_in[IOHx] + y_in[ICHx] + y_in[IHCO_plus] +
                         3.0 * y_in[IH3_plus] + 2.0 * y_in[IH2_plus] +
                         y_in[IH_plus] + 2.0 * y_in[IH2];
      if (total > 1.0) {
        const Real scale = 1.0 / total;
        y_in[IOHx] *= scale;
        y_in[ICHx] *= scale;
        y_in[IHCO_plus] *= scale;
        y_in[IH3_plus] *= scale;
        y_in[IH2_plus] *= scale;
        y_in[IH_plus] *= scale;
        y_in[IH2] *= scale;
      }
    }
  }

  /*!
   * \brief Computes `f` using the values in `y_in`
   */
  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION void evaluate_function(const Real /*t*/, const Real /*dt*/,
                                         const vec_type1& y_in, vec_type2& f,
                                         const bool species_only = false) const {
    // ----- Setup for the next step -----
    const auto ghosts = SetupNextStep(y_in, species_only);

    // ----- Internal energy equation -----
    f(IIE) = Edot(y_in, ghosts);

    // ----- Creation & Destruction Rates -----
    const auto rates = CDRates(y_in, ghosts);

    // Compute the changes
    for (size_t i = 0; i < neqs - 1; i++) {
      f(i) = rates.creation(i) - y_in(i) * rates.destruction(i);
    }

    // Verify abundances are finite and not NaN valued
    for (size_t i = 0; i < f.size(); i++) {
      // Check if inf or NaN valued and throw abort if that's the case
      if (Kokkos::isinf(f(i)) || Kokkos::isnan(f(i))) {
        Kokkos::abort("Error: NaN or Inf value found in GOW17 `f` array\n");
      }
    }
  }

  template <class vec_type, class mat_type>
  KOKKOS_FUNCTION void evaluate_jacobian(const Real t, const Real dt,
                                         const vec_type& y_in,
                                         const mat_type& jac) const {
    if (!jacobian_hoist) {
      chemistry::numerical_jacobian(*this, t, dt, y_in, jac);
      return;
    }

    // Hoisted Jacobian. The generic version calls evaluate_function 1 + neqs
    // times, and each call re-evaluates the ~45 temperature-only transcendentals
    // in UpdateRates_. Here the rate table is built once, the twelve species
    // columns reuse it, and the thermal coupling they would otherwise lose is
    // added back analytically. Two full rate evaluations instead of fourteen.
    // Derivation: ~/ai-notes/docs-claude/athenak-chemistry/
    //             gow17-jacobian-hoist-design.md
    RegisterArray<Real, neqs> f0, fp;

    const Real perturbation_factor =
        Kokkos::sqrt(Kokkos::ArithTraits<Real>::epsilon());

    // Base state, full evaluation. This is what populates the temperature cache
    // that every species column below then reuses.
    evaluate_function(t, dt, y_in, f0);
    // Capture the base-state cache now: the internal-energy column below runs
    // a full evaluation and overwrites cached_T_ / cached_N_ with the perturbed
    // state's values.
    const Real N_base = cached_N_;
    const Real dTdx_scale = cached_dTdx_scale_;

    // ----- species columns, temperature frozen -----
    for (int j = 0; j < neqs - 1; ++j) {
      const Real perturbation =
          perturbation_factor * Kokkos::fmax(Kokkos::abs(y_in(j)), Real(1.0));
      const Real y_unperturbed = y_in(j);
      y_in(j) += perturbation;

      evaluate_function(t, dt, y_in, fp, /*species_only=*/true);

      const Real inverse_diff = Real(1.0) / perturbation;
      for (int k = 0; k < neqs; ++k) {
        jac(k, j) = (fp(k) - f0(k)) * inverse_diff;
      }
      y_in(j) = y_unperturbed;
    }

    // ----- internal energy column, full evaluation -----
    // Done last so it does not disturb the cache the species columns needed.
    {
      const int j = IIE;
      const Real perturbation =
          perturbation_factor * Kokkos::fmax(Kokkos::abs(y_in(j)), Real(1.0));
      const Real y_unperturbed = y_in(j);
      y_in(j) += perturbation;

      evaluate_function(t, dt, y_in, fp);

      const Real inverse_diff = Real(1.0) / perturbation;
      for (int k = 0; k < neqs; ++k) {
        jac(k, j) = (fp(k) - f0(k)) * inverse_diff;
      }
      y_in(j) = y_unperturbed;
    }

    // ----- restore the thermal coupling the frozen columns dropped -----
    // T = E_ergs / Cv with E_ergs = y(IIE) * units / n_H, so
    //   dT/dy(IIE) = (units / n_H) / Cv,
    // and df/dT follows from the energy column already computed above. Taking
    // df/dT this way rather than analytically means any temperature clamping is
    // inherited automatically: if T is pinned, that column is flat and the
    // correction is correctly zero.
    if (dTdx_scale != Real(0.0)) {
      // Cv = k_B N / (gamma - 1) and E_ergs = y(IIE) * units / n_H, so
      // dT/dy(IIE) = (units / n_H) (gamma - 1) / (k_B N).
      const Real dTdE = (units_energy_density_cgs / n_H) *
                        (gamma - Real(1.0)) /
                        (units::Units::k_boltzmann_cgs *
                         Kokkos::fmax(N_base, Real(1.0e-300)));
      if (dTdE > Real(0.0)) {
        for (int j = 0; j < neqs - 1; ++j) {
          // dT/dx(H2) = +T/N; dT/dx = -T/N for each cation, since ghosts.e is
          // their plain sum; the neutrals OHx, CHx and CO leave T alone.
          Real dTdx = Real(0.0);
          if (j == IH2) {
            dTdx = dTdx_scale;
          } else if (j == IHE_plus || j == IC_plus || j == IHCO_plus ||
                     j == IH_plus || j == IH3_plus || j == IH2_plus ||
                     j == IO_plus || j == ISi_plus) {
            dTdx = -dTdx_scale;
          }
          if (dTdx == Real(0.0)) continue;
          const Real factor = dTdx / dTdE;
          for (int k = 0; k < neqs; ++k) {
            jac(k, j) += jac(k, IIE) * factor;
          }
        }
      }
    }
  }

 private:
  // ----- Raw array buffers -----
  // The current state
  Real y_buffer_[neqs];  // NOLINT(runtime/arrays)
  // The results of evaluating the ODEs
  Real y_new_buffer_[neqs];  // NOLINT(runtime/arrays)

  /// The temperature to use for an isothermal EOS
  const Real isothermal_temperature_;
  // ----- Photo Reactions -----
  // Reaction rates in Drain 1978 field units.
  // Reactions are, in order:
  // (0) h nu + *C -> C+ + *e
  // (1) h nu + CH -> *C + *H
  // (2) h nu + CO -> *C + *O            --self-shielding and shielding by H2
  // (3) h nu + OH -> *O + *H
  // ----added in GO2012--------
  // (4) h nu + H2 -> *H + *H            --self- and dust shielding
  // ----Si, from UMIST12
  // (5) h nu + *Si -> Si+

  /// Reaction Rate enum
  enum : size_t { iph_C, iph_CHx, iph_CO, iph_OHx, iph_H2, iph_Si };

  /// radiation field intensity
  // RegisterArray<Real, n_freq> rad_;
  DvceArray1D<Real> rad_;
  /// enum for indexing into the rad_ array
  enum : size_t { irad_GPE = n_ph, irad_CR };

  /// rates for photo-reactions in s^-1
  mutable RegisterArray<Real, n_ph> kph_;

  // ----- Grain Reactions -----
  // Grain assisted recombination of H, H2, C+ and H+
  // (0) *H + *H + gr -> H2 + gr
  // (1) H+ + *e + gr -> *H + gr
  // (2) C+ + *e + gr -> *C + gr
  // (3) He+ + *e + gr -> *He + gr
  // ------Si, from WD2001-----
  // (4) Si+ + *e + gr -> *Si + gr

  /// constants
  static constexpr int n_gr_ = 5;

  /// rates for grain assisted reactions in cm^3 s^-1 z_d^-1
  mutable RegisterArray<Real, n_gr_> kgr_;
  /// enum for indexing into kgr_
  enum : size_t { igr_H, igr_Hp, igr_Cp, igr_Hep, igr_Sip };

  // ----- Chemical Network -----
  // cosmic ray chemistry network
  // (0) cr + H2 -> H2+ + *e
  // (1) cr + *He -> He+ + *e
  // (2) cr + *H  -> H+ + *e
  // -----added as Clark + Glover 2015----
  // (3) cr + *C -> C+ + *e     --including direct and cr induce photo reactions
  // (4) crphoto + CO -> *O + *C
  // (5) cr + CO -> HCO+ + e  --schematic for cr + CO -> CO+ + e
  // -----Si, CR induced photo ionization, experimenting----
  // (6) cr + Si -> Si+ + e, UMIST12
  static constexpr int n_cr_ = 7;
  /// rates for cosmic-ray reactions  in s^-1
  mutable RegisterArray<Real, n_cr_> kcr_;

  // clang-format off
  // 2 body reactions
  // NOTE: photons from recombination are ignored
  // Reactions are, in order.
  //  -- are equations of special rate treatment in Glover, Federrath+ 2010:
  // (0) H3+ + *C -> CH + H2         --Vissapragada2016 new rates
  // (1) H3+ + *O -> OH + H2
  // (2) H3+ + CO -> HCO+ + H2
  // (3) He+ + H2 -> H+ + *He + *H    --fit to Schauer1989
  // (4) He+ + CO -> C+ + *O + *He
  // (5) C+ + H2 -> CH + *H         -- schematic reaction for C+ + H2 -> CH2+
  // (6) C+ + OH -> HCO+             -- Schematic equation for C+ + OH -> CO+ + H.
  // Use rates in KIDA website.
  // (7) CH + *O -> CO + *H
  // (8) OH + *C -> CO + *H          --exp(0.108/T)
  // (9) He+ + *e -> *He             --(17) Case B
  // (10) H3+ + *e -> H2 + *H
  // (11) C+ + *e -> *C              -- Include RR and DR, Badnell2003, 2006.
  // (12) HCO+ + *e -> CO + *H
  // ----added in GO2012--------
  // (13) H2+ + H2 -> H3+ + *H       --(54) exp(-T/46600)
  // (14) H+ + *e -> *H              --(12) Case B
  // ---collisional dissociation, only important at high temperature T>1e3---
  // (15) H2 + *H -> 3 *H            --(9) Density dependent. See Glover+MacLow2007
  // (16) H2 + H2 -> H2 + 2 *H       --(10) Density dependent. See Glover+MacLow2007
  // (17) *H + *e -> H+ + 2 *e       --(11) Relates to Te
  // ----added for H3+ destruction in addition to (10)----
  // (18) H3+ + *e -> *3H            --(111)
  // ----added He+ destruction in addition to (3), from UMIST12----
  // (19) He+ + H2 -> H2+ + *He
  // ----added CH reaction to match for abundances of CH---
  // (20) CH + *H -> H2 + *C
  // ----added to match the Meudon code ---
  // (21) OH + *O -> *O + *O + *H
  // ---branching of C+ + H2 ------
  // (22) C+ + H2 + *e -> *C + *H + *H
  // ---Si , rate from UMIST12---
  // (23) Si+ + *e -> *Si
  // --- H2O+ + e reaction ---
  // (24) H3+ + *O + *e -> H2 + *O + *H
  // --- OH destruction with He+
  // (25) He+ + OH -> O+ + *He + *H
  // --- H2+ charge exchange with H ---
  // (26) H2+ + *H -> H+ + H2
  //  --- O+ reactions ---
  // (27) H+ + *O -> O+ + *H -- exp(-232/T)
  // (28) O+ + *H -> H+ + *O
  // (29) O+ + H2 -> OH + *H     -- branching of H2O+
  // (30) O+ + H2 -> *O + *H + *H  -- branching of H2O+
  // clang-format on

  static constexpr int n_2body_ = 31;
  /// rates for 2 body reactions in s^-1 cm^3
  mutable RegisterArray<Real, n_2body_> k2body_;

  // ----- Cached temperature-only rate factors -----
  // Filled by the temperature phase of UpdateRates_ and reused by the species
  // phase. This is what lets the Jacobian's species columns skip the 45-odd
  // transcendental evaluations that depend on the state only through T. Only
  // the handful of quantities that a species factor later multiplies into need
  // caching, so this costs ~20 scalars rather than a copy of every rate array.
  // See ~/ai-notes/docs-claude/athenak-chemistry/gow17-jacobian-hoist-design.md
  /// Temperature the cache was built at, and its square root
  mutable Real cached_T_ = -1.0;
  mutable Real cached_Tcoll_ = -1.0;
  mutable Real cached_sqrtT_ = 0.0;
  /// Pure-T values of the four 2-body rates that the H2O+ branching scales
  mutable Real k2b_T_H2Oplus_[4] = {0.0, 0.0, 0.0, 0.0};
  /// Collisional dissociation: log10 of the low/high density limits, and
  /// whether the collisional branch is active at this temperature
  mutable Real cached_log10_k9l_ = 0.0, cached_log10_k9h_ = 0.0;
  mutable Real cached_log10_k10l_ = 0.0, cached_log10_k10h_ = 0.0;
  mutable Real cached_ncrH_ = 0.0, cached_ncrH2_ = 0.0;
  mutable bool cached_coll_active_ = false;
  /// Grain photoelectric prefactor, 1.7 * G_PE * sqrt(T) / n_H
  mutable Real cached_psi_gr_fac_ = 0.0;
  /// Heat capacity at the cached T, and N = Cv (gamma-1) / k_B, the quantity
  /// whose derivatives give dT/dx_j for the Jacobian's thermal correction
  mutable Real cached_N_ = 0.0;
  /// Zero when T does not respond to the species: isothermal, or x(H2) sitting
  /// on one of CvCold's clamps
  mutable Real cached_dTdx_scale_ = 0.0;
  /// Grain recombination: the two temperature-dependent sub-expressions of the
  /// H+, C+, He+ and Si+ fits, which otherwise cost 8 pow and 4 log per column
  mutable Real cached_gr_Tfac_[4] = {0.0, 0.0, 0.0, 0.0};
  mutable Real cached_gr_Texp_[4] = {0.0, 0.0, 0.0, 0.0};
  /// enum for indexing into k2body_
  enum : size_t {
    i2body_H3p_C,       // index for H3+ + *C -> CH + H2 reaction
    i2body_H3p_O,       // index for H3+ + *O -> OH + H2 reaction
    i2body_H3p_CO,      // index for H3+ + CO -> HCO+ + H2 reaction
    i2body_Hep_H2,      // index for He+ + H2 -> H+ + *He + *H reaction
    i2body_Hep_CO,      // index for He+ + CO -> C+ + *O + *He reaction
    i2body_Cp_H2,       // index for C+ + H2 -> CH + *H reaction
    i2body_Cp_OH,       // index for C+ + OH -> HCO+ reaction
    i2body_CH_O,        // index for CH + *O -> CO + *H reaction
    i2body_OH_C,        // index for OH + *C -> CO + *H reaction
    i2body_Hep_e,       // index for He+ + *e -> *He reaction
    i2body_H3p_e,       // index for H3+ + *e -> H2 + *H reaction
    i2body_Cp_e,        // index for C+ + *e -> *C reaction
    i2body_HCOp_e,      // index for HCO+ + *e -> CO + *H reaction
    i2body_H2p_H2,      // index for H2+ + H2 -> H3+ + *H reaction
    i2body_Hp_e,        // index for H+ + *e -> *H reaction
    i2body_H2_H,        // index for H2 + *H -> 3 *H reaction
    i2body_H2_H2,       // index for H2 + H2 -> H2 + 2 *H reaction
    i2body_H_e,         // index for *H + *e -> H+ + 2 *e reaction
    i2body_H3p_e_3H,    // index for H3+ + *e -> *3H reaction
    i2body_Hep_H2_H2p,  // index for He+ + H2 -> H2+ + *He reaction
    i2body_CH_H,        // index for CH + *H -> H2 + *C reaction
    i2body_OH_O,        // index for OH + *O -> *O + *O + *H reaction
    i2body_Cp_H2_e,     // index for C+ + H2 + *e -> *C + *H + *H reaction
    i2body_Sip_e,       // index for Si+ + *e -> *Si reaction
    i2body_H3p_O_H2,    // index for H3+ + *O + *e -> H2 + *O + *H reaction
    i2body_Hep_OH,      // index for He+ + OH -> O+ + *He + *H reaction
    i2body_H2p_H,       // index for H2+ + *H -> H+ + H2 reaction
    i2body_Hp_O,        // index for H+ + *O -> O+ + *H reaction
    i2body_Op_H,        // index for O+ + *H -> H+ + *O reaction
    i2body_Op_H2_OH,    // index for O+ + H2 -> OH + *H reaction
    i2body_Op_H2        // index for O+ + H2 -> *O + *H + *H reaction
  };

  // parameters related to CO cooling
  // these are needed for LVG approximation
  const Real gradv_;  // absolute value of velocity gradient in cgs, >0

  /*!
   * \brief Compute the ghost species from the conservation laws
   *
   * \details The ghosts are floored at zero since the sums can go negative
   * when the real species overshoot their elemental budgets. The electron
   * abundance needs no floor because the real species are already floored.
   *
   * \param y_in The current state
   * \return GhostSpecies The ghost species abundances
   */
  template <class vec_type>
  KOKKOS_FUNCTION GhostSpecies
  ComputeGhostSpecies_(const vec_type& y_in) const {
    GhostSpecies ghosts;
    ghosts.Si = Kokkos::fmax(xSi - y_in[ISi_plus], 0.0);
    ghosts.C = Kokkos::fmax(
        xC - y_in[IHCO_plus] - y_in[ICHx] - y_in[ICO] - y_in[IC_plus], 0.0);
    ghosts.O = Kokkos::fmax(
        xO - y_in[IHCO_plus] - y_in[IOHx] - y_in[ICO] - y_in[IO_plus], 0.0);
    ghosts.He = Kokkos::fmax(xHe - y_in[IHE_plus], 0.0);
    ghosts.e = y_in[IHE_plus] + y_in[IC_plus] + y_in[IHCO_plus] +
               y_in[IH3_plus] + y_in[IH2_plus] + y_in[IH_plus] + y_in[IO_plus] +
               y_in[ISi_plus];
    ghosts.H = Kokkos::fmax(1.0 - (y_in[IOHx] + y_in[ICHx] + y_in[IHCO_plus] +
                                   3.0 * y_in[IH3_plus] + 2.0 * y_in[IH2_plus] +
                                   y_in[IH_plus] + 2.0 * y_in[IH2]),
                            0.0);
    return ghosts;
  }

  //----------------------------------------------------------------------------------------
  /*!
   * \brief Update the rates for chemical reactions.
   *
   * \param y_in The current state to compute the rates from
   * \param ghosts The ghost species abundances
   */
  // PERFORMANCE / SPLIT POINT: 248 of the 260 lines below depend on the state
  // only through the scalar T, and hold all 58 transcendental calls (28 pow, 15
  // exp, 6 log10, 5 log, 4 sqrt). Only 12 lines touch y_in or ghosts: the T
  // computation, the cosmic-ray scalings by x(H2) and ghosts.H, h2oplus_ratio,
  // the ncr density blend, and the grain parameter psi. Splitting here so the
  // Jacobian's species columns reuse cached k(T) is the single largest available
  // saving. Note the `*=` accumulation into k2body_ -- a correct split needs the
  // pure-T coefficients kept in their own array rather than skipping lines in
  // place. See
  // ~/ai-notes/docs-claude/athenak-chemistry/gow17-jacobian-hoist-design.md
  template <class vec_type>
  KOKKOS_FUNCTION void UpdateRates_(const vec_type& y_in,
                                    const GhostSpecies& ghosts,
                                    const bool species_only = false) const {
    // Species phase: T and every coefficient that depends only on T are reused
    // from the last full call. Valid only when the caller knows T is unchanged.
    Real T, T_collisional;
    if (species_only) {
      T = cached_T_;
      T_collisional = cached_Tcoll_;
    } else {
    // energy per hydrogen atom
    const Real E_ergs = y_in(IIE) * units_energy_density_cgs / n_H;

    // constant or evolve temperature
    if (isothermal) {
      // isohermal EOS
      T = isothermal_temperature_;
    } else {
      // This line is why hoisting k(T) is not trivial: T depends on x(H2) and,
      // through ghosts.e, on all eight cations. A species perturbation in 10 of
      // the 13 Jacobian columns therefore moves T and every rate coefficient
      // with it. Cv is analytic though, so the coupling can be restored as a
      // rank-1 update -- see the design note above.
      T = E_ergs / Thermo::CvCold(y_in[IH2], xHe, ghosts.e, gamma);
    }

    // cap T above some minimum temperature
    T_collisional = T;
    if (T < temperature_min_rates) {
      T = temperature_min_rates;
      T_collisional = T;
    } else if (T > temperature_max_rates) {
      // do not put upper limit on the temperature for collisional ionization
      // and dissociation rates T_collisional
      T = temperature_max_rates;
    }
    cached_T_ = T;
    cached_Tcoll_ = T_collisional;
    cached_sqrtT_ = Kokkos::sqrt(T);
    // N = 1 - x(H2) + x(He) + x(e), so that Cv = k_B N / (gamma - 1) and
    // T = E / Cv. dT/dx(H2) = +T/N and dT/dx(cation) = -T/N; both vanish when
    // the EOS is isothermal or x(H2) is against a CvCold clamp.
    {
      const Real xH2c = Kokkos::fmin(Kokkos::fmax(y_in[IH2], 0.0), 0.5);
      cached_N_ = 1.0 - xH2c + xHe + Kokkos::fmax(ghosts.e, 0.0);
      const bool clamped = (y_in[IH2] <= 0.0) || (y_in[IH2] >= 0.5);
      cached_dTdx_scale_ =
          (isothermal || clamped || cached_N_ <= 0.0) ? 0.0 : T / cached_N_;
    }
    }  // end of temperature determination

    Real logT = 0.0, logT4coll = 0.0, lnTecoll = 0.0, kida_fac = 0.0;
    if (!species_only) {
      logT = Kokkos::log10(T);
      logT4coll = Kokkos::log10(T_collisional / 1.0e4);
      lnTecoll = Kokkos::log(T_collisional * 8.6173e-5);
      kida_fac = (0.62 + 45.41 / cached_sqrtT_) * n_H;
    }

    Real ncr, n2ncr;
    Real psi;        // H+ grain recombination parameter
    Real kcr_H_fac;  // ratio of total rate to primary rate
    Real t1_CHx, t2_CHx;

    // cosmic ray reactions
    /// coefficients of rates relative to H in s^-1
    constexpr Real kcr_base_[n_cr_] = {2.0, 1.1, 1.0, 520., 92., 6.52, 8400.};
    for (int i = 0; i < n_cr_; i++) {
      kcr_[i] = kcr_base_[i] * rad_[irad_CR];
    }
    // cosmic ray induced photo-reactions, proportional to x(H2)
    //  (0) cr + H2 -> H2+ + *e
    //  (1) cr + *He -> He+ + *e
    //  (2) cr + *H  -> H+ + *e
    //  (3) cr + *C -> C+ + *e     --including direct and cr induce photo
    //  reactions (4) crphoto + CO -> *O + *C (5) cr + CO -> HCO+ + e
    //  --schematic for cr + CO -> CO+ + e (6) cr + Si -> Si+ + e, UMIST12
    kcr_H_fac = 1.15 * 2 * y_in[IH2] + 1.5 * ghosts.H;
    kcr_[0] *= kcr_H_fac;
    kcr_[2] *= kcr_H_fac;
    kcr_[3] *= (2 * y_in[IH2] + 3.85 / kcr_base_[3]);
    kcr_[4] *= 2 * y_in[IH2];
    kcr_[6] *= 2 * y_in[IH2];
    // small number
    constexpr Real small_real = 1e-50;

    //--- H2O+ + e branching--
    // (1) H3+ + *O -> OH + H2
    // (24) H3+ + *O + *e -> H2 + *O + *H
    // Species-dependent, so it is computed in both phases. sqrt(T) comes from
    // the cache rather than being recomputed.
    Real h2oplus_ratio;
    if (ghosts.e < small_real) {
      h2oplus_ratio = 1.0e10;
    } else {
      h2oplus_ratio = 6e-10 * y_in[IH2] / (5.3e-6 / cached_sqrtT_ * ghosts.e);
    }
    const Real fac_H2Oplus_H2 = h2oplus_ratio / (h2oplus_ratio + 1.);
    const Real fac_H2Oplus_e = 1. / (h2oplus_ratio + 1.);

    // 2 body reactions. Everything in this block depends on the state only
    // through T, so the species phase skips it and reuses the coefficients
    // already sitting in k2body_.
    if (!species_only) {
    constexpr Real k2Texp[n_2body_] = {
        0.0,  -0.190, 0.0,    0.0, 0.0, -1.3, 0.0, 0.0,   -0.339, -0.5, -0.52,
        0.0,  -0.64,  0.042,  0.0, 0.0, 0.0,  0.0, -0.52, 0.0,    0.26, 0.0,
        -1.3, -0.62,  -0.190, 0.0, 0.0, 0.0,  0.0, 0.0,   0.0};
    constexpr Real k2body_base[n_2body_] = {
        1.00,    1.99e-9,  1.7e-9,    1.26e-13, 1.6e-9,        3.3e-13 * 0.7,
        1.00,    7.0e-11,  7.95e-10,  1.0e-11,  4.54e-7,       1.00,
        1.06e-5, 1.76e-9,  2.753e-14, 1.00,     1.00,          1.00,
        8.46e-7, 7.20e-15, 2.81e-11,  3.5e-11,  3.3e-13 * 0.3, 1.46e-10,
        1.99e-9, 1.00,     6.4e-10,   1.00,     1.00,          1.6e-9,
        1.6e-9};
    for (int i = 0; i < n_2body_; i++) {
      k2body_[i] = k2body_base[i] * Kokkos::pow(T, k2Texp[i]) * n_H;
    }

    // Special treatment of rates for some equations
    // (0) H3+ + *C -> CH + H2         --Vissapragada2016 new rates
    // rates for H3+ + C forming CH+ and CH2+
    constexpr Real A_kCHx_ = 1.04e-9;
    constexpr Real n_kCHx_ = 2.31e-3;
    constexpr Real c_kCHx_[4] = {3.4e-8, 6.97e-9, 1.31e-7, 1.51e-4};
    constexpr Real Ti_kCHx_[4] = {7.62, 1.38, 2.66e1, 8.11e3};
    t1_CHx = A_kCHx_ * Kokkos::pow(300. / T, n_kCHx_);
    t2_CHx = c_kCHx_[0] * Kokkos::exp(-Ti_kCHx_[0] / T) +
             c_kCHx_[1] * Kokkos::exp(-Ti_kCHx_[1] / T) +
             c_kCHx_[2] * Kokkos::exp(-Ti_kCHx_[2] / T) +
             c_kCHx_[3] * Kokkos::exp(-Ti_kCHx_[3] / T);
    k2body_[0] *= t1_CHx + Kokkos::pow(T, -1.5) * t2_CHx;
    // (3) He+ + H2 -> H+ + *He + *H   --fit to Schauer1989
    k2body_[3] *= Kokkos::exp(-22.5 / T);
    // (5) C+ + H2 -> CH + *H         -- schematic reaction for C+ + H2 -> CH2+
    k2body_[5] *= Kokkos::exp(-23. / T);
    // ---branching of C+ + H2 ------
    // (22) C+ + H2 + *e -> *C + *H + *H
    k2body_[22] *= Kokkos::exp(-23. / T);
    // (6) C+ + OH -> HCO+         -- Schematic equation for C+ + OH -> CO+ + H.
    // Use rates in KIDA website.
    k2body_[6] = 9.15e-10 * kida_fac;
    // (8) OH + *C -> CO + *H          --exp(0.108/T)
    k2body_[8] *= Kokkos::exp(0.108 / T);
    // (9) He+ + *e -> *He             --(17) Case B
    k2body_[9] *= 11.19 + (-1.676 + (-0.2852 + 0.04433 * logT) * logT) * logT;
    // (11) C+ + *e -> *C              -- Include RR and DR, Badnell2003, 2006.
    k2body_[11] = CII_rec_rate_(T) * n_H;
    // (13) H2+ + H2 -> H3+ + *H       --(54) exp(-T/46600)
    k2body_[13] *= Kokkos::exp(-T / 46600.);
    // (14) H+ + *e -> *H              --(12) Case B
    k2body_[14] *= Kokkos::pow(315614.0 / T, 1.5) *
                   Kokkos::pow(1.0 + Kokkos::pow(115188.0 / T, 0.407), -2.242);
    // (25) He+ + OH -> *H + *He + *O(O+)
    k2body_[25] = 1.35e-9 * kida_fac;
    //  --- O+ reactions ---
    //  (27) H+ + *O -> O+ + *H -- exp(-227/T)
    //  (28) O+ + *H -> H+ + *O
    //  (29) O+ + H2 -> OH + *H     -- branching of H2O+
    //  (30) O+ + H2 -> *O + *H + *H  -- branching of H2O+ */
    k2body_[27] *=
        (1.1e-11 * Kokkos::pow(T, 0.517) + 4.0e-10 * Kokkos::pow(T, 6.69e-3)) *
        Kokkos::exp(-227. / T);
    k2body_[28] *=
        4.99e-11 * Kokkos::pow(T, 0.405) + 7.5e-10 * Kokkos::pow(T, -0.458);
    // Stash the pure-T values of the four rates that the H2O+ branching scales,
    // so the species phase can reapply the branching without recomputing them.
    k2b_T_H2Oplus_[0] = k2body_[1];
    k2b_T_H2Oplus_[1] = k2body_[24];
    k2b_T_H2Oplus_[2] = k2body_[29];
    k2b_T_H2Oplus_[3] = k2body_[30];
    }  // end of temperature-only 2-body rates

    // ----- species-dependent H2O+ branching (always runs) -----
    k2body_[1] = k2b_T_H2Oplus_[0] * fac_H2Oplus_H2;
    k2body_[24] = k2b_T_H2Oplus_[1] * fac_H2Oplus_e;
    k2body_[29] = k2b_T_H2Oplus_[2] * fac_H2Oplus_H2;
    k2body_[30] = k2b_T_H2Oplus_[3] * fac_H2Oplus_e;

    // Collisional dissociation, k>~1.0e-30 at T>~5e2.
    // The low/high density limits and the critical densities are pure functions
    // of T, so they are computed once and cached as logs; only the density
    // blend, which needs ghosts.H and x(H2), runs in the species phase.
    Real div_ncr;
    constexpr Real temp_coll = 7.0e2;
    if (!species_only) {
      Real k9l, k9h, k10l, k10h, ncrH, ncrH2;
      cached_coll_active_ = (T_collisional > temp_coll && n_H > small_real);
      if (cached_coll_active_) {
      // (15) H2 + *H -> 3 *H
      // (16) H2 + H2 -> H2 + 2 *H
      // --(9) Density dependent. See Glover+MacLow2007
      k9l = 6.67e-12 * Kokkos::sqrt(T_collisional) *
            Kokkos::exp(-(1. + 63590. / T_collisional));
      k9h = 3.52e-9 * Kokkos::exp(-43900.0 / T_collisional);
      k10l = 5.996e-30 * Kokkos::pow(T_collisional, 4.1881) /
             Kokkos::pow((1.0 + 6.761e-6 * T_collisional), 5.6881) *
             Kokkos::exp(-54657.4 / T_collisional);
      k10h = 1.3e-9 * Kokkos::exp(-53300.0 / T_collisional);
      ncrH = Kokkos::pow(
          10, (3.0 - 0.416 * logT4coll - 0.327 * logT4coll * logT4coll));
      ncrH2 = Kokkos::pow(
          10, (4.845 - 1.3 * logT4coll + 1.62 * logT4coll * logT4coll));
      cached_ncrH_ = ncrH;
      cached_ncrH2_ = ncrH2;
      cached_log10_k9l_ = Kokkos::log10(k9l);
      cached_log10_k9h_ = Kokkos::log10(k9h);
      cached_log10_k10l_ = Kokkos::log10(k10l);
      cached_log10_k10h_ = Kokkos::log10(k10h);
      // (17) *H + *e -> H+ + 2 *e       --(11) Relates to Te
      k2body_[17] *= Kokkos::exp(
          -3.271396786e1 +
          (1.35365560e1 +
           (-5.73932875 +
            (1.56315498 +
             (-2.877056e-1 +
              (3.48255977e-2 +
               (-2.63197617e-3 +
                (1.11954395e-4 + (-2.03914985e-6) * lnTecoll) * lnTecoll) *
                   lnTecoll) *
                  lnTecoll) *
                 lnTecoll) *
                lnTecoll) *
               lnTecoll) *
              lnTecoll);  // NOLINT
      } else {
        k2body_[17] = 0.;
      }
    }  // end of temperature-only collisional rates

    // ----- density blend of the collisional rates (always runs) -----
    if (cached_coll_active_) {
      div_ncr = ghosts.H / (cached_ncrH_ + small_real) +
                y_in[IH2] / (cached_ncrH2_ + small_real);
      if (div_ncr < small_real) {
        ncr = 1. / small_real;
      } else {
        ncr = 1. / div_ncr;
      }
      n2ncr = n_H / ncr;
      k2body_[15] =
          Kokkos::pow(10, cached_log10_k9h_ * n2ncr / (1. + n2ncr) +
                              cached_log10_k9l_ / (1. + n2ncr)) *
          n_H;
      k2body_[16] =
          Kokkos::pow(10, cached_log10_k10h_ * n2ncr / (1. + n2ncr) +
                              cached_log10_k10l_ / (1. + n2ncr)) *
          n_H;
    } else {
      k2body_[15] = 0.;
      k2body_[16] = 0.;
    }

    // photo reactions
    if (!species_only) {
      constexpr Real kph_base_[n_ph] = {3.5e-10, 9.1e-10, 2.4e-10,
                                        3.8e-10, 5.7e-11, 4.5e-9};
      for (int i = 0; i < n_ph; i++) {
        kph_[i] = kph_base_[i] * rad_[i];
      }

      // Grain assisted recombination of H and H2
      //   (0) *H + *H + gr -> H2 + gr
      kgr_[0] = get_kgr_H2_(T) * n_H * zd;
    }
    //   (1) H+ + *e + gr -> *H + gr
    //   (2) C+ + *e + gr -> *C + gr
    //   (3) He+ + *e + gr -> *He + gr
    //   (4) Si+ + *e + gr -> *Si + gr
    //   , rate dependent on e abundance.
    //   (1) H+ + *e + gr -> *H + gr,  (2) C+, (3) He+, (4) Si+
    // The fits share the form
    //   1e-14 c0 / (1 + c1 psi^c2 (1 + c3 T^c4 psi^(-c5 - c6 ln T))) n_H zd
    // in which only psi carries species dependence, so c3 T^c4 and
    // (-c5 - c6 ln T) are cached and the species phase costs two pow per ion
    // instead of three pow and one log.
    constexpr Real c_gr_[4][7] = {
        {12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5},
        {45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4},
        {5.572, 3.185e-7, 1.512, 5.115e3, 3.903e-7, 0.4956, 5.494e-7},
        {2.166, 5.678e-8, 1.874, 4.375e4, 1.635e-6, 0.8964, 7.538e-5}
    };
    if (!species_only) {
      // set lower limit to radiation field in calculating kgr_ to avoid nan
      // values.
      Real GPE_limit = 1.0e-10;
      Real GPE0 = rad_[irad_GPE];
      if (GPE0 < GPE_limit) {
        GPE0 = GPE_limit;
      }
      cached_psi_gr_fac_ = 1.7 * GPE0 * cached_sqrtT_ / n_H;
      const Real lnT = Kokkos::log(T);
      for (int i = 0; i < 4; i++) {
        cached_gr_Tfac_[i] = c_gr_[i][3] * Kokkos::pow(T, c_gr_[i][4]);
        cached_gr_Texp_[i] = -c_gr_[i][5] - c_gr_[i][6] * lnT;
      }
    }

    if (ghosts.e > small_real) {
      psi = cached_psi_gr_fac_ / ghosts.e;
      for (int i = 0; i < 4; i++) {
        kgr_[1 + i] = 1.0e-14 * c_gr_[i][0] /
                      (1.0 + c_gr_[i][1] * Kokkos::pow(psi, c_gr_[i][2]) *
                                 (1.0 + cached_gr_Tfac_[i] *
                                            Kokkos::pow(psi,
                                                        cached_gr_Texp_[i]))) *
                      n_H * zd;
      }
    } else {
      for (int i = 1; i < 5; i++) {
        kgr_[i] = 0.;
      }
    }
  }


  //----------------------------------------------------------------------------------------
  /*!
   * \brief Calculate the rate for CII recombination
   *
   * \param T The temperature
   * \return Real The rate for CII recombination
   */
  KOKKOS_FUNCTION
  Real CII_rec_rate_(const Real T) const {
    constexpr Real A = 2.995e-9;
    constexpr Real B = 0.7849;
    constexpr Real T0 = 6.670e-3;
    constexpr Real T1 = 1.943e6;
    constexpr Real C = 0.1597;
    constexpr Real T2 = 4.955e4;
    const Real BN = B + C * Kokkos::exp(-T2 / T);
    const Real term1 = Kokkos::sqrt(T / T0);
    const Real term2 = Kokkos::sqrt(T / T1);
    const Real alpha_rr = A / (term1 * Kokkos::pow(1.0 + term1, 1.0 - BN) *
                               Kokkos::pow(1.0 + term2, 1.0 + BN));
    const Real alpha_dr =
        Kokkos::pow(T, -3.0 / 2.0) * (6.346e-9 * Kokkos::exp(-1.217e1 / T) +
                                      9.793e-09 * Kokkos::exp(-7.38e1 / T) +
                                      1.634e-06 * Kokkos::exp(-1.523e+04 / T));
    return (alpha_rr + alpha_dr);
  }

  //----------------------------------------------------------------------------------------
  /*!
   * \brief H2 formation rate on dust grains. TIGRESS-NCR (Kim+23)
   * implementation.
   *
   * \param T The temperature
   * \return Real H2 formation rate on dust grains
   */
  KOKKOS_FUNCTION
  Real get_kgr_H2_(const Real T) const {
    Real kgr;
    const Real kgr0 = 3.0e-17;
    if (is_kgrH2_const) {
      // Use temperature independent rate
      kgr = kgr0;
    } else {
      // Use temperature dependent rate from Hollenbach & McKee (1979)
      // Taking Tgr2 = 0 and renormalized to have kgr ~ kgr_H2 near 200>T>50
      const Real T2 = T * 1e-2;
      kgr = kgr0 * Kokkos::sqrt(T2) * 2.0 /
            (1 + 0.4 * Kokkos::sqrt(T2) + 0.2 * T2 + 0.08 * T2 * T2);
    }
    return kgr;
  }

  //----------------------------------------------------------------------------------------
  /*!
   * \brief set gradients of v and nH for CO cooling
   *
   * \param k The k index of the cell to work on
   * \param j The j index of the cell to work on
   * \param i The i index of the cell to work on
   */
  KOKKOS_FUNCTION
  Real SetGradv_(const int mb_idx, const int k, const int j, const int i,
                 const DvceArray5D<Real> w0,
                 const DualArray1D<RegionSize> sizes, const Real velocity_cgs,
                 const Real length_cgs, const bool multi_d,
                 const bool three_d) const {
    // velocity gradient, same as LVG approximation in RADMC-3D when calculating
    // CO line emission. Note that since we're taking the average of the left
    // and right slopes and dx is the same for both the velocity in the central
    // cell cancels out and we can just difference the neighboring cells

    // vx
    const Real vp1 = w0(mb_idx, IVX, k, j, i + 1);
    const Real vm1 = w0(mb_idx, IVX, k, j, i - 1);
    const Real dx = sizes.d_view(mb_idx).dx1;
    const Real dvdx = 0.5 * ((vp1 - vm1) / dx);

    // vy
    Real dvdy = 0;
    if (multi_d) {
      const Real vp1 = w0(mb_idx, IVY, k, j + 1, i);
      const Real vm1 = w0(mb_idx, IVY, k, j - 1, i);
      const Real dx = sizes.d_view(mb_idx).dx2;
      dvdy = 0.5 * ((vp1 - vm1) / dx);
    }

    // vz
    Real dvdz = 0;
    if (three_d) {
      const Real vp1 = w0(mb_idx, IVZ, k + 1, j, i);
      const Real vm1 = w0(mb_idx, IVZ, k - 1, j, i);
      const Real dx = sizes.d_view(mb_idx).dx3;
      dvdz = 0.5 * ((vp1 - vm1) / dx);
    }
    const Real dvdr_avg =
        (Kokkos::abs(dvdx) + Kokkos::abs(dvdy) + Kokkos::abs(dvdz)) / 3.;
    // return gradv_, in cgs.
    return dvdr_avg * velocity_cgs / length_cgs;
  }
};  // class GOW17Network
};  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_GOW17_HPP_
