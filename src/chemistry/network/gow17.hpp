#ifndef CHEMISTRY_NETWORK_GOW17_HPP_
#define CHEMISTRY_NETWORK_GOW17_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gow17.hpp
//  \brief The implementation for the struct for the GOW17 chemistry network

#include <utility>

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

  /// Species order for the semi-implicit solver. Each species reads the new
  /// values of those before it: He+ feeds H+, C+, H2+ and O+; the H2+ -> H3+
  /// chain comes before the ions it creates; CHx and OHx before CO; H2 last,
  /// since it changes slowly and every ion reacts with it. HCO+ is not listed:
  /// CO and HCO+ create each other, and UpdateSpecies<ICO> solves them together.
  using UpdateOrder =
      std::integer_sequence<int, IHE_plus, ISi_plus, IH2_plus, IH3_plus,
                            IH_plus, IO_plus, IC_plus, ICHx, IOHx, ICO, IH2>;

  /// Species whose relative change limits the semi-implicit substep. The ions
  /// relax to equilibrium within a substep and need no limit; H2 and CO are the
  /// slow species that carry the evolution.
  using StepLimitingSpecies = std::integer_sequence<int, IH2, ICO>;

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
   * \details Computes the ghost species and updates the reaction rates.
   *
   * \param y_in The current state
   * \return GhostSpecies The ghost species abundances
   */
  template <class vec_type>
  KOKKOS_FUNCTION GhostSpecies SetupNextStep(const vec_type& y_in) const {
    // Set the ghost species
    const GhostSpecies ghosts = ComputeGhostSpecies_(y_in);

    // Compute rates
    UpdateRates_(y_in, ghosts);

    return ghosts;
  }

  /// The semi-implicit solver calls UpdateSpecies<K> for each species
  static constexpr bool custom_species_update = true;

  /*!
   * \brief One backward-Euler step of species K for the semi-implicit solver,
   * y_K = (y_K^n + h C) / (1 + h D), with C and D from the current state.
   *
   * \details Written per species so that each step evaluates only its own
   * terms. He+, Si+, H+, O+ and C+ are created by ionizing a neutral X0 that is
   * not stored but is the remainder of its element, x_X0 = x_tot - x_X+ - ....
   * With a the total ionization rate of X0 and C' the other creation terms, the
   * ion equation
   *   dx_X+/dt = a x_X0 + C' - D x_X+
   * is written as
   *   dx_X+/dt = a (x_X0 + x_X+) + C' - (a + D) x_X+,
   * which has the same right-hand side. For C' = D = 0, the backward Euler step
   * of this form gives x_X0(new) = x_X0 / (1 + h a), a backward Euler step for
   * the neutral. The first form gives x_X0(new) = x_X0 (1 - h a), a forward
   * Euler step that goes negative for h > 1/a.
   *
   * \param y The state, updated in place
   * \param y_n The state at the start of the substep
   * \param g The ghost species at the start of the substep
   * \param h The substep size
   */
  template <int K, class vec_type>
  KOKKOS_FUNCTION void UpdateSpecies(const vec_type& y, const Real* y_n,
                                     const GhostSpecies& g,
                                     const Real h) const {
    Real c, d;
    if constexpr (K == IHE_plus) {
      const Real a = kcr_[1];
      c = a * (g.He + y[IHE_plus]);
      d = k2body_[i2body_Hep_H2] * y[IH2] + k2body_[i2body_Hep_CO] * y[ICO] +
          k2body_[i2body_Hep_e] * g.e + k2body_[i2body_Hep_H2_H2p] * y[IH2] +
          k2body_[i2body_Hep_OH] * y[IOHx] + kgr_[igr_Hep] + a;
    } else if constexpr (K == ISi_plus) {
      const Real a = kcr_[6] + kph_[iph_Si];
      c = a * (g.Si + y[ISi_plus]);
      d = k2body_[i2body_Sip_e] * g.e + kgr_[igr_Sip] + a;
    } else if constexpr (K == IH2_plus) {
      c = kcr_[0] * y[IH2] + k2body_[i2body_Hep_H2_H2p] * y[IHE_plus] * y[IH2];
      d = k2body_[i2body_H2p_H2] * y[IH2] + k2body_[i2body_H2p_H] * g.H;
    } else if constexpr (K == IH3_plus) {
      c = k2body_[i2body_H2p_H2] * y[IH2_plus] * y[IH2];
      d = k2body_[i2body_H3p_C] * g.C + k2body_[i2body_H3p_O] * g.O +
          k2body_[i2body_H3p_CO] * y[ICO] + k2body_[i2body_H3p_e] * g.e +
          k2body_[i2body_H3p_e_3H] * g.e + k2body_[i2body_H3p_O_H2] * g.O;
    } else if constexpr (K == IH_plus) {
      const Real a = kcr_[2] + k2body_[i2body_H_e] * g.e;
      c = a * (g.H + y[IH_plus]) +
          k2body_[i2body_Hep_H2] * y[IHE_plus] * y[IH2] +
          k2body_[i2body_H2p_H] * y[IH2_plus] * g.H +
          k2body_[i2body_Op_H] * y[IO_plus] * g.H;
      d = k2body_[i2body_Hp_e] * g.e + k2body_[i2body_Hp_O] * g.O +
          kgr_[igr_Hp] + a;
    } else if constexpr (K == IO_plus) {
      const Real a = k2body_[i2body_Hp_O] * y[IH_plus];
      c = k2body_[i2body_Hep_OH] * y[IHE_plus] * y[IOHx] +
          a * (g.O + y[IO_plus]);
      d = k2body_[i2body_Op_H] * g.H + k2body_[i2body_Op_H2_OH] * y[IH2] +
          k2body_[i2body_Op_H2] * y[IH2] + a;
    } else if constexpr (K == IC_plus) {
      const Real a = kcr_[3] + kph_[iph_C];
      c = a * (g.C + y[IC_plus]) +
          k2body_[i2body_Hep_CO] * y[IHE_plus] * y[ICO];
      d = k2body_[i2body_Cp_H2] * y[IH2] + k2body_[i2body_Cp_OH] * y[IOHx] +
          k2body_[i2body_Cp_e] * g.e + k2body_[i2body_Cp_H2_e] * y[IH2] +
          kgr_[igr_Cp] + a;
    } else if constexpr (K == ICHx) {
      c = k2body_[i2body_H3p_C] * y[IH3_plus] * g.C +
          k2body_[i2body_Cp_H2] * y[IC_plus] * y[IH2];
      d = k2body_[i2body_CH_O] * g.O + k2body_[i2body_CH_H] * g.H +
          kph_[iph_CHx];
    } else if constexpr (K == IOHx) {
      c = k2body_[i2body_H3p_O] * y[IH3_plus] * g.O +
          k2body_[i2body_Op_H2_OH] * y[IO_plus] * y[IH2];
      d = k2body_[i2body_Cp_OH] * y[IC_plus] + k2body_[i2body_OH_C] * g.C +
          k2body_[i2body_OH_O] * g.O + k2body_[i2body_Hep_OH] * y[IHE_plus] +
          kph_[iph_OHx];
    } else if constexpr (K == ICO) {
      // CO and HCO+ together, backward Euler on the 2x2
      //   (1 + d1 h) CO - b h HCO+ = CO^n + a h
      //   -cc h CO + (1 + d2 h) HCO+ = HCO+^n + f h
      // b <= d2 and cc <= d1, so det >= 1 + (d1 + d2) h > 0 and both solutions
      // are sums of non-negative terms.
      const Real u = units_time_cgs;
      const Real a = u * (k2body_[i2body_CH_O] * y[ICHx] * g.O +
                          k2body_[i2body_OH_C] * y[IOHx] * g.C);
      const Real b = u * k2body_[i2body_HCOp_e] * g.e;
      const Real d1 = u * (kcr_[4] + kcr_[5] +
                           k2body_[i2body_H3p_CO] * y[IH3_plus] +
                           k2body_[i2body_Hep_CO] * y[IHE_plus] + kph_[iph_CO]);
      const Real f = u * k2body_[i2body_Cp_OH] * y[IC_plus] * y[IOHx];
      const Real cc = u * (kcr_[5] + k2body_[i2body_H3p_CO] * y[IH3_plus]);
      const Real d2 = u * k2body_[i2body_HCOp_e] * g.e;
      const Real A = 1.0 + d1 * h;
      const Real B = 1.0 + d2 * h;
      const Real det = A * B - b * cc * h * h;
      y[ICO] = (B * (y_n[ICO] + a * h) + b * h * (y_n[IHCO_plus] + f * h)) / det;
      y[IHCO_plus] = (y_n[IHCO_plus] + f * h + cc * h * y[ICO]) / B;
      return;
    } else if constexpr (K == IH2) {
      c = k2body_[i2body_H3p_C] * y[IH3_plus] * g.C +
          k2body_[i2body_H3p_O] * y[IH3_plus] * g.O +
          k2body_[i2body_H3p_CO] * y[IH3_plus] * y[ICO] +
          k2body_[i2body_H3p_e] * y[IH3_plus] * g.e +
          k2body_[i2body_H2_H2] * y_n[IH2] * y_n[IH2] +
          k2body_[i2body_CH_H] * y[ICHx] * g.H +
          k2body_[i2body_H3p_O_H2] * y[IH3_plus] * g.O +
          k2body_[i2body_H2p_H] * y[IH2_plus] * g.H + kgr_[igr_H] * g.H;
      d = kcr_[0] + k2body_[i2body_Hep_H2] * y[IHE_plus] +
          k2body_[i2body_Cp_H2] * y[IC_plus] +
          k2body_[i2body_H2p_H2] * y[IH2_plus] + k2body_[i2body_H2_H] * g.H +
          2.0 * k2body_[i2body_H2_H2] * y_n[IH2] +
          k2body_[i2body_Hep_H2_H2p] * y[IHE_plus] +
          k2body_[i2body_Cp_H2_e] * y[IC_plus] +
          k2body_[i2body_Op_H2_OH] * y[IO_plus] +
          k2body_[i2body_Op_H2] * y[IO_plus] + kph_[iph_H2];
    } else {
      static_assert(K < 0, "no semi-implicit update for this species");
    }
    y[K] = (y_n[K] + h * units_time_cgs * c) / (1.0 + h * units_time_cgs * d);
  }

  /*!
   * \brief Computes `f` using the values in `y_in`
   */
  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION __attribute__((noinline)) void evaluate_function(
      const Real /*t*/, const Real /*dt*/, const vec_type1& y_in,
      vec_type2& f) const {
    RegisterArray<Real, neqs> y_floor;
    for (size_t i = 0; i < neqs; i++) {
      // Check if inf or NaN valued and throw abort if that's the case
      if (Kokkos::isinf(y_in(i)) || Kokkos::isnan(y_in(i))) {
        Kokkos::abort("Error: NaN or Inf value found in GOW17 `y` array\n");
      }

      // Floor the values
      y_floor(i) = Kokkos::fmax(y_in(i), 0.0);
    }

    // ----- Setup for the next step -----
    const auto ghosts = SetupNextStep(y_floor);

    // ----- Internal energy equation -----
    f(IIE) = Edot(y_floor, ghosts);

    // ----- Creation & Destruction Rates -----
    const auto rates = CDRates(y_floor, ghosts);

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
    chemistry::numerical_jacobian(*this, t, dt, y_in, jac);
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
  template <class vec_type>
  KOKKOS_FUNCTION void UpdateRates_(const vec_type& y_in,
                                    const GhostSpecies& ghosts) const {
    // energy per hydrogen atom
    const Real E_ergs = y_in(IIE) * units_energy_density_cgs / n_H;

    Real T;
    // constant or evolve temperature
    if (isothermal) {
      // isohermal EOS
      T = isothermal_temperature_;
    } else {
      T = E_ergs / Thermo::CvCold(y_in[IH2], xHe, ghosts.e, gamma);
    }

    // cap T above some minimum temperature
    Real T_collisional = T;
    if (T < temperature_min_rates) {
      T = temperature_min_rates;
      T_collisional = T;
    } else if (T > temperature_max_rates) {
      // do not put upper limit on the temperature for collisional ionization
      // and dissociation rates T_collisional
      T = temperature_max_rates;
    }

    const Real logT = Kokkos::log10(T);
    // Natural log of the (clamped) temperature. Many rates below are powers of
    // T or of psi; rewriting pow(a,b) as exp(b*log(a)) lets them share this one
    // log instead of each transcendental recomputing one internally.
    const Real lnT = Kokkos::log(T);
    // pow(10, x) == exp(x*ln10), which avoids the general power routine.
    constexpr Real ln10_ = 2.30258509299404590e0;
    const Real logT4coll = Kokkos::log10(T_collisional / 1.0e4);
    const Real lnTecoll = Kokkos::log(T_collisional * 8.6173e-5);

    Real ncr, n2ncr;
    Real psi;        // H+ grain recombination parameter
    Real kcr_H_fac;  // ratio of total rate to primary rate
    Real psi_gr_fac_;
    const Real kida_fac = (0.62 + 45.41 / Kokkos::sqrt(T)) * n_H;
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
    // 2 body reactions
    constexpr Real k2body_base[n_2body_] = {
        1.00,    1.99e-9,  1.7e-9,    1.26e-13, 1.6e-9,        3.3e-13 * 0.7,
        1.00,    7.0e-11,  7.95e-10,  1.0e-11,  4.54e-7,       1.00,
        1.06e-5, 1.76e-9,  2.753e-14, 1.00,     1.00,          1.00,
        8.46e-7, 7.20e-15, 2.81e-11,  3.5e-11,  3.3e-13 * 0.3, 1.46e-10,
        1.99e-9, 1.00,     6.4e-10,   1.00,     1.00,          1.6e-9,
        1.6e-9};
    for (int i = 0; i < n_2body_; i++) {
      k2body_[i] = k2body_base[i] * n_H;
    }
    // The nine distinct nonzero exponents, applied to their reactions.
    // Each pow(T, e) is written exp(e*lnT) to reuse the shared lnT above.
    const Real tp_m0190 = Kokkos::exp(-0.190 * lnT);  // (1) (24)
    const Real tp_m13 = Kokkos::exp(-1.3 * lnT);      // (5) (22)
    const Real tp_m0339 = Kokkos::exp(-0.339 * lnT);  // (8)
    const Real tp_m05 = Kokkos::exp(-0.5 * lnT);      // (9)
    const Real tp_m052 = Kokkos::exp(-0.52 * lnT);    // (10) (18)
    const Real tp_m064 = Kokkos::exp(-0.64 * lnT);    // (12)
    const Real tp_0042 = Kokkos::exp(0.042 * lnT);    // (13)
    const Real tp_026 = Kokkos::exp(0.26 * lnT);      // (20)
    const Real tp_m062 = Kokkos::exp(-0.62 * lnT);    // (23)
    k2body_[i2body_H3p_O] *= tp_m0190;
    k2body_[i2body_H3p_O_H2] *= tp_m0190;
    k2body_[i2body_Cp_H2] *= tp_m13;
    k2body_[i2body_Cp_H2_e] *= tp_m13;
    k2body_[i2body_OH_C] *= tp_m0339;
    k2body_[i2body_Hep_e] *= tp_m05;
    k2body_[i2body_H3p_e] *= tp_m052;
    k2body_[i2body_H3p_e_3H] *= tp_m052;
    k2body_[i2body_HCOp_e] *= tp_m064;
    k2body_[i2body_H2p_H2] *= tp_0042;
    k2body_[i2body_CH_H] *= tp_026;
    k2body_[i2body_Sip_e] *= tp_m062;

    // Special treatment of rates for some equations
    // (0) H3+ + *C -> CH + H2         --Vissapragada2016 new rates
    // rates for H3+ + C forming CH+ and CH2+
    constexpr Real A_kCHx_ = 1.04e-9;
    constexpr Real n_kCHx_ = 2.31e-3;
    // Natural logs of the constants below, used to rewrite pow(c/T, e) as
    // exp(e*(log(c)-log(T))) so the powers share the single lnT.
    constexpr Real ln300_ = 5.70378247465620092e0;
    constexpr Real ln315614_ = 1.26622752269192862e1;
    constexpr Real ln115188_ = 1.16543208551515374e1;
    constexpr Real c_kCHx_[4] = {3.4e-8, 6.97e-9, 1.31e-7, 1.51e-4};
    constexpr Real Ti_kCHx_[4] = {7.62, 1.38, 2.66e1, 8.11e3};
    t1_CHx = A_kCHx_ * Kokkos::exp(n_kCHx_ * (ln300_ - lnT));
    t2_CHx = c_kCHx_[0] * Kokkos::exp(-Ti_kCHx_[0] / T) +
             c_kCHx_[1] * Kokkos::exp(-Ti_kCHx_[1] / T) +
             c_kCHx_[2] * Kokkos::exp(-Ti_kCHx_[2] / T) +
             c_kCHx_[3] * Kokkos::exp(-Ti_kCHx_[3] / T);
    k2body_[i2body_H3p_C] *= t1_CHx + Kokkos::exp(-1.5 * lnT) * t2_CHx;
    // (3) He+ + H2 -> H+ + *He + *H   --fit to Schauer1989
    k2body_[i2body_Hep_H2] *= Kokkos::exp(-22.5 / T);
    // (5) C+ + H2 -> CH + *H         -- schematic reaction for C+ + H2 -> CH2+
    // (22) C+ + H2 + *e -> *C + *H + *H  (branching of C+ + H2) share this
    // activation factor, so evaluate it once.
    const Real exp_m23_T = Kokkos::exp(-23. / T);
    k2body_[i2body_Cp_H2] *= exp_m23_T;
    k2body_[i2body_Cp_H2_e] *= exp_m23_T;
    // (6) C+ + OH -> HCO+         -- Schematic equation for C+ + OH -> CO+ + H.
    // Use rates in KIDA website.
    k2body_[i2body_Cp_OH] = 9.15e-10 * kida_fac;
    // (8) OH + *C -> CO + *H          --exp(0.108/T)
    k2body_[i2body_OH_C] *= Kokkos::exp(0.108 / T);
    // (9) He+ + *e -> *He             --(17) Case B
    k2body_[i2body_Hep_e] *=
        11.19 + (-1.676 + (-0.2852 + 0.04433 * logT) * logT) * logT;
    // (11) C+ + *e -> *C              -- Include RR and DR, Badnell2003, 2006.
    k2body_[i2body_Cp_e] = CII_rec_rate_(T) * n_H;
    // (13) H2+ + H2 -> H3+ + *H       --(54) exp(-T/46600)
    k2body_[i2body_H2p_H2] *= Kokkos::exp(-T / 46600.);
    // (14) H+ + *e -> *H              --(12) Case B
    // pow(c/T, e) rewritten as exp(e*(log c - log T)) to reuse lnT; the outer
    // power has a non-constant base so it stays as pow.
    k2body_[i2body_Hp_e] *=
        Kokkos::exp(1.5 * (ln315614_ - lnT)) *
        Kokkos::pow(1.0 + Kokkos::exp(0.407 * (ln115188_ - lnT)), -2.242);
    //--- H2O+ + e branching--
    // (1) H3+ + *O -> OH + H2
    // (24) H3+ + *O + *e -> H2 + *O + *H
    Real h2oplus_ratio, fac_H2Oplus_H2, fac_H2Oplus_e;
    // small number
    constexpr Real small_real = 1e-50;
    if (ghosts.e < small_real) {
      h2oplus_ratio = 1.0e10;
    } else {
      h2oplus_ratio = 6e-10 * y_in[IH2] / (5.3e-6 / Kokkos::sqrt(T) * ghosts.e);
    }
    fac_H2Oplus_H2 = h2oplus_ratio / (h2oplus_ratio + 1.);
    fac_H2Oplus_e = 1. / (h2oplus_ratio + 1.);
    k2body_[i2body_H3p_O] *= fac_H2Oplus_H2;
    k2body_[i2body_H3p_O_H2] *= fac_H2Oplus_e;
    // (25) He+ + OH -> *H + *He + *O(O+)
    k2body_[i2body_Hep_OH] = 1.35e-9 * kida_fac;
    //  --- O+ reactions ---
    //  (27) H+ + *O -> O+ + *H -- exp(-227/T)
    //  (28) O+ + *H -> H+ + *O
    //  (29) O+ + H2 -> OH + *H     -- branching of H2O+
    //  (30) O+ + H2 -> *O + *H + *H  -- branching of H2O+ */
    k2body_[i2body_Hp_O] *= (1.1e-11 * Kokkos::exp(0.517 * lnT) +
                             4.0e-10 * Kokkos::exp(6.69e-3 * lnT)) *
                            Kokkos::exp(-227. / T);
    k2body_[i2body_Op_H] *= 4.99e-11 * Kokkos::exp(0.405 * lnT) +
                            7.5e-10 * Kokkos::exp(-0.458 * lnT);
    k2body_[i2body_Op_H2_OH] *= fac_H2Oplus_H2;
    k2body_[i2body_Op_H2] *= fac_H2Oplus_e;

    // Collisional dissociation, k>~1.0e-30 at T>~5e2.
    Real k9l, k9h, k10l, k10h, ncrH, ncrH2, div_ncr;
    constexpr Real temp_coll = 7.0e2;
    if (T_collisional > temp_coll && n_H > small_real) {
      // (15) H2 + *H -> 3 *H
      // (16) H2 + H2 -> H2 + 2 *H
      // --(9) Density dependent. See Glover+MacLow2007
      // T_collisional is only clamped above, so it needs its own log; the two
      // powers of it below then share it.
      const Real lnTcoll = Kokkos::log(T_collisional);
      k9l = 6.67e-12 * Kokkos::sqrt(T_collisional) *
            Kokkos::exp(-(1. + 63590. / T_collisional));
      k9h = 3.52e-9 * Kokkos::exp(-43900.0 / T_collisional);
      k10l = 5.996e-30 * Kokkos::exp(4.1881 * lnTcoll) /
             Kokkos::pow((1.0 + 6.761e-6 * T_collisional), 5.6881) *
             Kokkos::exp(-54657.4 / T_collisional);
      k10h = 1.3e-9 * Kokkos::exp(-53300.0 / T_collisional);
      // pow(10, x) == exp(x*log(10)), which avoids the general power routine.
      ncrH = Kokkos::exp(
          ln10_ * (3.0 - 0.416 * logT4coll - 0.327 * logT4coll * logT4coll));
      ncrH2 = Kokkos::exp(
          ln10_ * (4.845 - 1.3 * logT4coll + 1.62 * logT4coll * logT4coll));
      div_ncr =
          ghosts.H / (ncrH + small_real) + y_in[IH2] / (ncrH2 + small_real);
      if (div_ncr < small_real) {
        ncr = 1. / small_real;
      } else {
        ncr = 1. / div_ncr;
      }
      n2ncr = n_H / ncr;
      k2body_[i2body_H2_H] =
          Kokkos::exp(ln10_ * (Kokkos::log10(k9h) * n2ncr / (1. + n2ncr) +
                               Kokkos::log10(k9l) / (1. + n2ncr))) *
          n_H;
      k2body_[i2body_H2_H2] =
          Kokkos::exp(ln10_ * (Kokkos::log10(k10h) * n2ncr / (1. + n2ncr) +
                               Kokkos::log10(k10l) / (1. + n2ncr))) *
          n_H;
      // (17) *H + *e -> H+ + 2 *e       --(11) Relates to Te
      k2body_[i2body_H_e] *= Kokkos::exp(
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
              lnTecoll);
    } else {
      k2body_[i2body_H2_H] = 0.;
      k2body_[i2body_H2_H2] = 0.;
      k2body_[i2body_H_e] = 0.;
    }

    // photo reactions
    constexpr Real kph_base_[n_ph] = {3.5e-10, 9.1e-10, 2.4e-10,
                                      3.8e-10, 5.7e-11, 4.5e-9};
    for (int i = 0; i < n_ph; i++) {
      kph_[i] = kph_base_[i] * rad_[i];
    }

    // Grain assisted recombination of H and H2
    //   (0) *H + *H + gr -> H2 + gr
    kgr_[0] = get_kgr_H2_(T) * n_H * zd;
    //   (1) H+ + *e + gr -> *H + gr
    //   (2) C+ + *e + gr -> *C + gr
    //   (3) He+ + *e + gr -> *He + gr
    //   (4) Si+ + *e + gr -> *Si + gr
    //   , rate dependent on e abundance.
    if (ghosts.e > small_real) {
      constexpr Real cHp_[7] = {12.25,    8.074e-6, 1.378,   5.087e2,
                                1.586e-2, 0.4723,   1.102e-5};
      constexpr Real cCp_[7] = {45.58,    6.089e-3, 1.128,   4.331e2,
                                4.845e-2, 0.8120,   1.333e-4};
      constexpr Real cHep_[7] = {5.572,    3.185e-7, 1.512,   5.115e3,
                                 3.903e-7, 0.4956,   5.494e-7};
      constexpr Real cSip_[7] = {2.166,    5.678e-8, 1.874,   4.375e4,
                                 1.635e-6, 0.8964,   7.538e-5};
      // set lower limit to radiation field in calculating kgr_ to avoid nan
      // values.
      Real GPE_limit = 1.0e-10;
      Real GPE0 = rad_[irad_GPE];
      if (GPE0 < GPE_limit) {
        GPE0 = GPE_limit;
      }
      psi_gr_fac_ = 1.7 * GPE0 * Kokkos::sqrt(T) / n_H;
      psi = psi_gr_fac_ / ghosts.e;
      // ln(psi) appears in the exponents of all four species below (lnT is
      // already in scope), so compute it once and rewrite each pow as
      // exp(exponent * log(base)).
      const Real lnPsi = Kokkos::log(psi);
      kgr_[1] =
          1.0e-14 * cHp_[0] /
          (1.0 +
           cHp_[1] * Kokkos::exp(cHp_[2] * lnPsi) *
               (1.0 + cHp_[3] * Kokkos::exp(cHp_[4] * lnT) *
                          Kokkos::exp((-cHp_[5] - cHp_[6] * lnT) * lnPsi))) *
          n_H * zd;
      kgr_[2] =
          1.0e-14 * cCp_[0] /
          (1.0 +
           cCp_[1] * Kokkos::exp(cCp_[2] * lnPsi) *
               (1.0 + cCp_[3] * Kokkos::exp(cCp_[4] * lnT) *
                          Kokkos::exp((-cCp_[5] - cCp_[6] * lnT) * lnPsi))) *
          n_H * zd;
      kgr_[3] =
          1.0e-14 * cHep_[0] /
          (1.0 +
           cHep_[1] * Kokkos::exp(cHep_[2] * lnPsi) *
               (1.0 + cHep_[3] * Kokkos::exp(cHep_[4] * lnT) *
                          Kokkos::exp((-cHep_[5] - cHep_[6] * lnT) * lnPsi))) *
          n_H * zd;
      kgr_[4] =
          1.0e-14 * cSip_[0] /
          (1.0 +
           cSip_[1] * Kokkos::exp(cSip_[2] * lnPsi) *
               (1.0 + cSip_[3] * Kokkos::exp(cSip_[4] * lnT) *
                          Kokkos::exp((-cSip_[5] - cSip_[6] * lnT) * lnPsi))) *
          n_H * zd;
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
    // T^(-3/2) = T0^(-3/2) * (T/T0)^(-3/2) = T0^(-3/2) / term1^3, so the
    // power comes for free from term1 instead of a pow().
    constexpr Real T0_inv15_ = 1.83574032975375144e3;
    const Real alpha_dr = (T0_inv15_ / (term1 * term1 * term1)) *
                          (6.346e-9 * Kokkos::exp(-1.217e1 / T) +
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
