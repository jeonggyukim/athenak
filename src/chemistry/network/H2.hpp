#ifndef CHEMISTRY_NETWORK_H2_HPP_
#define CHEMISTRY_NETWORK_H2_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file H2.hpp
//  \brief The implementation for the struct for the H2 chemistry network

#include <utility>

#include <Kokkos_ArithTraits.hpp>

#include "athena.hpp"
#include "chemistry/chemistry_utils.hpp"
#include "chemistry/thermo/thermo.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
struct H2Settings {
  /// If C_v should be held constant
  bool const_cv;
  /// If we're using an isothermal equation of state
  bool isothermal;
};

/*!
 * \brief The class for the H2 network. This also serves as a template for other
 * chemistry networks.
 *
 * \details All chemistry networks are required to have a handful of specific
 * things so that the ODE solvers have a common interface to work with. They
 * need:
 *   - A `neqs` variable that specifies the number of equations. That should be
 *     the number of species plus 1 for the internal energy .
 *   - `y` and `f` RegisterArray variables to hold the current state and the
 *     result of evaluating the equations respectively.
 *   - An `evaluate_function` method that computes `f` from `y`
 * For the semi-implicit solver they also need an `UpdateOrder`, and
 * `SetupNextStep`, `CDRates` and `Edot` methods; see semi_implicit.hpp.
 */
class H2Network {
 public:
  KOKKOS_FUNCTION H2Network(H2Settings const settings, const int mb_idx,
                            const int k, const int j, const int i,
                            const DvceArray5D<Real> w0,
                            const DualArray1D<RegionSize> sizes,
                            const DvceArray1D<Real> ir, Real const density_cgs,
                            Real const mu_H, Real const gamma,
                            Real const hydrogen_mass_cgs,
                            Real const units_time_cgs,
                            Real const units_energy_density_cgs)
      : n_H(w0(mb_idx, IDN, k, j, i) * density_cgs /
            (mu_H * hydrogen_mass_cgs)),
        gamma(gamma),
        units_time_cgs(units_time_cgs),
        units_energy_density_cgs(units_energy_density_cgs),
        const_cv(settings.const_cv),
        isothermal(settings.isothermal),
        y(y_buffer_, neqs),
        y_new(y_new_buffer_, neqs) {}

  // ----- Number of equations -----
  static constexpr int neqs = 3;

  // ----- If Cv is const or not -----
  const bool const_cv;

  /// If the network is using an isothermal equation of state
  const bool isothermal;

  // ----- Views to store ODE state -----
  // The current state
  Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> y;
  // The results of evaluating the ODEs
  Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> y_new;

  // ----- Species indices within the ODE system ------
  enum {
    IH2,  // H_2
    IH,   // H
    IIE   // internal energy, must be last
  };

  // ----- Names, used for output, must be the same order as the enum -----
  static constexpr std::array<std::string_view, neqs - 1> species_names = {"H2",
                                                                           "H"};

  /// Species order for the semi-implicit solver. Only H2 is listed;
  /// UpdateSpecies<IH2> sets H from hydrogen conservation.
  using UpdateOrder = std::integer_sequence<int, IH2>;

  /// Species whose relative change limits the semi-implicit substep
  using StepLimitingSpecies = std::integer_sequence<int, IH2>;

  /// The H2 network has no ghost species; this exists for API consistency
  /// across networks
  struct GhostSpecies {};

  // ----- cell values -----
  Real const n_H;  // The number density of hydrogen
  Real const gamma;

  // ----- unit conversion factors -----
  Real const units_time_cgs;
  Real const units_energy_density_cgs;

  // ----- Reaction rate constants -----
  static constexpr Real k_gr = 3.0e-17;
  // xi_cr is the primary cosmic-ray ionization rate per H
  static constexpr Real xi_cr = 2.0e-16;
  static constexpr Real k_cr = 3.0 * xi_cr;

  // ----- Member Functions -----
  /*!
   * \brief Get the settings for the H2 network from the input file
   *
   * \param pin The ParameterInput object
   * \return H2Settings The settings for the H2 network
   */
  static H2Settings GetSettings(ParameterInput* pin, MeshBlockPack* ppack) {
    return H2Settings{
        pin->GetOrAddBoolean("chemistry", "h2_constant_cv", false),
        pin->GetOrAddBoolean("chemistry", "h2_isothermal", false)};
  }

  /*!
   * \brief Compute the temperature in the cell
   *
   * \param y_in The current state to compute the temperature from
   * \return Real The temperature in the cell
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real Temperature(const vec_type& y_in) const {
    // Constants needed for computing temperature
    static constexpr Real x_He = 0.1;
    static constexpr Real x_e = 0.0;
    const Real x_H2 = (const_cv) ? 0.0 : y_in(IH2);

    // energy per hydrogen atom
    const Real E_ergs = y_in(IIE) * units_energy_density_cgs / n_H;

    // Temperature
    return E_ergs / Thermo::CvCold(x_H2, x_He, x_e, gamma);
  }

  /*!
   * \brief Compute the cooling term from the temperature
   *
   * \param T The temperature in the cell
   * \return Real The cooling term, i.e. how much the energy decreases. Note
   * that this is a positive value so the energy update should look like `E =
   * HeatingTerm() - CoolingTerm();`
   */
  KOKKOS_FUNCTION
  Real CoolingTerm(Real const& T) const {
    return Thermo::alpha_GD_ * n_H * Kokkos::sqrt(T) * T;
  }

  /*!
   * \brief Compute the heating term from the temperature. It's just zero for
   * the H2 network and this function exists for API consistency across
   * networks.
   *
   * \param T The temperature in the cell
   * \return Real The heating term, i.e. how much the energy increases. The
   * energy update should look like `E = HeatingTerm() - CoolingTerm();`
   */
  KOKKOS_FUNCTION
  Real HeatingTerm(Real const& T) const { return 0.0; }

  /*!
   * \brief Evaluate the internal energy equation
   *
   * \param y_in The current state to evaluate the internal energy equation from
   * \return Real The result of evaluating the internal energy equation
   */
  template <class vec_type>
  KOKKOS_FUNCTION Real Edot(const vec_type& y_in,
                            const GhostSpecies& /*ghosts*/) const {
    if (isothermal) {
      return 0.0;
    }

    const Real T = Temperature(y_in);

    static constexpr Real T_floor = 1.0;  // temperature floor for cooling
    if (T < T_floor) {
      return 0;
    } else {
      const Real dEdt = HeatingTerm(T) - CoolingTerm(T);
      // convert to code units
      return units_time_cgs * (dEdt * n_H / units_energy_density_cgs);
    }
  }

  /*!
   * \brief Compute the creation and destruction rates. These are used like
   * `f(i) = rate.creation(i) - y(i) * rate.destruction(i);`
   *
   * \param y_in The current state to compute the rates from
   * \return CDRates_t A struct containing the creation and destruction rate
   * arrays.
   */
  template <class vec_type>
  KOKKOS_FUNCTION CDRates_t<neqs - 1> CDRates(
      const vec_type& y_in, const GhostSpecies& /*ghosts*/) const {
    CDRates_t<neqs - 1> rates;

    // cr = cosmic ray, gr = dust grain
    const Real rate_cr = k_cr;
    const Real rate_gr = k_gr * n_H;

    // H_2 equation
    rates.creation(IH2) = rate_gr * y_in(IH);
    // H equation
    rates.creation(IH) = 2 * rate_cr * y_in(IH2);

    // H_2 equation
    rates.destruction(IH2) = rate_cr;
    // H equation
    rates.destruction(IH) = 2 * rate_gr;

    // convert to code units
    for (size_t i = 0; i < neqs - 1; i++) {
      rates.creation(i) *= units_time_cgs;
      rates.destruction(i) *= units_time_cgs;
    }

    return rates;
  }

  /*!
   * \brief Setup the network for the next iteration of the ODE solver. The H2
   * network has nothing to set up; this exists for API consistency across
   * networks.
   *
   * \return GhostSpecies The (empty) ghost species
   */
  template <class vec_type>
  KOKKOS_FUNCTION GhostSpecies SetupNextStep(const vec_type& /*y_in*/) const {
    return GhostSpecies{};
  }

  /// The semi-implicit solver calls UpdateSpecies<K> for each species
  static constexpr bool custom_species_update = true;

  /*!
   * \brief One backward-Euler step of H2 for the semi-implicit solver, then H
   * from hydrogen conservation, x_H = x_H,tot - 2 x_H2. Updating H with its own
   * backward-Euler step would use H2 at a different time level and let
   * 2 x_H2 + x_H drift.
   */
  template <int K, class vec_type>
  KOKKOS_FUNCTION void UpdateSpecies(const vec_type& y, const Real* y_n,
                                     const GhostSpecies& ghosts,
                                     const Real h) const {
    static_assert(K == IH2, "H2 network updates only H2; H follows from it");
    const auto rates = CDRates(y, ghosts);
    const Real x_H_tot = 2.0 * y_n[IH2] + y_n[IH];
    y(IH2) = (y_n[IH2] + h * rates.creation(IH2)) /
             (1.0 + h * rates.destruction(IH2));
    y(IH) = Kokkos::fmax(x_H_tot - 2.0 * y(IH2), 0.0);
  }

  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION void evaluate_function(const Real /*t*/, const Real /*dt*/,
                                         const vec_type1& y_in,
                                         vec_type2& f) const {
    const GhostSpecies ghosts = SetupNextStep(y_in);

    // ----- Internal energy equation -----
    f(IIE) = Edot(y_in, ghosts);

    // ----- Creation & Destruction Rates -----
    const auto rates = CDRates(y_in, ghosts);

    // Compute the changes
    for (size_t i = 0; i < neqs - 1; i++) {
      f(i) = rates.creation(i) - y_in(i) * rates.destruction(i);
    }
  }

  template <class vec_type, class mat_type>
  KOKKOS_FUNCTION void evaluate_jacobian(const Real t, const Real dt,
                                         const vec_type& y_in,
                                         const mat_type& jac) const {
    // This analytic Jacobian was compute by an LLM. It passes the tests and
    // appears to be correct at first inspection but I cannot promise that it
    // does not contain a subtle error

    //         ∂/∂y(IH2)         ∂/∂y(IH)     ∂/∂y(IIE)
    // IH2  [    -a                 b              0      ]
    // IH   [    2a                -2b             0      ]
    // IIE  [ ∂Edot/∂y(IH2)         0        ∂Edot/∂y(IIE)]
    // with a = units_time_cgs·k_cr, b = units_time_cgs·k_gr·n_H

    const Real a = units_time_cgs * k_cr;        // rate_cr
    const Real b = units_time_cgs * k_gr * n_H;  // rate_gr

    // ----- H2 / H block (exactly linear) -----
    jac(IH2, IH2) = -a;
    jac(IH2, IH) = b;
    jac(IH2, IIE) = 0.0;

    jac(IH, IH2) = 2.0 * a;
    jac(IH, IH) = -2.0 * b;
    jac(IH, IIE) = 0.0;

    // ----- Energy row -----
    jac(IIE, IH2) = 0.0;
    jac(IIE, IH) = 0.0;
    jac(IIE, IIE) = 0.0;

    if (!isothermal) {
      static constexpr Real x_He = 0.1;
      const Real x_H2_raw = const_cv ? 0.0 : y_in(IH2);
      const Real x_H2 = Kokkos::fmin(Kokkos::fmax(x_H2_raw, 0.0), 0.5);

      const Real K = units_energy_density_cgs / n_H;
      const Real Cv = Thermo::CvCold(x_H2_raw, x_He, 0.0, gamma);
      const Real T = y_in(IIE) * K / Cv;

      static constexpr Real T_floor = 1.0;
      if (T >= T_floor) {
        const Real C = units_time_cgs * n_H * n_H * Thermo::alpha_GD_ /
                       units_energy_density_cgs;
        const Real dEdot_dT = -1.5 * C * Kokkos::sqrt(T);

        jac(IIE, IIE) = dEdot_dT * (K / Cv);

        // dCv/dx_H2 = -kB/(gamma-1), only in the unclamped interior
        const bool interior = !const_cv && (x_H2_raw > 0.0) && (x_H2_raw < 0.5);
        if (interior) {
          const Real kB = units::Units::k_boltzmann_cgs;
          const Real dCv_dxH2 = -kB / (gamma - 1.0);
          jac(IIE, IH2) = -dEdot_dT * (T / Cv) * dCv_dxH2;
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
};
}  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_H2_HPP_
