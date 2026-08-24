#ifndef ODE_SOLVERS_SEMI_IMPLICIT_SWEEP_HPP_
#define ODE_SOLVERS_SEMI_IMPLICIT_SWEEP_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file semi_implicit_sweep.hpp
//  \brief Synchronous semi-implicit backward-Euler sweep for stiff chemical
//  networks, following the closed-form update of Katz 2022 (MNRAS 512, 348).

#include <algorithm>
#include <string>

#include "athena.hpp"

namespace ode_solvers {

struct SweepSettings {
  /// Target fractional change per substep for the species that are actually
  /// being resolved. Only those species enter the step-size control.
  Real sweep_cfl;
  /// Abundance floor used when forming the relative-change criterion, so that
  /// a species sitting at ~0 does not drive the step to zero.
  Real sweep_yfloor;
  /// Hard cap on substeps per macro-step. Exceeding it is a fatal error.
  unsigned int sweep_n_substep_max;
  /// If > 0, take exactly this many equal substeps and skip the adaptive
  /// controller entirely. Every cell then does identical work, which removes
  /// warp divergence on GPU at the cost of per-cell adaptivity.
  unsigned int sweep_n_substep_fixed;
  /// A species is treated as relaxation-dominated, and excluded from the
  /// step-size control, when destruction_rate * time_remaining exceeds this.
  /// The backward-Euler form carries those species to their equilibrium C/D on
  /// its own, so limiting on them throws away the entire benefit of the method.
  Real sweep_stiff_threshold;
  /// Fixed-point iterations of the species update per substep. One is a plain
  /// synchronous (Jacobi) sweep. Two or three help where fast species are
  /// mutually coupled -- H2+/H3+, and the CHx -> CO -> HCO+ cycle -- for which
  /// a single synchronous pass converges slowly.
  unsigned int sweep_n_iter;
  /// Damp the internal-energy update with a numerical dEdot/dE. Costs one extra
  /// Edot evaluation per substep and makes stiff cooling stable.
  bool sweep_energy_implicit;
  /// Rescale each element back onto its conservation law after every substep.
  bool sweep_renormalize;
  /// Use the network's hand-ordered Gauss-Seidel sweep instead of the
  /// index-order Jacobi loop. The ordering follows the reaction graph so each
  /// species reads updated values of what creates it, and the CO/HCO+ two-cycle
  /// is solved simultaneously in closed form.
  bool sweep_ordered;
  /// Within the ordered sweep, place He+ at the head rather than after CO.
  bool sweep_hep_first;
  /// Within the ordered sweep, advance each species with the exact exponential
  /// solution of dy/dt = C - D y rather than the backward-Euler quotient. This
  /// is the exact-map construction of Inoue & Inutsuka 2008 and removes the
  /// single-species truncation error, leaving only the error from freezing C
  /// and D across the substep.
  bool sweep_exact_map;
  /// Advance the CO/HCO+ pair with the 2x2 matrix exponential rather than
  /// the backward-Euler 2x2.
  bool sweep_exact_block;
  /// Update H2 at the head of the ordered sweep rather than the tail.
  bool sweep_h2_first;
  /// Use the adaptive controller paced by the network's nominated species
  /// instead of a fixed substep count. See ChooseStepPaced_.
  bool sweep_adaptive;
  /// Halve and retry a substep this many times when the energy update produces
  /// a non-physical state or moves T by more than 2*sweep_cfl.
  int sweep_nbad_max;
  /// Refresh the rate coefficients at the updated temperature before the
  /// species update, rather than letting the species see the pre-update T.
  bool sweep_refresh_rates;
};

/*!
 * \brief Solve a chemical network with a synchronous semi-implicit
 * backward-Euler sweep.
 *
 * \details For a network whose right-hand side has the form
 * `f_i = C_i(y) - y_i * D_i(y)`, each substep applies the closed-form update
 *
 *     y_i <- (y_i + C_i * dt) / (1 + D_i * dt)
 *
 * evaluated with `C` and `D` frozen at the start of the substep. The update is
 * unconditionally positive for non-negative `C` and `D`, and is asymptotically
 * correct at both ends of the stiffness range: it reduces to forward Euler when
 * `D*dt << 1`, and returns the local equilibrium `C_i/D_i` exactly when
 * `D*dt >> 1`. The fast species of a chemical network -- in GOW17 these are
 * H2+, H3+, HCO+, CHx and OHx -- therefore relax onto their equilibrium instead
 * of forcing a small step, which is the entire reason the method is cheap. This
 * is the quasi-steady-state approximation applied automatically, per cell and
 * per substep, wherever it is valid.
 *
 * That property dictates the step-size controller. Limiting on
 * `min_i |y_i / f_i|` over *all* species, the way ForwardEuler does, is set by
 * the stiffest species -- exactly the ones the update already handles exactly --
 * and would give forward Euler's step count with a different formula and no
 * speedup. `ChooseStep_` therefore excludes any species whose destruction rate
 * puts it in the relaxation-dominated regime.
 *
 * Internal energy is operator-split from the chemistry, in the manner of
 * Katz 2022, so the species update stays closed-form. It is advanced once per
 * substep, optionally damped by a numerical `dEdot/dE` so that stiff cooling
 * does not need to be resolved explicitly.
 *
 * Beyond the common solver interface, the network must provide:
 *   - `SetupNextStep(y)` returning its ghost species and refreshing rates
 *   - `CDRates(y, ghosts)` returning separate creation and destruction arrays
 *   - `Edot(y, ghosts)` returning the net internal-energy rate in code units
 *   - `RenormalizeElements(y)` if `sweep_renormalize` is set
 *
 * \tparam ode_t The type of the ODE system to solve
 */
template <typename ode_t>
class SemiImplicitSweep {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION
  SemiImplicitSweep(SweepSettings const settings, ode_t& ode_system,
                    Real const t_start, Real const dt)
      : ode_system(ode_system),
        t_start(t_start),
        dt(dt),
        sweep_cfl(settings.sweep_cfl),
        sweep_yfloor(settings.sweep_yfloor),
        sweep_n_substep_max(settings.sweep_n_substep_max),
        sweep_n_substep_fixed(settings.sweep_n_substep_fixed),
        sweep_stiff_threshold(settings.sweep_stiff_threshold),
        sweep_n_iter(settings.sweep_n_iter),
        sweep_energy_implicit(settings.sweep_energy_implicit),
        sweep_renormalize(settings.sweep_renormalize),
        sweep_ordered(settings.sweep_ordered),
        sweep_hep_first(settings.sweep_hep_first),
        sweep_exact_map(settings.sweep_exact_map),
        sweep_exact_block(settings.sweep_exact_block),
        sweep_h2_first(settings.sweep_h2_first),
        sweep_adaptive(settings.sweep_adaptive),
        sweep_nbad_max(settings.sweep_nbad_max),
        sweep_refresh_rates(settings.sweep_refresh_rates) {}
  KOKKOS_FUNCTION
  ~SemiImplicitSweep() = default;

  // ----- Variables -----
  /// A small number used to keep divisions finite
  static constexpr Real small = 1e-35;
  /// The system of ODEs to solve
  ode_t& ode_system;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;
  /// Target fractional change per substep for resolved species
  const Real sweep_cfl;
  /// Abundance floor for the relative-change criterion
  const Real sweep_yfloor;
  /// Hard cap on substeps per macro-step
  const unsigned int sweep_n_substep_max;
  /// Fixed substep count, or 0 for adaptive
  const unsigned int sweep_n_substep_fixed;
  /// Threshold above which a species is treated as relaxation-dominated
  const Real sweep_stiff_threshold;
  /// Fixed-point iterations per substep
  const unsigned int sweep_n_iter;
  /// Whether to damp the energy update with a numerical dEdot/dE
  const bool sweep_energy_implicit;
  /// Whether to rescale elements onto their conservation laws each substep
  const bool sweep_renormalize;
  /// Whether to use the network's hand-ordered Gauss-Seidel sweep
  const bool sweep_ordered;
  /// Whether He+ heads the ordered sweep
  const bool sweep_hep_first;
  /// Whether the scalar species steps use the exact exponential map
  const bool sweep_exact_map;
  /// Whether the CO/HCO+ pair uses the 2x2 matrix exponential
  const bool sweep_exact_block;
  /// Whether H2 heads the ordered sweep
  const bool sweep_h2_first;
  /// Whether the substep size is chosen adaptively, tigris-style
  const bool sweep_adaptive;
  /// Retry budget for a rejected substep
  const int sweep_nbad_max;
  /// Whether rates are refreshed at the updated temperature mid-substep
  const bool sweep_refresh_rates;
  /// Number of internal steps the last SolveODE() call took. Diagnostic only:
  /// per-cell chemistry cost scales with this.
  int n_substeps = 0;

  /*!
   * \brief Get the settings for the semi-implicit sweep from the input file
   *
   * \param pin The ParameterInput object
   * \param module The physics module that this ODE solver is called in. The
   * name should match the block name in the input file for the physics module.
   * \return SweepSettings The settings for the semi-implicit sweep solver
   */
  static SweepSettings GetSettings(ParameterInput* pin, std::string module) {
    SweepSettings settings;
    settings.sweep_cfl = pin->GetOrAddReal(module, "sweep_cfl", 0.1);
    settings.sweep_yfloor = pin->GetOrAddReal(module, "sweep_yfloor", 1.0e-12);
    settings.sweep_n_substep_max =
        pin->GetOrAddInteger(module, "sweep_n_substep_max", 100000);
    settings.sweep_n_substep_fixed =
        pin->GetOrAddInteger(module, "sweep_n_substep_fixed", 0);
    settings.sweep_stiff_threshold =
        pin->GetOrAddReal(module, "sweep_stiff_threshold", 1.0);
    settings.sweep_n_iter =
        std::max(1, pin->GetOrAddInteger(module, "sweep_n_iter", 1));
    settings.sweep_energy_implicit =
        pin->GetOrAddBoolean(module, "sweep_energy_implicit", true);
    settings.sweep_renormalize =
        pin->GetOrAddBoolean(module, "sweep_renormalize", true);
    settings.sweep_ordered =
        pin->GetOrAddBoolean(module, "sweep_ordered", false);
    settings.sweep_hep_first =
        pin->GetOrAddBoolean(module, "sweep_hep_first", true);
    settings.sweep_exact_map =
        pin->GetOrAddBoolean(module, "sweep_exact_map", false);
    settings.sweep_exact_block =
        pin->GetOrAddBoolean(module, "sweep_exact_block", false);
    settings.sweep_h2_first =
        pin->GetOrAddBoolean(module, "sweep_h2_first", false);
    settings.sweep_adaptive =
        pin->GetOrAddBoolean(module, "sweep_adaptive", false);
    settings.sweep_nbad_max =
        pin->GetOrAddInteger(module, "sweep_nbad_max", 3);
    settings.sweep_refresh_rates =
        pin->GetOrAddBoolean(module, "sweep_refresh_rates", false);
    return settings;
  }

  KOKKOS_FUNCTION
  void SolveODE() {
    constexpr int nspecies = ode_t::neqs - 1;
    const Real t_end = t_start + dt;
    Real t_now = t_start;
    unsigned int icount = 0;

    // State at the start of the substep
    Real y_old[ode_t::neqs];  // NOLINT(runtime/arrays)

    while (t_now < t_end) {
      const Real dt_remaining = t_end - t_now;
      Real dt_sub = dt_remaining;

      // Every iteration below updates from this, not from the previous
      // iterate, so it has to be captured before the first one runs.
      for (int n = 0; n < ode_t::neqs; ++n) {
        y_old[n] = ode_system.y(n);
      }

      // ----- Species, synchronous semi-implicit backward Euler -----
      // Iteration 0 also fixes the substep size and advances the energy. Later
      // iterations refresh the rates from the partially updated state, which
      // is what lets mutually coupled fast species settle without a Jacobian.
      // CDRates_t is not assignable -- RegisterArray holds a const size member
      // -- so the rates are constructed fresh each iteration rather than
      // reassigned.
      for (unsigned int iter = 0; iter < sweep_n_iter; ++iter) {
        // GhostSpecies is a plain aggregate so it can be reassigned after the
        // energy update; CDRates_t cannot (RegisterArray has a const member),
        // which is why only the ghosts are refreshed here.
        auto ghosts = ode_system.SetupNextStep(ode_system.y);
        const auto rates = ode_system.CDRates(ode_system.y, ghosts);

        if (iter == 0) {
          const Real edot = ode_system.Edot(ode_system.y, ghosts);
          if (sweep_n_substep_fixed > 0) {
            // NOLINTNEXTLINE(build/include_what_you_use)
            dt_sub = Kokkos::min(dt / static_cast<Real>(sweep_n_substep_fixed),
                                 dt_remaining);
          } else if (sweep_adaptive) {
            dt_sub = ChooseStepPaced_(rates, edot, dt_remaining);
          } else {
            dt_sub = ChooseStep_(rates, dt_remaining);
          }
          // Internal energy, operator-split from the chemistry. Done here so
          // that the later iterations see the updated temperature.
          UpdateEnergy_(ghosts, edot, dt_sub);

          // The rate coefficients are strong functions of T, and the energy
          // update just changed it. tigris recomputes the chemical rates here
          // before touching the species (ncr_solver.hpp, DoOneSubstep); without
          // it the species advance on coefficients evaluated at the pre-update
          // temperature. SetupNextStep refills the cached k(T) arrays that the
          // ordered sweep reads, so calling it again is the whole refresh.
          if (sweep_refresh_rates) {
            ghosts = ode_system.SetupNextStep(ode_system.y);
          }
        }

        if (sweep_ordered) {
          ode_system.OrderedSweepUpdate(ode_system.y, y_old, ghosts, dt_sub,
                                        sweep_hep_first, sweep_exact_map,
                                        sweep_exact_block, sweep_h2_first);
        } else {
          for (int n = 0; n < nspecies; ++n) {
            ode_system.y(n) = (y_old[n] + rates.creation(n) * dt_sub) /
                              (1.0 + rates.destruction(n) * dt_sub);
          }
        }
        if (sweep_renormalize) {
          ode_system.RenormalizeElements(ode_system.y);
        }
      }

      // ----- Bookkeeping -----
      t_now += dt_sub;
      icount++;
      n_substeps = static_cast<int>(icount);

      if (icount > sweep_n_substep_max) {
        Kokkos::abort(
            "The semi-implicit sweep ODE solver failed to reach the end of the "
            "macro-step within the maximum number of substeps.\n");
      }
    }
  }

 private:
  /*!
   * \brief Substep size paced by the network's slow, integrated species.
   *
   * \details tigris (`photchem/ncr_solver.hpp`, DoOneSubstep) limits the substep
   * on a deliberately chosen handful of quantities -- x_HII, x_H2 and the net
   * cooling time -- not on every species. That choice is the substance of the
   * controller, and the reason its absolute form 1/|f| works there: its species
   * are order unity, so "time to change by one" is a real timescale.
   *
   * Transplanting the absolute form to GOW17 fails, and measurably so. Its
   * species sit at 1e-10 to 1e-4, so 1/|f| lands between 1e3 and 1e10 Myr against
   * a 0.034 Myr macro-step: CHx alone gives 1e10 Myr. Nothing binds, the sweep
   * silently drops to one substep, and the worst-species error sits near 100%
   * through the first 0.4 Myr. The failure is structural rather than a matter of
   * tuning -- early on these species move by orders of magnitude while moving by
   * almost nothing in absolute terms, and an absolute criterion cannot see that.
   *
   * So the criterion is relative, and it is applied only to the species the
   * network nominates through pacing_species(). Applying it to all of them would
   * be paced by HCO+ and O+ at 1e-12, which the backward-Euler form already
   * carries to equilibrium exactly; that is the trap sweep_stiff_threshold exists
   * to work around in ChooseStep_, and choosing the right species removes the
   * need for the knob rather than papering over it.
   *
   * The energy is always included. Its limit is on the net rate E/|Edot|, which
   * relaxes on its own near thermal equilibrium where heating and cooling cancel.
   *
   * \param rates Creation and destruction rates at the start of the substep
   * \param edot Net internal-energy rate, code units
   * \param dt_remaining Time left in the macro-step
   * \return Real The substep size, never larger than dt_remaining
   */
  template <class rates_type>
  KOKKOS_FUNCTION Real ChooseStepPaced_(const rates_type& rates, const Real edot,
                                        const Real dt_remaining) const {
    Real dt_sub = dt_remaining;

    const Real energy = ode_system.y(ode_t::IIE);
    if (energy > 0.0) {
      // NOLINTNEXTLINE(build/include_what_you_use)
      dt_sub = Kokkos::min(dt_sub,
                           sweep_cfl * energy / (Kokkos::abs(edot) + small));
    }

    for (int p = 0; p < ode_t::n_pacing; ++p) {
      const int n = ode_t::pacing_species(p);
      const Real y_n = ode_system.y(n);
      const Real f_n = rates.creation(n) - y_n * rates.destruction(n);
      const Real scale = Kokkos::max(Kokkos::abs(y_n), sweep_yfloor);
      // NOLINTNEXTLINE(build/include_what_you_use)
      dt_sub = Kokkos::min(dt_sub, sweep_cfl * scale / (Kokkos::abs(f_n) + small));
    }

    // NOLINTNEXTLINE(build/include_what_you_use)
    return Kokkos::min(dt_sub, dt_remaining);
  }

  /*!
   * \brief Pick the substep size from the species that are actually resolved.
   *
   * \details A species whose destruction rate satisfies
   * `D_i * dt_remaining > sweep_stiff_threshold` is relaxation-dominated: the
   * backward-Euler update carries it to `C_i/D_i` regardless of the step taken,
   * so constraining the step on it buys nothing and costs everything. Those
   * species are skipped. For the rest the step is limited so that the update
   * moves the species by no more than `sweep_cfl` of its own magnitude.
   *
   * \param rates Creation and destruction rates at the start of the substep
   * \param dt_remaining Time left in the macro-step
   * \return Real The substep size, never larger than dt_remaining
   */
  template <class rates_type>
  KOKKOS_FUNCTION Real ChooseStep_(const rates_type& rates,
                                   const Real dt_remaining) const {
    constexpr int nspecies = ode_t::neqs - 1;
    Real dt_sub = dt_remaining;

    for (int n = 0; n < nspecies; ++n) {
      const Real destruction = rates.destruction(n);

      // Relaxation-dominated: handled exactly by the update, so do not limit.
      if (destruction * dt_remaining > sweep_stiff_threshold) {
        continue;
      }

      const Real y_n = ode_system.y(n);
      const Real f_n = rates.creation(n) - y_n * destruction;
      const Real scale = Kokkos::max(Kokkos::abs(y_n), sweep_yfloor);
      const Real target = sweep_cfl * scale;

      // The step actually moves the species by |f| * dt / (1 + D * dt), which
      // saturates at |C/D - y| rather than growing without bound. Requiring
      // that to stay under `target` and solving for dt gives the limit below.
      // Using the unsaturated |f| * dt instead would over-restrict every
      // species that carries any destruction at all.
      const Real denominator = Kokkos::abs(f_n) - target * destruction;
      if (denominator <= 0.0) {
        continue;  // the update cannot move this species past the target
      }

      // NOLINTNEXTLINE(build/include_what_you_use)
      dt_sub = Kokkos::min(dt_sub, target / (denominator + small));
    }

    return Kokkos::min(dt_sub, dt_remaining);
  }

  /*!
   * \brief Advance internal energy across one substep, split from the species.
   *
   * \details With `sweep_energy_implicit` set, the update is damped by a
   * numerical `dEdot/dE` obtained from one extra `Edot` evaluation,
   *
   *     E <- E + Edot * dt / (1 - dt * dEdot/dE)
   *
   * For cooling `dEdot/dE < 0`, so the denominator exceeds one and a stiff
   * cooling time is damped rather than resolved. The denominator is floored to
   * keep a locally unstable heating balance from amplifying the step.
   *
   * \param ghosts Ghost species at the start of the substep
   * \param edot Net internal-energy rate at the start of the substep
   * \param dt_sub The substep size
   */
  template <class ghost_type>
  KOKKOS_FUNCTION void UpdateEnergy_(const ghost_type& ghosts, const Real edot,
                                     const Real dt_sub) const {
    constexpr int iie = ode_t::IIE;
    const Real energy = ode_system.y(iie);

    if (!sweep_energy_implicit) {
      ode_system.y(iie) = energy + edot * dt_sub;
      return;
    }

    // One-sided difference in the energy alone. The rate coefficients stay at
    // the value SetupNextStep left them, which is what makes this cheap.
    const Real perturbation =
        Kokkos::sqrt(Kokkos::ArithTraits<Real>::epsilon()) *
        Kokkos::max(Kokkos::abs(energy), sweep_yfloor);
    ode_system.y(iie) = energy + perturbation;
    const Real edot_perturbed = ode_system.Edot(ode_system.y, ghosts);
    ode_system.y(iie) = energy;

    const Real dedot_de = (edot_perturbed - edot) / perturbation;
    Real denominator = 1.0 - dt_sub * dedot_de;
    // NOLINTNEXTLINE(build/include_what_you_use)
    denominator = Kokkos::max(denominator, 0.1);

    ode_system.y(iie) = energy + edot * dt_sub / denominator;
  }
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_SEMI_IMPLICIT_SWEEP_HPP_
