#ifndef CHEMISTRY2_SOLVERS_KOKKOS_BDF_HPP_
#define CHEMISTRY2_SOLVERS_KOKKOS_BDF_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kokkos_bdf.hpp
//! \brief The Kokkos Kernels BDF solver on a chemistry2 chemical network: right-hand
//!        side f_K = c_K - d_K y_K from Rates<K> and Edot, Jacobian by finite
//!        differences of it. A reference solver for converged test solutions.

#include <KokkosODE_BDF.hpp>
#include <string>       // NOLINT(build/include_order)
#include <string_view>  // NOLINT(build/include_order)
#include <utility>      // NOLINT(build/include_order)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "chemistry2/network/network_config.hpp"

namespace chemistry2 {

//! <chemistry2> kokkos_BDF_* keys
struct KokkosBDFSettings {
  Real first_step_frac;  // first internal step / dt; 0 lets the solver choose
  Real rtol, atol;

  static KokkosBDFSettings Read(ParameterInput *pin, const std::string &block) {
    return KokkosBDFSettings{pin->GetOrAddReal(block, "kokkos_BDF_first_step_frac", 0.0),
                             pin->GetOrAddReal(block, "kokkos_BDF_rtol", 1.0e-3),
                             pin->GetOrAddReal(block, "kokkos_BDF_atol", 1.0e-6)};
  }
};

//! Adaptive BDF from Kokkos Kernels on the species and energy of one cell
template <class Network>
class KokkosBDF {
 public:
  static constexpr std::string_view NAME = "kokkos_BDF";  // <chemistry2>/ode_solver
  using Settings = KokkosBDFSettings;
  using Cell = typename Network::Cell;
  static constexpr int NSPEC = Network::NSPEC;

  //! The ODE system in the form KokkosODE::Experimental::BDFSolve calls: species, then e
  struct System {
    static constexpr int neqs = NSPEC + 1;
    const Cell &cell;

    template <class V1, class V2>
    KOKKOS_FUNCTION void evaluate_function(const Real, const Real, const V1 &y_in,
                                           V2 &f) const {
      Real y[NSPEC];  // NOLINT(runtime/arrays)
      for (int k = 0; k < NSPEC; ++k) y[k] = Kokkos::fmax(y_in(k), 0.0);
      const Real e = Kokkos::fmax(y_in(NSPEC), 0.0);
      const typename Network::Substep sub = Network::Prepare(y, e, cell);
      Fill(y, e, sub, y_in, f, std::make_integer_sequence<int, NSPEC>{});
      f(NSPEC) = Network::Edot(y, e, cell, sub);
    }

    template <class V, class M>
    KOKKOS_FUNCTION void evaluate_jacobian(const Real t, const Real dt, const V &y_in,
                                           const M &jac) const {
      Real f0b[neqs], fpb[neqs];  // NOLINT(runtime/arrays)
      Kokkos::View<Real*, Kokkos::AnonymousSpace, Kokkos::MemoryUnmanaged> f0(f0b, neqs),
          fp(fpb, neqs);
      evaluate_function(t, dt, y_in, f0);
      const Real eps = Kokkos::sqrt(Kokkos::Experimental::epsilon_v<Real>);
      for (int j = 0; j < neqs; ++j) {
        const Real y0 = y_in(j);
        const Real h = eps*Kokkos::fmax(Kokkos::fabs(y0), 1.0e-6);
        y_in(j) = y0 + h;
        evaluate_function(t, dt, y_in, fp);
        y_in(j) = y0;
        const Real inv_h = 1.0/h;
        for (int k = 0; k < neqs; ++k) jac(k, j) = (fp(k) - f0(k))*inv_h;
      }
    }

   private:
    template <class V1, class V2, int... K>
    KOKKOS_INLINE_FUNCTION void Fill(const Real *y, const Real e,
                                     const typename Network::Substep &sub, const V1 &y_in,
                                     V2 &f, std::integer_sequence<int, K...>) const {
      auto one = [&](auto k) {
        constexpr int KK = decltype(k)::value;
        const CD r = Network::template Rates<KK>(y, e, cell, sub);
        f(KK) = r.c - r.d*y_in(KK);
      };
      (one(std::integral_constant<int, K>{}), ...);
    }
  };

  //! Integrate y and e over [t_start, t_start + dt]; returns 1 (the step count is the
  //! solver's own)
  KOKKOS_INLINE_FUNCTION static int Solve(const Settings &s, const Cell &cell,
                                          const Real t_start, const Real dt, Real *y,
                                          Real &e) {
    constexpr int N = NSPEC + 1;
    Real yb[N], ynb[N];  // NOLINT(runtime/arrays)
    Real tmpb[N][23 + 2*N + 4], tmp2b[6][7];
    using V1 = Kokkos::View<Real*, Kokkos::AnonymousSpace, Kokkos::MemoryUnmanaged>;
    using V2 = Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::AnonymousSpace,
                            Kokkos::MemoryUnmanaged>;
    for (int k = 0; k < NSPEC; ++k) yb[k] = y[k];
    yb[NSPEC] = e;
    V1 y0(yb, N), y1(ynb, N);
    V2 tmp(&tmpb[0][0], N, 23 + 2*N + 4), tmp2(&tmp2b[0][0], 6, 7);
    const System sys{cell};
    const auto status = KokkosODE::Experimental::BDFSolve(
        sys, t_start, t_start + dt, s.first_step_frac*dt, dt, y0, y1, tmp, tmp2, s.rtol,
        s.atol);
    if (status != KokkosODE::Experimental::ode_solver_status::SUCCESS) {
      Kokkos::printf("chemistry2 KokkosBDF: status %d at t_start=%.17g dt=%.17g\n",
                     static_cast<int>(status), t_start, dt);
      Kokkos::abort("chemistry2 KokkosBDF: BDF solve failed");
    }
    for (int k = 0; k < NSPEC; ++k) y[k] = Kokkos::fmax(y0(k), 0.0);
    e = Kokkos::fmax(y0(NSPEC), 0.0);
    return 1;
  }
};

}  // namespace chemistry2

#endif  // CHEMISTRY2_SOLVERS_KOKKOS_BDF_HPP_
