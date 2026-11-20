//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file equil_survey.cpp
//! \brief Equilibria of the NCR chemical network (tabulated or analytic) over a grid of
//!        n_H, Z_g, Z_d, chi (FUV), xi_CR and T, one cell per grid point, no mesh.
//!
//!   Usage: chemistry2_equil_survey -i <athinput> [block/key=value ...]. The
//!   <chemistry2> block sets the network as in a run (network = NCR or NCR_analytic);
//!   <problem> sets the grid. An axis is "min max points" in log10: log_nH, log_Zg,
//!   log_Zd ("tied": Z_d = Z_g), log_chi, log_xi ("tied": xi = xi_per_chi chi), log_T
//!   (the fixed T, or the start T of a full solve). mode = full (chemistry and T) or
//!   fixed_t (chemistry at fixed T). Output: a text table, one row per grid point.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "chemistry2/network/ncr/ncr.hpp"
#include "chemistry2/solvers/semi_implicit.hpp"
#include "chemistry2/util/code_units.hpp"
#include "units/units.hpp"

namespace {
constexpr Real GAMMA = 5.0/3.0;
constexpr Real U_TIME = 3.15576e13;  // code time: 1 Myr [s]
constexpr Real U_ENERGY = 1.0e-12;   // code energy density [erg cm^-3]

// one log10 axis: n points from lo to hi (one point at lo if n = 1)
struct Axis {
  Real lo = 0.0, hi = 0.0;
  int n = 1;
  KOKKOS_INLINE_FUNCTION Real At(const int i) const {
    const Real l = (n > 1) ? lo + (hi - lo)*i/(n - 1) : lo;
    return Kokkos::pow(10.0, l);
  }
};

Axis ReadAxis(ParameterInput *pin, const std::string &key, const std::string &dflt) {
  std::istringstream is(pin->GetOrAddString("problem", key, dflt));
  Axis a;
  is >> a.lo >> a.hi >> a.n;
  if (!is || a.n < 1) {
    std::cout << "### FATAL ERROR: <problem>/" << key << " must be 'min max points'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return a;
}

// the grid and the solver limits, from <problem>
struct Grid {
  Axis n, zg, zd, chi, xi, t;
  bool zd_tied, xi_tied, fixed_t;
  Real xi_per_chi;
  chemistry2::EquilibriumSettings q;
  std::string output;

  explicit Grid(ParameterInput *pin) {
    fixed_t = pin->GetOrAddString("problem", "mode", "full") == "fixed_t";
    n = ReadAxis(pin, "log_nH", "-2 3 161");
    zg = ReadAxis(pin, "log_Zg", "0 0 1");
    zd_tied = pin->GetOrAddString("problem", "log_Zd", "tied") == "tied";
    zd = zd_tied ? Axis{} : ReadAxis(pin, "log_Zd", "0 0 1");
    chi = ReadAxis(pin, "log_chi", "0 0 1");
    xi_tied = pin->GetOrAddString("problem", "log_xi", "tied") == "tied";
    xi = xi_tied ? Axis{} : ReadAxis(pin, "log_xi", "-15.7 -15.7 1");
    xi_per_chi = pin->GetOrAddReal("problem", "xi_per_chi", 2.0e-16);
    t = ReadAxis(pin, "log_T", "3 3 1");
    q = chemistry2::EquilibriumSettings{pin->GetOrAddReal("problem", "dt_max", 1.0e5),
                                        pin->GetOrAddInteger("problem", "nsub_max", 20000),
                                        pin->GetOrAddReal("problem", "tol", 1.0e-6),
                                        fixed_t};
    output = pin->GetOrAddString("problem", "output", "equil_survey.txt");
  }
};

// output columns
enum : int { C_NH, C_ZG, C_ZD, C_CHI, C_XI, C_T0, C_XH2, C_XHP, C_XE, C_XCP, C_XCO,
             C_XC, C_XO, C_T, C_POK, C_HEAT, C_COOL, C_TCOOL, C_NSUB, NCOL };
const char *const NAMES[NCOL] = {"nH", "Zg", "Zd", "chi", "xi_CR", "T_start", "x_H2",
                                 "x_H+", "x_e", "x_C+", "x_CO", "x_C", "x_O", "T",
                                 "P/k", "heating", "cooling", "t_cool", "nsub"};

//! Solve every grid point to equilibrium with network Net and write the table
template <class Net>
int Survey(ParameterInput *pin, const Grid &g) {
  using Solver = chemistry2::SemiImplicit<Net>;
  constexpr int NSPEC = Net::NSPEC;

  // the network as in a run, and code units with code density = n_H
  const Real mu_h = pin->GetOrAddReal("chemistry2", "mu_H", 1.4);
  const typename Net::Settings base = Net::ReadSettings(pin, "chemistry2");
  const chemistry2::SemiImplicitSettings si =
      chemistry2::SemiImplicitSettings::Read(pin, "chemistry2");
  const chemistry2::UnitFactors units = chemistry2::MakeUnitFactors(
      mu_h*units::Units::hydrogen_mass_cgs, U_TIME, U_ENERGY, mu_h);

  // grid points, n_H fastest; Z_d varies with Z_g unless it has its own axis
  const Axis ax_n = g.n, ax_zg = g.zg, ax_zd = g.zd, ax_chi = g.chi, ax_xi = g.xi,
             ax_t = g.t;
  const bool zd_tied = g.zd_tied, xi_tied = g.xi_tied;
  const Real xi_per_chi = g.xi_per_chi;
  const chemistry2::EquilibriumSettings q = g.q;
  const int n_zd = zd_tied ? 1 : ax_zd.n, n_xi = xi_tied ? 1 : ax_xi.n;
  const int npts = ax_n.n*ax_zg.n*n_zd*ax_chi.n*n_xi*ax_t.n;
  Kokkos::View<Real**, Kokkos::LayoutRight, DevExeSpace::memory_space> out("out", npts,
                                                                          NCOL);
  int not_converged = 0;
  Kokkos::parallel_reduce("equil_survey", Kokkos::RangePolicy<>(DevExeSpace(), 0, npts),
                          KOKKOS_LAMBDA(const int idx, int &nc) {
    int r = idx;
    const int i_n = r % ax_n.n; r /= ax_n.n;
    const int i_zg = r % ax_zg.n; r /= ax_zg.n;
    const int i_zd = r % n_zd; r /= n_zd;
    const int i_chi = r % ax_chi.n; r /= ax_chi.n;
    const int i_xi = r % n_xi; r /= n_xi;
    const int i_t = r;
    const Real n_h = ax_n.At(i_n), zg = ax_zg.At(i_zg);
    const Real zd = zd_tied ? zg : ax_zd.At(i_zd);
    const Real chi = ax_chi.At(i_chi);
    const Real xi = xi_tied ? xi_per_chi*chi : ax_xi.At(i_xi);
    const Real t0 = ax_t.At(i_t);

    // this point's metallicity in a copy of the settings, then atomic gas, 1% ionized
    typename Net::Settings s = base;
    s.zg = zg;
    s.zd = zd;
    const typename Net::Cell c = Net::MakeCell(n_h, GAMMA, s, units,
                                               chemistry2::CellInputs{chi, xi, 0.0});
    Real y[NSPEC] = {};
    y[Net::IH_PLUS] = 0.01;
    y[Net::IE] = 0.01;
    Real e = Net::Energy(y, t0, c);
    const int nsub = Solver::SolveToEquilibrium(si, c, q, y, e);
    if (nsub < 0) nc += 1;

    // the equilibrium: closures, temperature, pressure, heating and cooling per H
    const typename Net::Substep sub = Net::Prepare(y, e, c);
    const auto hc = Net::HeatingCooling(y, e, c, sub);
    const Real e_cgs = e*units.energy_cgs;
    const Real v[NCOL] = {n_h, zg, zd, chi, xi, t0, y[Net::IH2], y[Net::IH_PLUS],
                          y[Net::IE], sub.x_cii, sub.x_co, sub.x_ci, sub.x_oi,
                          Net::Temperature(y, e, c),
                          (GAMMA - 1.0)*e_cgs/units::Units::k_boltzmann_cgs,
                          hc.heating, hc.cooling,
                          e_cgs/(n_h*hc.cooling + 1e-300)/U_TIME,
                          static_cast<Real>(nsub)};
    for (int k = 0; k < NCOL; ++k) out(idx, k) = v[k];
  }, Kokkos::Sum<int>(not_converged));

  // one row per grid point
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  std::FILE *f = std::fopen(g.output.c_str(), "w");
  if (f == nullptr) {
    std::cout << "### FATAL ERROR: cannot write " << g.output << std::endl;
    return 1;
  }
  std::fprintf(f, "# %s equilibrium survey, mode %s; heating and cooling [erg s^-1 "
               "per H], P/k [K cm^-3], t_cool [Myr], nsub < 0: not converged\n#",
               std::string(Net::NAME).c_str(), g.fixed_t ? "fixed_t" : "full");
  for (int k = 0; k < NCOL; ++k) std::fprintf(f, " %s", NAMES[k]);
  std::fprintf(f, "\n");
  for (int i = 0; i < npts; ++i) {
    for (int k = 0; k < NCOL; ++k) {
      std::fprintf(f, k == 0 ? "%.6e" : " %.6e", h(i, k));
    }
    std::fprintf(f, "\n");
  }
  std::fclose(f);
  std::printf("equil_survey: %s, %d points, %d not converged, written to %s\n",
              std::string(Net::NAME).c_str(), npts, not_converged, g.output.c_str());
  return 0;
}
}  // namespace

int main(int argc, char *argv[]) {
  global_variable::my_rank = 0;
  global_variable::nranks = 1;
  std::string infile;
  for (int a = 1; a < argc - 1; ++a) {
    if (std::string(argv[a]) == "-i") infile = argv[a + 1];
  }
  if (infile.empty()) {
    std::cout << "usage: " << argv[0] << " -i <athinput> [block/key=value ...]"
              << std::endl;
    return 1;
  }
  Kokkos::ScopeGuard guard(argc, argv);
  ParameterInput pin(infile);
  pin.ModifyFromCmdline(argc, argv);
  const Grid g(&pin);
  const std::string net = pin.GetString("chemistry2", "network");
  if (net == chemistry2::NCR::NAME) return Survey<chemistry2::NCR>(&pin, g);
  if (net == chemistry2::NCRAnalytic::NAME) return Survey<chemistry2::NCRAnalytic>(&pin, g);
  std::cout << "### FATAL ERROR: <chemistry2>/network must be NCR or NCR_analytic, not "
            << net << std::endl;
  return 1;
}
