//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiative_snr.cpp
//! \brief One supernova in a uniform medium with the chemistry2 module (NCR or GOW17),
//!        ported from Tigris (src/pgen/radiative_snr.cpp, the run without particles).
//!
//!        The ambient medium is uniform at rest (n_H nH_0, pressure pgas_0 in k_B K
//!        cm^-3), with the chemical network's ambient abundances, advanced by the
//!        chemistry alone for n_equilibrate steps of dt_equilibrate so that chemistry and
//!        temperature settle. The supernova is injected once at t = 0 at the origin by
//!        the Tigris Subcell scheme (src/feedback/subcell_scheme.cpp, injection.cpp):
//!        N_sc^3 subcells per cell out to rfb_res cells, a linear taper of width half a
//!        subcell at the edge, mass M_ej and energy E_SN spread over the same weighted
//!        subcell volume so that both are exact on the grid, the kinetic fraction rising
//!        linearly with radius, and the radial momentum of each subcell solved so that
//!        the kinetic energy rises by exactly its share. The ejecta abundances replace
//!        the chemistry in the cells that receive mass (ionized hydrogen, and for GOW17
//!        He+ and the C, O and Si ions at their totals).
//!
//!        History (<problem>/user_hist = true): total mass, thermal energy and radial
//!        momentum; and mass, volume, thermal energy and radial momentum of the hot gas
//!        (T > 2e4 K) and of the shell (T < 2e4 K moving out faster than 0.1 km/s), with
//!        the shell's mass-weighted radius (divide by its mass). T comes from the
//!        pressure with n = n_H (1 + x_He - x_H2 + x_e), x_He = 0.1.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"
#include "chemistry2/chemistry2.hpp"
#include "chemistry2/network/ncr_rates.hpp"
#include "chemistry2/registry.hpp"

namespace {

constexpr Real T_HOT = 2.0e4;  // [K]
constexpr Real VR_SHELL = 0.1;  // [km/s]
constexpr Real X_HE = 0.1;

// species indices and charges of the active chemical network, for the history
struct SpeciesInfo {
  int nspec = 0, i_h2 = -1, i_e = -1;
  std::vector<int> charge;
  std::vector<std::string> name;
};
SpeciesInfo spec;
int first_scalar = 0;
Real mu_h = 1.4;

// look up the species table of the network/solver pair selected in <chemistry2>
SpeciesInfo GetSpecies(ParameterInput *pin) {
  const std::string net = pin->GetString("chemistry2", "network");
  const std::string solver = pin->GetOrAddString("chemistry2", "ode_solver",
                                                 "semi_implicit");
  SpeciesInfo s;
  chemistry2::Dispatch(chemistry2::Find(net, solver), [&](auto pair) {
    using N = typename decltype(pair)::Network;
    s.nspec = N::NSPEC;
    for (int n = 0; n < N::NSPEC; ++n) {
      s.name.emplace_back(N::rows[n].name);
      s.charge.push_back(N::rows[n].charge);
      if (N::rows[n].name == "H2") s.i_h2 = n;
      if (N::rows[n].name == "e") s.i_e = n;
    }
  });
  return s;
}

//! Subcell centres out to rmax, mirrored into the 8 octants (Tigris radial_loop: each
//! loop runs while the radius with the inner coordinates at their first centre is
//! below rmax)
template <typename F>
void RadialLoop(const Real rmax, const Real dx, const F &f) {
  const Real c0 = 0.5*dx;
  for (Real xs = c0; std::sqrt(xs*xs + 2.0*c0*c0) < rmax; xs += dx) {
    for (Real ys = c0; std::sqrt(xs*xs + ys*ys + c0*c0) < rmax; ys += dx) {
      for (Real zs = c0; std::sqrt(xs*xs + ys*ys + zs*zs) < rmax; zs += dx) {
        const Real r = std::sqrt(xs*xs + ys*ys + zs*zs);
        for (int sx = -1; sx <= 1; sx += 2) {
          for (int sy = -1; sy <= 1; sy += 2) {
            for (int sz = -1; sz <= 1; sz += 2) f(sx*xs, sy*ys, sz*zs, r);
          }
        }
      }
    }
  }
}

//! Weight of a subcell at r (in units of the feedback radius) near the edge
Real BoundaryWeight(const Real r, const Real dx_space) {
  if (r < 1.0 - dx_space) return 1.0;
  if (r < 1.0 + dx_space) return 0.5*(1.0 + (1.0 - r)/dx_space);
  return 0.0;
}

void SNRHistory(HistoryData *pdata, Mesh *pm);

// diagnostic (diag_chem_steps > 0): at cycle diag_after_cycles (0: right after the
// injection) advance the chemistry and energy diag_chem_steps times by diag_chem_dt with
// the hydro frozen, print the energy loss and the cells of largest loss, then exit
int diag_steps = 0, diag_cycle = 0;
Real diag_dt = 1.0e-4;

void DiagChem(MeshBlockPack *pmbp, const char *when) {
  auto &w0 = pmbp->phydro->w0;
  const auto &indcs = pmbp->pmesh->mb_indcs;
  const int nmb = pmbp->nmb_thispack, s0 = first_scalar, nspec = spec.nspec;
  auto &mbsize = pmbp->pmb->mb_size;
  const Real vol_c = mbsize.h_view(0).dx1*mbsize.h_view(0).dx2*mbsize.h_view(0).dx3;
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  const Real e_cgs = pmbp->punit->energy_cgs(), t_cgs = pmbp->punit->time_cgs();
  const Real pk = pmbp->punit->pressure_cgs()/units::Units::k_boltzmann_cgs;
  const Real nh_per_rho = pmbp->punit->density_cgs()/
                          (mu_h*units::Units::hydrogen_mass_cgs);
  // create_mirror (not _view) so the copy never aliases w0 on a host backend
  auto w_old = Kokkos::create_mirror(HostMemSpace(), w0);
  Kokkos::deep_copy(w_old, w0);
  std::printf("diag_chem %s (cycle %d, t %.6e)\n", when, pmbp->pmesh->ncycle,
              pmbp->pmesh->time);
  {
    auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->phydro->u0);
    Real eu = 0.0, ew = 0.0;
    for (int m = 0; m < nmb; ++m) {
      for (int k = indcs.ks; k <= indcs.ke; ++k) {
        for (int j = indcs.js; j <= indcs.je; ++j) {
          for (int i = indcs.is; i <= indcs.ie; ++i) {
            eu += u_h(m, IEN, k, j, i)*vol_c;
            const Real v2 = w_old(m, IVX, k, j, i)*w_old(m, IVX, k, j, i) +
                            w_old(m, IVY, k, j, i)*w_old(m, IVY, k, j, i) +
                            w_old(m, IVZ, k, j, i)*w_old(m, IVZ, k, j, i);
            ew += (w_old(m, IEN, k, j, i) + 0.5*w_old(m, IDN, k, j, i)*v2)*vol_c;
          }
        }
      }
    }
    std::printf("  E_tot from u0 %.8e erg, from w0 %.8e erg\n", eu*e_cgs, ew*e_cgs);
  }
  // gas temperature [K] of one cell of a host copy of w0
  auto temp = [&](auto &w, int m, int k, int j, int i) {
    Real x_e = 0.0;
    if (spec.i_e >= 0) {
      x_e = w(m, s0 + spec.i_e, k, j, i);
    } else {
      for (int n = 0; n < nspec; ++n) x_e += spec.charge[n]*w(m, s0 + n, k, j, i);
    }
    const Real x_h2 = (spec.i_h2 >= 0) ? w(m, s0 + spec.i_h2, k, j, i) : 0.0;
    const Real n_h = w(m, IDN, k, j, i)*nh_per_rho;
    return w(m, IEN, k, j, i)*gm1*pk/(n_h*(1.0 + X_HE - x_h2 + x_e));
  };
  // each step: one chemistry update, then the thermal energy lost per cell
  for (int s = 0; s < diag_steps; ++s) {
    pmbp->pchemistry2->Advance(diag_dt);
    auto w_new = Kokkos::create_mirror_view_and_copy(HostMemSpace(), w0);
    struct Cell { Real de; int m, k, j, i; };
    std::vector<Cell> cells;
    Real e_tot = 0.0, de_tot = 0.0;
    for (int m = 0; m < nmb; ++m) {
      for (int k = indcs.ks; k <= indcs.ke; ++k) {
        for (int j = indcs.js; j <= indcs.je; ++j) {
          for (int i = indcs.is; i <= indcs.ie; ++i) {
            const Real de = (w_old(m, IEN, k, j, i) - w_new(m, IEN, k, j, i))*vol_c;
            e_tot += w_new(m, IEN, k, j, i)*vol_c;
            de_tot += de;
            cells.push_back({de, m, k, j, i});
          }
        }
      }
    }
    // the five cells that lost the most energy, with their state before the step
    const int ntop = std::min<int>(5, cells.size());
    std::partial_sort(cells.begin(), cells.begin() + ntop, cells.end(),
                      [](const Cell &a, const Cell &b) { return a.de > b.de; });
    std::printf("  step %d: E_th %.6e erg, loss rate %.4e erg/s\n", s + 1, e_tot*e_cgs,
                de_tot*e_cgs/(diag_dt*t_cgs));
    for (int c = 0; c < ntop; ++c) {
      const Cell &q = cells[c];
      std::printf("    (%d,%d,%d,%d) %.3e erg/s  n_H %.3e  T %.4e -> %.4e K ", q.m, q.k,
                  q.j, q.i, q.de*e_cgs/(diag_dt*t_cgs),
                  w_old(q.m, IDN, q.k, q.j, q.i)*nh_per_rho,
                  temp(w_old, q.m, q.k, q.j, q.i), temp(w_new, q.m, q.k, q.j, q.i));
      for (int n = 0; n < nspec; ++n) {
        std::printf(" %s %.3e", spec.name[n].c_str(), w_old(q.m, s0 + n, q.k, q.j, q.i));
      }
      std::printf("\n");
    }
    Kokkos::deep_copy(w_old, w_new);
  }
  std::exit(EXIT_SUCCESS);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadiativeSNR()

void ProblemGenerator::RadiativeSNR(ParameterInput *pin, const bool restart) {
  user_hist_func = SNRHistory;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  // without a <chemistry2> block: adiabatic hydro, no species (for energy checks)
  const bool chem = (pmbp->pchemistry2 != nullptr);
  if (chem) {
    spec = GetSpecies(pin);
    first_scalar = chemistry2::ChemistryScalarStart(pmbp);
    mu_h = chemistry2::ChemistryMuH(pmbp);
  } else {
    spec = SpeciesInfo();
    mu_h = pin->GetOrAddReal("problem", "mu_H", 1.4);
  }
  if (restart) return;

  // ambient density and pressure in code units
  auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;
  const auto &indcs = pmy_mesh_->mb_indcs;
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  const Real n_h0 = pin->GetReal("problem", "nH_0");
  const Real pok = pin->GetReal("problem", "pgas_0");
  const Real rho_0 = n_h0*pmbp->punit->hydrogen_mass_cgs*mu_h/pmbp->punit->density_cgs();
  const Real p_0 = pok*pmbp->punit->k_boltzmann_cgs/pmbp->punit->pressure_cgs();

  // ambient and ejecta abundances (Tigris radiative_snr.cpp)
  const std::string net = chem ? pin->GetString("chemistry2", "network") : "";
  const bool gow17 = net.rfind("GOW17", 0) == 0;
  const Real zg = chem ? pin->GetOrAddReal("chemistry2", gow17 ? "GOW17_Z_g" : "NCR_Z_g",
                                           1.0) : 1.0;
  std::vector<Real> ambient(spec.nspec, 0.0), ejecta(spec.nspec, 0.0);
  for (int n = 0; n < spec.nspec; ++n) {
    const std::string &s = spec.name[n];
    if (s == "H+") {
      ambient[n] = 0.01;
      ejecta[n] = 1.0;
    }
    if (gow17) {
      const Real xc = zg*pin->GetOrAddReal("chemistry2", "GOW17_xC", 1.6e-4);
      const Real xo = zg*pin->GetOrAddReal("chemistry2", "GOW17_xO", 3.2e-4);
      const Real xsi = zg*pin->GetOrAddReal("chemistry2", "GOW17_xSi", 1.7e-6);
      if (s == "C+") ambient[n] = ejecta[n] = xc;
      if (s == "Si+") ambient[n] = ejecta[n] = xsi;
      if (s == "He+") ejecta[n] = pin->GetOrAddReal("chemistry2", "GOW17_xHe", X_HE);
      if (s == "O+") ejecta[n] = xo;
    } else {
      if (s == "e") {
        ambient[n] = 0.01;
        ejecta[n] = chemistry2::ncr::X_EL_MAX;
      }
    }
  }
  // copy the ambient abundances to the device
  DualArray1D<Real> amb("snr_ambient", spec.nspec);
  for (int n = 0; n < spec.nspec; ++n) amb.view_host()(n) = ambient[n];
  amb.modify_host();
  amb.sync_device();
  auto amb_d = amb.view_device();

  // uniform state, ghost cells included
  const int nmb = pmbp->nmb_thispack, nspec = spec.nspec, s0 = first_scalar;
  const int n3 = indcs.nx3 + 2*indcs.ng, n2 = indcs.nx2 + 2*indcs.ng;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const Real eint_0 = p_0/gm1;
  par_for("snr_ambient", DevExeSpace(), 0, nmb - 1, 0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m, IDN, k, j, i) = rho_0;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = eint_0;
    for (int n = 0; n < nspec; ++n) w0(m, s0 + n, k, j, i) = amb_d(n);
  });

  // chemistry and temperature of the ambient gas: the chemistry alone at fixed density
  const int n_eq = pin->GetOrAddInteger("problem", "n_equilibrate", 0);
  const Real dt_eq = pin->GetOrAddReal("problem", "dt_equilibrate", 0.1);
  if (chem) {
    for (int s = 0; s < n_eq; ++s) pmbp->pchemistry2->Advance(dt_eq);
  }
  pmbp->phydro->peos->PrimToCons(w0, u0, 0, n1 - 1, 0, n2 - 1, 0, n3 - 1);

  // the supernova, on the host (once)
  const Real m_ej = pin->GetOrAddReal("problem", "M_ej", 10.0)*pmbp->punit->msun();
  const Real e_sn = pin->GetOrAddReal("problem", "E_SN", 1.0)*1.0e51/
                    pmbp->punit->energy_cgs();
  const int n_sc = pin->GetOrAddInteger("problem", "N_sc", 3);
  const Real rfb_res = pin->GetOrAddReal("problem", "rfb_res", 3.0);
  const Real dx_sc = 1.0/n_sc;
  // weighted volume of the feedback region in grid units
  Real vol_g = 0.0;
  RadialLoop(rfb_res + 0.5*dx_sc, dx_sc, [&](Real, Real, Real, Real r) {
    vol_g += dx_sc*dx_sc*dx_sc*BoundaryWeight(r/rfb_res, 0.5*dx_sc/rfb_res);
  });

  // deposit each subcell's share into the cell that contains it, on a host copy of u0
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u0);
  auto size_h = pmbp->pmb->mb_size.h_view;
  Real m_added = 0.0, e_added = 0.0;
  for (int m = 0; m < nmb; ++m) {
    const auto &sz = size_h(m);
    // mass and energy per unit weight of one subcell
    const Real dx = sz.dx1, vol_cell = sz.dx1*sz.dx2*sz.dx3;
    const Real vol_fb = vol_g*vol_cell;
    const Real rho_small = m_ej/vol_fb*dx_sc*dx_sc*dx_sc;
    const Real eps_small = e_sn/vol_fb*dx_sc*dx_sc*dx_sc;
    const Real rmax_sc = rfb_res + 0.5*dx_sc;
    RadialLoop(rmax_sc*dx, dx_sc*dx, [&](Real xs, Real ys, Real zs, Real rs) {
      // the cell of this MeshBlock that holds the subcell (skip it if none)
      const int i = indcs.is + static_cast<int>(std::floor((xs - sz.x1min)/sz.dx1));
      const int j = indcs.js + static_cast<int>(std::floor((ys - sz.x2min)/sz.dx2));
      const int k = indcs.ks + static_cast<int>(std::floor((zs - sz.x3min)/sz.dx3));
      if (i < indcs.is || i > indcs.ie || j < indcs.js || j > indcs.je ||
          k < indcs.ks || k > indcs.ke) return;
      const Real w = BoundaryWeight(rs/(rfb_res*dx), 0.5*dx_sc/rfb_res);
      if (w <= 0.0) return;
      // Tigris SubCellScheme::WeightFunc(rs/rfb): r in units of the feedback radius over
      // rmax in cells, so fkin rises to (1 + dx_sc/(2 rfb_res))/(rfb_res + dx_sc/2),
      // 0.32 for rfb_res 3, N_sc 3
      const Real fkin = (rs/(rfb_res*dx))/rmax_sc;
      const Real rho_inj = w*rho_small, eps_inj = w*eps_small;
      const Real nx = xs/rs, ny = ys/rs, nz = zs/rs;
      Real &d = u_h(m, IDN, k, j, i), &m1 = u_h(m, IM1, k, j, i);
      Real &m2 = u_h(m, IM2, k, j, i), &m3 = u_h(m, IM3, k, j, i);
      Real &en = u_h(m, IEN, k, j, i);
      // radial momentum density so that the kinetic energy rises by fkin eps_inj
      const Real ek0 = 0.5*(m1*m1 + m2*m2 + m3*m3)/d;
      const Real mcos = nx*m1 + ny*m2 + nz*m3;
      const Real msq = m1*m1 + m2*m2 + m3*m3;
      const Real pr = std::sqrt(mcos*mcos + 2.0*(ek0 + fkin*eps_inj)*(rho_inj + d) - msq)
                      - mcos;
      // add the mass, the radial momentum and the thermal share (1 - fkin) of the energy
      d += rho_inj;
      m1 += pr*nx;
      m2 += pr*ny;
      m3 += pr*nz;
      const Real ekf = 0.5*(m1*m1 + m2*m2 + m3*m3)/d;
      en += (ekf - ek0) + eps_inj*(1.0 - fkin);
      // the cell takes the ejecta abundances
      for (int n = 0; n < nspec; ++n) u_h(m, s0 + n, k, j, i) = d*ejecta[n];
      m_added += rho_inj*vol_cell;
      e_added += (ekf - ek0 + eps_inj*(1.0 - fkin))*vol_cell;
    });
  }
  // back to the device, then the primitives; report the injected totals
  Kokkos::deep_copy(u0, u_h);
  pmbp->phydro->peos->ConsToPrim(u0, w0, false, indcs.is, indcs.ie, indcs.js, indcs.je,
                                  indcs.ks, indcs.ke);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &m_added, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &e_added, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "radiative_snr: injected mass " << m_added/pmbp->punit->msun()
              << " Msun, energy " << e_added*pmbp->punit->energy_cgs()/1.0e51
              << " x 1e51 erg" << std::endl;
  }

  // optional chemistry-only diagnostic (see DiagChem)
  diag_steps = pin->GetOrAddInteger("problem", "diag_chem_steps", 0);
  diag_dt = pin->GetOrAddReal("problem", "diag_chem_dt", 1.0e-4);
  diag_cycle = pin->GetOrAddInteger("problem", "diag_after_cycles", 0);
  if (chem && diag_steps > 0 && diag_cycle == 0) DiagChem(pmbp, "after injection");
}

namespace {

//! Total mass, thermal energy, radial momentum; hot gas and shell
void SNRHistory(HistoryData *pdata, Mesh *pm) {
  enum : int {MASS, ETH, PR, M_HOT, VOL_HOT, ETH_HOT, PR_HOT, M_SH, VOL_SH, ETH_SH, PR_SH,
              RM_SH, NH};
  pdata->nhist = NH;
  const char *labels[NH] = {"mass", "eth", "pr", "M_hot", "vol_hot", "eth_hot", "pr_hot",
                            "M_sh", "vol_sh", "eth_sh", "pr_sh", "RM_sh"};
  for (int n = 0; n < NH; ++n) pdata->label[n] = labels[n];

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (diag_steps > 0 && diag_cycle > 0 && pm->ncycle >= diag_cycle) {
    DiagChem(pmbp, "after hydro cycles");
  }
  // unit conversions, and the species charges on the device for x_e
  auto &w0 = pmbp->phydro->w0;
  auto &size = pmbp->pmb->mb_size;
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  const Real nh_per_rho = pmbp->punit->density_cgs()/
                          (pmbp->punit->hydrogen_mass_cgs*mu_h);
  const Real t_per_p = pmbp->punit->pressure_cgs()/pmbp->punit->k_boltzmann_cgs;
  const Real v_km_s = pmbp->punit->velocity_cgs()/1.0e5;
  const int s0 = first_scalar, nspec = spec.nspec, i_h2 = spec.i_h2, i_e = spec.i_e;
  DualArray1D<int> chg("snr_charge", nspec);
  for (int n = 0; n < nspec; ++n) chg.view_host()(n) = spec.charge[n];
  chg.modify_host();
  chg.sync_device();
  auto chg_d = chg.view_device();

  const auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("snr_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &acc) {
    // cell indices, position, radial velocity and temperature
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = (idx - m*nkji - (k - ks)*nji - (j - js)*nx1) + is;
    const auto &sz = size.d_view(m);
    const Real vol = sz.dx1*sz.dx2*sz.dx3;
    const Real x = sz.x1min + (i - is + 0.5)*sz.dx1;
    const Real y = sz.x2min + (j - js + 0.5)*sz.dx2;
    const Real z = sz.x3min + (k - ks + 0.5)*sz.dx3;
    const Real rad = Kokkos::sqrt(x*x + y*y + z*z);
    const Real d = w0(m, IDN, k, j, i);
    const Real vr = (w0(m, IVX, k, j, i)*x + w0(m, IVY, k, j, i)*y +
                     w0(m, IVZ, k, j, i)*z)/rad;
    const Real eth = w0(m, IEN, k, j, i);
    Real x_e = 0.0;
    if (i_e >= 0) {
      x_e = w0(m, s0 + i_e, k, j, i);
    } else {
      for (int n = 0; n < nspec; ++n) x_e += chg_d(n)*w0(m, s0 + n, k, j, i);
    }
    const Real x_h2 = (i_h2 >= 0) ? w0(m, s0 + i_h2, k, j, i) : 0.0;
    const Real n_h = d*nh_per_rho;
    const Real T = gm1*eth*t_per_p/(n_h*(1.0 + X_HE - x_h2 + x_e));
    // this cell's contribution: all gas, then hot gas or shell
    array_sum::GlobalSum h;
    for (int n = 0; n < NHISTORY_VARIABLES; ++n) h.the_array[n] = 0.0;
    h.the_array[MASS] = d*vol;
    h.the_array[ETH] = eth*vol;
    h.the_array[PR] = d*vr*vol;
    if (T > T_HOT) {
      h.the_array[M_HOT] = d*vol;
      h.the_array[VOL_HOT] = vol;
      h.the_array[ETH_HOT] = eth*vol;
      h.the_array[PR_HOT] = d*vr*vol;
    } else if (vr*v_km_s > VR_SHELL) {
      h.the_array[M_SH] = d*vol;
      h.the_array[VOL_SH] = vol;
      h.the_array[ETH_SH] = eth*vol;
      h.the_array[PR_SH] = d*vr*vol;
      h.the_array[RM_SH] = d*vol*rad;
    }
    acc += h;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  Kokkos::fence();
  for (int n = 0; n < NH; ++n) pdata->hdata[n] = sum.the_array[n];
}

}  // namespace
