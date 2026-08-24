#ifndef CHEMISTRY_THERMO_THERMO_HPP_
#define CHEMISTRY_THERMO_THERMO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file thermo.hpp
//  \brief Definitions for heating and cooling processes

#include "athena.hpp"
#include "units/units.hpp"

namespace chemistry {
/*!
 * \brief Stores the thermodynamic properties for chemistry.
 *
 */
class Thermo {
  friend class H2Network;

 public:
  Thermo() {}
  ~Thermo() = default;

  //-------------------------------------------------------------------------------------
  /*!
   * \brief This computes the specific heat (C_v) of a cold gas, i.e. assuming
   * that H2 rotational and vibrational levels not excited.
   *
   * xH2, xe = nH2 or ne / nH
   * xHe_total = xHeI + xHeII = 0.1 for solar value.
   *
   * \return Real specific heat per H atom.
   */
  // The analytic form here is what lets the GOW17 Jacobian hoist restore the
  // thermal coupling as a rank-1 update: with N = 1 - xH2 + xHe + xe we have
  // Cv = k_B N / (gamma - 1) and T = E / Cv, so dT/dxH2 = +T/N and dT/dxe = -T/N.
  // Mind the clamps below -- at a clamp boundary the derivative is zero.
  KOKKOS_FUNCTION static Real CvCold(const Real xH2, const Real xHe_total,
                                     const Real xe, const Real gamma) {
    // floor abundances and 0 to avoid mathematical issues with sqrt and log
    // later
    const Real xH20 = Kokkos::fmin(Kokkos::fmax(xH2, 0.0), 0.5);
    return units::Units::k_boltzmann_cgs *
           ((1. - 2. * xH20) + xH20 + xHe_total + Kokkos::fmax(xe, 0.0)) /
           (gamma - 1.0);
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Heating by cosmic ray ionization of H, He, and H2.
   *
   * \param xe
   * \param nH number density of H atom
   * \param xHI
   * \param xH2
   * \param crir_prim
   * \return Real cosmic ray heating in erg H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real HeatingCr(const Real xe, const Real nH,
                                        const Real xHI, const Real xH2,
                                        const Real crir_prim) {
    // heating for CR interaction with free electrons
    // Draine ISM book eq (30.4)
    // const Real qe = 287.1 * eV_;
    // heating rate per ionization in atomic region.
    // Draine ISM book eq (30.1)
    Real qHI;
    if (xe > 1.0e-9) {
      qHI = (6.5 + 26.4 * Kokkos::sqrt(xe / (xe + 0.07))) * eV_;
    } else {  // prevent sqrt of small negative number
      qHI = 6.5 * eV_;
    }

    // Heating rate per ionization in molecular region.
    // Despotic paper Appendix B
    Real qH2;
    const Real lognH = Kokkos::log10(nH);
    if (nH < 100.) {  // prevent log of small negative number
      qH2 = 10. * eV_;
    } else if (lognH < 4) {
      qH2 = (10. + 3. * (lognH - 2.) / 2.) * eV_;
    } else if (lognH < 7) {
      qH2 = (13. + 4. * (lognH - 4.) / 3.) * eV_;
    } else if (lognH < 10) {
      qH2 = (17. + (lognH - 7.) / 3.) * eV_;
    } else {
      qH2 = 18. * eV_;
    }

    const Real qtot = xHI * qHI + 2 * xH2 * qH2;
    return (crir_prim * qtot);
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Heating by photo electric effect on dust, including collisional
   * cooling. WD2001 Table 2 and 3. Use the diffuse ISM. Rv=3.1, ISRF, bc=4.
   *
   * \param G UV radiation scaled by solar neighborhood value, including
   * extinction.  G=G0 * exp(-NH*sigmaPE_) at one line of sight.
   * \param Zd dust abundance scaled by solar neighborhood value.
   * \param T temperature in K
   * \param ne electron number density in cm^-3.
   * \return Real Photoelectric heating by dust. Heating rate in erg H^-1 s^-1.
   */
  KOKKOS_FUNCTION static Real HeatingPE(const Real G, const Real Zd,
                                        const Real T, const Real ne) {
    const double small_ = 1e-10;
    if (ne < small_) {
      return 0.;
    }
    const Real x = 1.7 * G * Kokkos::sqrt(T) / ne + 50.;
    const Real fac = (CPE_[0] + CPE_[1] * Kokkos::pow(T, CPE_[4])) /
                     (1. + CPE_[2] * Kokkos::pow(x, CPE_[5]) *
                               (1. + CPE_[3] * Kokkos::pow(x, CPE_[6])));
    const Real heating = 1.7e-26 * G * Zd * fac;
    return heating;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Heating by H2 formation on dust grains, from Hollenbach + McKee
   * 1979, collisional rates from Visser et al. (2018)
   * (0) *H + *H + gr -> H2 + gr
   *
   * NOTE: All but the return line of this function is identical to
   * HeatingH2Pump, there might be optimization opportunities to combine them.
   *
   * \param xHI
   * \param xH2
   * \param nH Number density of hydrogen
   * \param T temperature in K
   * \param kgr grain reaction rates, kgr_ in NL99p.
   * \param k_xH2_photo photo dissociation of H2 by UV light per H2.
   * \return Real Heating rate by H2 formation on dust grains in erg H^-1 s^-1.
   */
  KOKKOS_FUNCTION static Real HeatingH2gr(const Real xHI, const Real xH2,
                                          const Real nH, const Real T,
                                          const Real kgr,
                                          const Real k_xH2_photo) {
    const Real A = 2.0e-7;
    const Real D = k_xH2_photo;
    const Real t = 1. + T / 1000.;
    const Real geff_H = Kokkos::pow(10, -11.06 + 0.0555 / t - 2.390 / (t * t));
    const Real geff_H2 = Kokkos::pow(10, -11.08 - 3.671 / t - 2.023 / (t * t));
    // critical density ncr, heating only effective at n > ncr
    const Real ncr = (A + D) / (geff_H * xHI + geff_H2 * xH2);
    const Real f = 1. / (1. + ncr / nH);
    return kgr * xHI * (0.2 + 4.2 * f) * eV_;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Heating by H2 UV pumping, Visser et al. (2018)
   *
   * NOTE: All but the return line of this function is identical to
   * HeatingH2gr, there might be optimization opportunities to combine them.
   *
   * \param xHI
   * \param xH2
   * \param nH Number density of hydrogen
   * \param T temperature in K
   * \param k_xH2_photo photo dissociation of H2 by UV light per H2. Calculated
   * in RHS in NL99p.
   * \return Real Heating rate by H2 UV pumping in erg H^-1 s^-1.
   */
  KOKKOS_FUNCTION static Real HeatingH2pump(const Real xHI, const Real xH2,
                                            const Real nH, const Real T,
                                            const Real k_xH2_photo) {
    const Real A = 2.0e-7;
    const Real D = k_xH2_photo;
    const Real t = 1. + T / 1000.;
    const Real geff_H = Kokkos::pow(10, -11.06 + 0.0555 / t - 2.390 / (t * t));
    const Real geff_H2 = Kokkos::pow(10, -11.08 - 3.671 / t - 2.023 / (t * t));
    // critical density ncr, heating only effective at n > ncr
    const Real ncr = (A + D) / (geff_H * xHI + geff_H2 * xH2);
    const Real f = 1. / (1. + ncr / nH);
    return D * 8. * 2.0 * f * eV_ * xH2;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Heating by H2 photodissociation. From Black + Dalgarno 1977, 0.4eV
   * per reaction.
   *
   * \param k_xH2_photo photo dissociation of H2 by UV light per H2. Calculated
   * in RHS in NL99p.
   * \param xH2
   * \return Real Heating rate by H2 photodissociation in erg H^-1 s^-1.
   */
  KOKKOS_FUNCTION static Real HeatingH2diss(const Real k_xH2_photo,
                                            const Real xH2) {
    return k_xH2_photo * xH2 * 0.4 * eV_;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by C+ fine structure line. Collisional species: HI, H2, e.
   *
   * \param xCII xCII = nC+/nH
   * \param nHI number density of HI, in cm^-3.
   * \param nH2 number density of H2, in cm^-3.
   * \param ne number density of e, in cm^-3.
   * \param T temperature in K
   * \return Real Cooling rate for C+ fine structure line in erg H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real CoolingCII(const Real xCII, const Real nHI,
                                         const Real nH2, const Real ne,
                                         const Real T) {
    const Real q10 = q10CII_(nHI, nH2, ne, T);
    const Real q01 =
        (g1CII_ / g0CII_) * q10 *
        Kokkos::exp(-E10CII_ / (units::Units::k_boltzmann_cgs * T));
    return Cooling2Level_(q01, q10, A10CII_, E10CII_, xCII);
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by CI fine structure line. Collisional species: HI, H2.
   * Ignores H+.
   *
   * \param xCI xCI = nCI/nH
   * \param nHI number density of HI, in cm^-3.
   * \param nH2 number density of H2, in cm^-3.
   * \param ne number density of e, in cm^-3.
   * \param T temperature in K
   * \return Real Cooling rate for C fine structure line in erg H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real CoolingCI(const Real xCI, const Real nHI,
                                        const Real nH2, const Real ne,
                                        const Real T) {
    // cut off CI cooling at very high temperature
    if (T > 1.0e6) {
      return 0;
    }
    // e collisional coefficients from Johnson, Burke, & Kingston 1987,
    //  JPhysB, 20, 2553
    const Real T2 = T / 100.;
    const Real lnT2 = Kokkos::log(T2);
    const Real lnT = Kokkos::log(T);
    // ke(u,l) = fac*gamma(u,l)/g(u)
    const Real fac = 8.629e-8 * Kokkos::sqrt(1.0e4 / T);
    Real k10e, k20e, k21e;
    Real lngamma10e, lngamma20e, lngamma21e;  // collisional strength
    if (T < 1.0e3) {
      lngamma10e =
          (((-6.56325e-4 * lnT - 1.50892e-2) * lnT + 3.61184e-1) * lnT -
           7.73782e-1) *
              lnT -
          9.25141;
      lngamma20e =
          (((0.705277e-2 * lnT - 0.111338) * lnT + 0.697638) * lnT - 1.30743) *
              lnT -
          7.69735;
      lngamma21e =
          (((2.35272e-3 * lnT - 4.18166e-2) * lnT + 0.358264) * lnT - 0.57443) *
              lnT -
          7.4387;

    } else {
      lngamma10e =
          (((1.0508e-1 * lnT - 3.47620) * lnT + 4.2595e1) * lnT - 2.27913e2) *
              lnT +
          4.446e2;
      lngamma20e =
          (((9.38138e-2 * lnT - 3.03283) * lnT + 3.61803e1) * lnT - 1.87474e2) *
              lnT +
          3.50609e2;
      lngamma21e =
          (((9.78573e-2 * lnT - 3.19268) * lnT + 3.85049e1) * lnT - 2.02193e2) *
              lnT +
          3.86186e2;
    }
    k10e = fac * Kokkos::exp(lngamma10e) / g1CI_;
    k20e = fac * Kokkos::exp(lngamma20e) / g2CI_;
    k21e = fac * Kokkos::exp(lngamma21e) / g2CI_;
    // HI collisional rates, Draine (2011) ISM book Appendix F Table F.6
    //  NOTE: this is more updated than the LAMBDA database.
    const Real k10HI = 1.26e-10 * Kokkos::pow(T2, 0.115 + 0.057 * lnT2);
    const Real k20HI = 0.89e-10 * Kokkos::pow(T2, 0.228 + 0.046 * lnT2);
    const Real k21HI = 2.64e-10 * Kokkos::pow(T2, 0.231 + 0.046 * lnT2);
    // H2 collisional rates, Draine (2011) ISM book Appendix F Table F.6
    const Real k10H2p = 0.67e-10 * Kokkos::pow(T2, -0.085 + 0.102 * lnT2);
    const Real k10H2o = 0.71e-10 * Kokkos::pow(T2, -0.004 + 0.049 * lnT2);
    const Real k20H2p = 0.86e-10 * Kokkos::pow(T2, -0.010 + 0.048 * lnT2);
    const Real k20H2o = 0.69e-10 * Kokkos::pow(T2, 0.169 + 0.038 * lnT2);
    const Real k21H2p = 1.75e-10 * Kokkos::pow(T2, 0.072 + 0.064 * lnT2);
    const Real k21H2o = 1.48e-10 * Kokkos::pow(T2, 0.263 + 0.031 * lnT2);
    const Real k10H2 = k10H2p * fp_ + k10H2o * fo_;
    const Real k20H2 = k20H2p * fp_ + k20H2o * fo_;
    const Real k21H2 = k21H2p * fp_ + k21H2o * fo_;
    // The total collisional rates
    const Real q10 = k10HI * nHI + k10H2 * nH2 + k10e * ne;
    const Real q20 = k20HI * nHI + k20H2 * nH2 + k20e * ne;
    const Real q21 = k21HI * nHI + k21H2 * nH2 + k21e * ne;
    const Real kbT = units::Units::k_boltzmann_cgs * T;
    const Real q01 = (g1CI_ / g0CI_) * q10 * Kokkos::exp(-E10CI_ / kbT);
    const Real q02 = (g2CI_ / g0CI_) * q20 * Kokkos::exp(-E20CI_ / kbT);
    const Real q12 = (g2CI_ / g1CI_) * q21 * Kokkos::exp(-E21CI_ / kbT);

    return Cooling3Level_(q01, q10, q02, q20, q12, q21, A10CI_, A20CI_, A21CI_,
                          E10CI_, E20CI_, E21CI_, xCI);
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by OI fine structure line. Collisional species: HI, H2, e.
   * The cooling rate is very insensitive to ne, for xe <~0.5. At xe >~0.5
   * region, the O+ and other cooling will start to be important anyway.
   *
   * \param xOI xOI = nOI/nH.
   * \param nHI number density of HI, in cm^-3.
   * \param nH2 number density of H2, in cm^-3.
   * \param ne number density of e, in cm^-3.
   * \param T temperature in K
   * \return Real Cooling rate for OI fine structure line in erg H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real CoolingOI(const Real xOI, const Real nHI,
                                        const Real nH2, const Real ne,
                                        const Real T) {
    // collisional rates from  Draine (2011) ISM book Appendix F Table F.6
    const Real T2 = T / 100;
    const Real lnT2 = Kokkos::log(T2);
    // HI
    const Real k10HI = 3.57e-10 * Kokkos::pow(T2, 0.419 - 0.003 * lnT2);
    const Real k20HI = 3.19e-10 * Kokkos::pow(T2, 0.369 - 0.006 * lnT2);
    const Real k21HI = 4.34e-10 * Kokkos::pow(T2, 0.755 - 0.160 * lnT2);
    // H2
    const Real k10H2p = 1.49e-10 * Kokkos::pow(T2, 0.264 + 0.025 * lnT2);
    const Real k10H2o = 1.37e-10 * Kokkos::pow(T2, 0.296 + 0.043 * lnT2);
    const Real k20H2p = 1.90e-10 * Kokkos::pow(T2, 0.203 + 0.041 * lnT2);
    const Real k20H2o = 2.23e-10 * Kokkos::pow(T2, 0.237 + 0.058 * lnT2);
    const Real k21H2p = 2.10e-12 * Kokkos::pow(T2, 0.889 + 0.043 * lnT2);
    const Real k21H2o = 3.00e-12 * Kokkos::pow(T2, 1.198 + 0.525 * lnT2);
    const Real k10H2 = k10H2p * fp_ + k10H2o * fo_;
    const Real k20H2 = k20H2p * fp_ + k20H2o * fo_;
    const Real k21H2 = k21H2p * fp_ + k21H2o * fo_;
    // e
    // fit from Bell+1998
    const Real k10e = 5.12e-10 * Kokkos::pow(T, -0.075);
    const Real k20e = 4.86e-10 * Kokkos::pow(T, -0.026);
    const Real k21e = 1.08e-14 * Kokkos::pow(T, 0.926);
    // total collisional rates
    const Real q10 = k10HI * nHI + k10H2 * nH2 + k10e * ne;
    const Real q20 = k20HI * nHI + k20H2 * nH2 + k20e * ne;
    const Real q21 = k21HI * nHI + k21H2 * nH2 + k21e * ne;
    const Real kbT = units::Units::k_boltzmann_cgs * T;
    const Real q01 = (g1OI_ / g0OI_) * q10 * Kokkos::exp(-E10OI_ / (kbT));
    const Real q02 = (g2OI_ / g0OI_) * q20 * Kokkos::exp(-E20OI_ / (kbT));
    const Real q12 = (g2OI_ / g1OI_) * q21 * Kokkos::exp(-E21OI_ / (kbT));

    return Cooling3Level_(q01, q10, q02, q20, q12, q21, A10OI_, A20OI_, A21OI_,
                          E10OI_, E20OI_, E21OI_, xOI);
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by collisional excited lyman alpha line.
   *
   * \param xHI xHI = nHI/nH.
   * \param ne number density of e, in cm^-3.
   * \param T temperature in K
   * \return Real Cooling rate for Lyman alpha line in erg H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real CoolingLya(const Real xHI, const Real ne,
                                         const Real T) {
    const Real T4 = T / 1.0e4;
    const Real fac =
        5.31e-8 * Kokkos::pow(T4, 0.15) / (1. + Kokkos::pow(T4 / 5., 0.65));
    const Real k01e = fac * Kokkos::exp(-11.84 / T4);
    const Real q01 = k01e * ne;
    const Real q10 = (g0HI_ / g1HI_) * fac * ne;
    return Cooling2Level_(q01, q10, A10HI_, E10HI_, xHI);
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by CO rotational lines. Collision species: HI, H2, e. From
   * Omukai+2010
   *
   * \param xCO xCO = nCO/nH
   * \param nHI number density of HI, in cm^-3.
   * \param nH2 number density of H2, in cm^-3.
   * \param ne number density of e, in cm^-3.
   * \param T temperature in K
   * \param NCOeff effective column density of CO using LVG approximation.
   * \return Real Cooling rate for CO rotational lines in erg H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real CoolingCOR(const Real xCO, const Real nHI,
                                         const Real nH2, const Real ne, Real T,
                                         const Real NCOeff) {
    // effective number density of colliders
    // TODO(Gong): potentially can use despotic to generate a more accurate
    // value for interpolation, might be faster too
    const Real Tmax_CO = 2000.;  // maximum temperature above which use Tmax
    T = Kokkos::fmin(T, Tmax_CO);

    // factor to make the cooling rate goes to zero at T=0.
    const Real facT = Kokkos::pow(1. - Kokkos::exp(-T), 1.0e3);
    // small number for a very small NCOeff
    const Real eps = 1.0e13;
    const Real log_NCOeff =
        Kokkos::log10(NCOeff * 1.0e5 + eps);  // unit: cm^-2 / (km/s)
    const Real Troot4 = Kokkos::pow(T, 0.25);
    const Real neff = nH2 + 1.75 * Troot4 * nHI + 680.1 / Troot4 * ne;
    // interpolate parameters using given T and NCOeff
    // index of T and Neff
    const int iT0 =
        chemistry::interpolation::LinearInterpIndex(lenTCO_, TCO_, T);
    const int iNeff0 = chemistry::interpolation::LinearInterpIndex(
        lenNeffCO_, NeffCO_, log_NCOeff);
    // L0
    const Real log_L0 = -chemistry::interpolation::LP1Di(TCO_, L0CO_, iT0, T);
    const Real L0 = Kokkos::pow(10, log_L0);
    // LLTE
    const Real log_LLTE = -chemistry::interpolation::LP2Di(
        TCO_, NeffCO_, lenTCO_, iT0, iNeff0, LLTECO_, T, log_NCOeff);
    const Real LLTE = Kokkos::pow(10, log_LLTE);
    // n1/2
    const Real log_nhalf = chemistry::interpolation::LP2Di(
        TCO_, NeffCO_, lenTCO_, iT0, iNeff0, nhalfCO_, T, log_NCOeff);
    const Real nhalf = Kokkos::pow(10, log_nhalf);
    // alpha
    const Real alpha = chemistry::interpolation::LP2Di(
        TCO_, NeffCO_, lenTCO_, iT0, iNeff0, alphaCO_, T, log_NCOeff);
    const Real inv_LCO =
        1. / L0 + neff / LLTE +
        1. / L0 * Kokkos::pow(neff / nhalf, alpha) * (1. - nhalf * L0 / LLTE);

    return (1. / inv_LCO) * neff * xCO * facT;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by H2 vibration and rotation lines. Collision species: HI,
   * H2, He, H+, e. Note: Using Glover + Abel 2008 fitting formulas in Table 8,
   * assuming that ortho to para ration of H2 is 3:1. Updated H2-e and H2-H+
   * rates from Glover (2015) MNRAS.451.2082G Appendix
   *
   * \param xH2 xH2 = nH2 / nH
   * \param nHI ni: number density of species i, in cm^-3
   * \param nH2 ni: number density of species i, in cm^-3
   * \param nHe ni: number density of species i, in cm^-3
   * \param nHplus ni: number density of species i, in cm^-3
   * \param ne ni: number density of species i, in cm^-3
   * \param T: temperature in K
   * \return Real Cooling rate for H2 vibrational and rotational lines in erg
   * H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real CoolingH2(const Real xH2, const Real nHI,
                                        const Real nH2, const Real nHe,
                                        const Real nHplus, const Real ne,
                                        Real T) {
    // Tmax_H2 set by comparing extrapolation of Glover15 to Mosely21
    const Real Tmax_H2 = 1.0e4;  // maximum temperature above which use Tmax
    const Real Tmin_H2 = 10.;    // min temperature below which cut off cooling

    // Note: limit extended to T< 10K and T>6000K
    if (T > Tmax_H2) {
      T = Tmax_H2;
    } else if (T < Tmin_H2) {
      return 0.;
    }

    const Real logT3 = Kokkos::log10(T / 1.0e3);
    const Real logT3_2 = logT3 * logT3;
    const Real logT3_3 = logT3_2 * logT3;
    const Real logT3_4 = logT3_3 * logT3;
    const Real logT3_5 = logT3_4 * logT3;
    const Real logT3_6 = logT3_5 * logT3;
    const Real logT3_7 = logT3_6 * logT3;
    const Real logT3_8 = logT3_7 * logT3;
    Real LHI, LH2, LHe, LHplus, Le;
    // HI
    if (T < 100) {
      LHI = Kokkos::pow(10, -16.818342e0 + 3.7383713e1 * logT3 +
                                5.8145166e1 * logT3_2 + 4.8656103e1 * logT3_3 +
                                2.0159831e1 * logT3_4 + 3.8479610e0 * logT3_5);
    } else if (T < 1000) {
      LHI = Kokkos::pow(10, -2.4311209e1 + 3.5692468e0 * logT3 -
                                1.1332860e1 * logT3_2 - 2.7850082e1 * logT3_3 -
                                2.1328264e1 * logT3_4 - 4.2519023e0 * logT3_5);
    } else {
      LHI = Kokkos::pow(10, -2.4311209e1 + 4.6450521e0 * logT3 -
                                3.7209846e0 * logT3_2 + 5.9369081e0 * logT3_3 -
                                5.5108049e0 * logT3_4 + 1.5538288e0 * logT3_5);
    }
    // H2
    LH2 = Kokkos::pow(10, -2.3962112e1 + 2.09433740e0 * logT3 -
                              0.77151436e0 * logT3_2 + 0.43693353e0 * logT3_3 -
                              0.14913216e0 * logT3_4 - 0.033638326e0 * logT3_5);
    // He
    LHe = Kokkos::pow(10, -2.3689237e1 + 2.1892372e0 * logT3 -
                              0.81520438e0 * logT3_2 + 0.29036281e0 * logT3_3 -
                              0.16596184e0 * logT3_4 + 0.19191375e0 * logT3_5);
    // H+
    LHplus =
        Kokkos::pow(10, -2.2089523e1 + 1.5714711e0 * logT3 +
                            0.015391166e0 * logT3_2 - 0.23619985e0 * logT3_3 -
                            0.51002221e0 * logT3_4 + 0.32168730e0 * logT3_5);
    // e
    if (T < 500) {
      Le = Kokkos::pow(10, -2.1928796e1 + 1.6815730e1 * logT3 +
                               9.6743155e1 * logT3_2 + 3.4319180e2 * logT3_3 +
                               7.3471651e2 * logT3_4 + 9.8367576e2 * logT3_5 +
                               8.0181247e2 * logT3_6 + 3.6414446e2 * logT3_7 +
                               7.0609154e1 * logT3_8);
    } else {
      Le = Kokkos::pow(10, -2.2921189e1 + 1.6802758e0 * logT3 +
                               0.93310622e0 * logT3_2 + 4.0406627e0 * logT3_3 -
                               4.7274036e0 * logT3_4 - 8.8077017e0 * logT3_5 +
                               8.9167183 * logT3_6 + 6.4380698 * logT3_7 -
                               6.3701156 * logT3_8);
    }
    // total cooling in low density limit
    const Real Gamma_n0 =
        LHI * nHI + LH2 * nH2 + LHe * nHe + LHplus * nHplus + Le * ne;
    // cooling rate at LTE, from Hollenbach + McKee 1979
    const Real T3 = T / 1.0e3;
    const Real Gamma_LTE_HR = (9.5e-22 * Kokkos::pow(T3, 3.76)) /
                                  (1. + 0.12 * Kokkos::pow(T3, 2.1)) *
                                  Kokkos::exp(-Kokkos::pow(0.13 / T3, 3)) +
                              3.e-24 * Kokkos::exp(-0.51 / T3);
    const Real Gamma_LTE_HV =
        6.7e-19 * Kokkos::exp(-5.86 / T3) + 1.6e-18 * Kokkos::exp(-11.7 / T3);
    const Real Gamma_LTE = Gamma_LTE_HR + Gamma_LTE_HV;
    // Total cooling rate
    Real Gamma_tot;
    if (Gamma_n0 > 1e-100) {
      Gamma_tot = Gamma_LTE / (1.0 + Gamma_LTE / Gamma_n0);
    } else {
      Gamma_tot = 0;
    }
    return Gamma_tot * xH2;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by recombination of e on PAHs, from WD2001 Eq(45).
   *
   * \param Zd dust metalicity compared to solar neighborhood.
   * \param T temperature in K.
   * \param ne number density of electrons.
   * \param G UV radiation scaled by solar neighborhood value, including
   * extinction. G=G0 * exp(-NH*sigmaPE_) at one line of sight.
   * \return Real Cooling rate for  recombination of e on PAHs in erg H^-1 s^-1
   */
  KOKKOS_FUNCTION static Real CoolingRec(const Real Zd, const Real T,
                                         const Real ne, const Real G) {
    const Real x = 1.7 * G * Kokkos::sqrt(T) / (ne + 1e-50) + 50.;
    const Real lnx = Kokkos::log(x);
    const Real cooling = 1.0e-28 * ne *
                         Kokkos::pow(T, DPE_[0] + DPE_[1] / lnx) *
                         Kokkos::exp(DPE_[2] + (DPE_[3] - DPE_[4] * lnx) * lnx);
    return cooling * Zd;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by collisional dissociation of H2.
   *  H2 + *H -> 3 *H
   *  H2 + H2 -> H2 + 2 *H
   * reaction heat: 4.48 eV from Krome Paper
   *
   * \param xHI xi = ni/nH
   * \param xH2 xi = ni/nH
   * \param k_H2_H reaction rate coefficients (k2body_ in NL99p)
   * \param k_H2_H2 reaction rate coefficients (k2body_ in NL99p)
   * \return Real Cooling rate for H2 collisional dissociation in erg H^-1 s^-1.
   */
  KOKKOS_FUNCTION static Real CoolingH2diss(const Real xHI, const Real xH2,
                                            const Real k_H2_H,
                                            const Real k_H2_H2) {
    const Real rate15 = k_H2_H * xH2 * xHI;
    const Real rate16 = k_H2_H2 * xH2 * xH2;
    return 4.48 * eV_ * (rate15 + rate16);
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief Cooling by collisional ionization of HI
   *  *H + *e -> H+ + 2 *e
   * reaction heat: 13.6 eV from Krome Paper
   *
   * \param xHI xi = ni/nH
   * \param xe xi = ni/nH
   * \param k_H_e reaction rate coefficients (k2body_ in NL99p)
   * \return Real Cooling rate for collisional ionization of HI in erg H^-1
   * s^-1.
   */
  KOKKOS_FUNCTION static Real CoolingHIion(const Real xHI, const Real xe,
                                           const Real k_H_e) {
    const Real rate = k_H_e * xHI * xe;
    return 13.6 * eV_ * rate;
  }

 private:
  //----------------------------------------------------------------------------------------
  /*!
   * \brief Compute collisional rate for C+ atom in s^-1. Collisional species:
   * HI, H2, e.
   *
   * \param nHI number density of HI, in cm^-3.
   * \param nH2 number density of H2, in cm^-3.
   * \param ne number density of e, in cm^-3.
   * \param T  temperature in K
   * \return Real Collisional rate for C+ atom in s^-1.
   */
  KOKKOS_FUNCTION static Real q10CII_(const Real nHI, const Real nH2,
                                      const Real ne, const Real T) {
    // Draine (2011) ISM book eq (17.16) and (17.17)
    const Real T2 = T / 100.;
    const Real k10e = 4.53e-8 * Kokkos::sqrt(1.0e4 / T);
    const Real k10HI =
        7.58e-10 * Kokkos::pow(T2, 0.1281 + 0.0087 * Kokkos::log(T2));
    Real k10oH2 = 0;
    Real k10pH2 = 0;
    Real tmp = 0;
    if (T < 500.) {
      // fit in Wiesenfeld & Goldsmith 2014
      k10oH2 = (5.33 + 0.11 * T2) * 1.0e-10;
      k10pH2 = (4.43 + 0.33 * T2) * 1.0e-10;
    } else {
      // Glover+ Jappsen 2007, for high temperature scales similar to HI
      tmp = Kokkos::pow(T, 0.07);
      k10oH2 = 3.74757785025e-10 * tmp;
      k10pH2 = 3.88997286356e-10 * tmp;
    }
    const Real k10H2 = k10oH2 * fo_ + k10pH2 * fp_;
    return (k10e * ne + k10HI * nHI + k10H2 * nH2);
  }

  //----------------------------------------------------------------------------------------
  /*!
   * \brief line cooling rate per H for 2 level atom. Ignore radiative
   * excitation and de-excitation, and assume optically thin.
   *
   * \param q01 \f$ \sum (n_c  k_{s, 01}) \f$. Collisional excitation rate per
   * second for all the collider species summed up together from level 0 to 1.
   * \param q10 \f$ \sum (n_c k_{s, 10}) \f$. Collisional de-excitation rate per
   * second, similar to q01.
   * \param A10 Einstein A coefficient for spontaneous emission from level 1 to
   * 0, in sec^-1.
   * \param E10 energy difference of levels E1 - E0, in erg.
   * \param xs ns/nH, abundances of species s.
   * \return Real Line cooling rate in erg H^-1 s^-1.
   */
  KOKKOS_FUNCTION static Real Cooling2Level_(const Real q01, const Real q10,
                                             const Real A10, const Real E10,
                                             const Real xs) {
    const Real f1 = q01 / (q01 + q10 + A10);
    return f1 * A10 * E10 * xs;
  }

  //-------------------------------------------------------------------------------------
  /*!
   * \brief line cooling rate per H for 3 level atom. Ignore radiative
   * excitation and de-excitation, and assume optically thin.
   *
   * \param qij \f$ \sum (n_c  k_{s, ij}) \f$. Collisional excitation rate per
   * second for all the collider species summed up together from level i to j.
   * \param Aij Einstein A coefficient for spontaneous emission from level i to
   * j, in sec^-1, i > j.
   * \param Eij energy difference of levels Ei - Ej, in erg.
   * \param xs ns/nH, abundances of species s.
   * \return Real Total line cooling rate in erg H^-1 s^-1.
   */
  KOKKOS_FUNCTION static Real Cooling3Level_(const Real q01, const Real q10,
                                             const Real q02, const Real q20,
                                             const Real q12, const Real q21,
                                             const Real A10, const Real A20,
                                             const Real A21, const Real E10,
                                             const Real E20, const Real E21,
                                             const Real xs) {
    const Real R10 = q10 + A10;
    const Real R20 = q20 + A20;
    const Real R21 = q21 + A21;
    const Real a0 = R10 * R20 + R10 * R21 + q12 * R20;
    const Real a1 = q01 * R20 + q01 * R21 + R21 * q02;
    const Real a2 = q02 * R10 + q02 * q12 + q12 * q01;
    const Real de = a0 + a1 + a2;
    const Real f1 = a1 / de;
    const Real f2 = a2 / de;
    return (f1 * A10 * E10 + f2 * (A20 * E20 + A21 * E21)) * xs;
  }

  //-------------------------------------------------------------------------------------
  // physical constants
  static constexpr Real eV_ = 1.602e-12;  // eV in erg
  // speed of light * radiation constant, or stephan-boltzmann constant*4
  static constexpr Real ca_ = 2.27e-4;
  static constexpr Real TCMB_ = 2.73;  // CMB temperature
  // ortho to para ratio of H2
  static constexpr Real o2p_ = 3.;   // ratio of ortho to para H2
  static constexpr Real fo_ = 0.75;  // ortho H2 fraction
  static constexpr Real fp_ = 0.25;  // para H2 fraction
  // dust cross-section for 8-13.6eV photons in cm2
  // DESPOTIC: Krumhols, MNRAS 437, 1662–1680 (2014)
  static constexpr Real sigmaPE_ = 1.0e-21;
  static constexpr Real sigmaISRF_ =
      3.0e-22;  // dust cross-section for ISRF in cm2, DESPOTIC
  static constexpr Real sigmad10_ =
      2.0e-25;  // dust cross-section for IR at T=10K, DESPOTIC
  static constexpr Real alpha_GD_ =
      3.2e-34;  // gas-dust coupling coefficient, DESPOTIC
  // atomic constants
  // Aij: Einstein A coefficient for level i->j, in s^-1
  // Eij: Energy level Eij = Ei - Ej, in erg
  // gi: degeneracy in level i
  // ----C+, 2 level system---
  static constexpr Real A10CII_ = 2.3e-6;  // Silva+Viegas2002
  static constexpr Real E10CII_ = 1.26e-14;
  static constexpr Real g0CII_ = 2.;
  static constexpr Real g1CII_ = 4.;
  // ----HI+, 2 level system---
  static constexpr Real A10HI_ = 6.265e8;
  static constexpr Real E10HI_ = 1.634e-11;
  static constexpr Real g0HI_ = 1.;
  static constexpr Real g1HI_ = 3.;
  // ----CI, 3 level system---
  static constexpr Real g0CI_ = 1;
  static constexpr Real g1CI_ = 3;
  static constexpr Real g2CI_ = 5;
  static constexpr Real A10CI_ = 7.880e-08;
  static constexpr Real A20CI_ = 1.810e-14;
  static constexpr Real A21CI_ = 2.650e-07;
  static constexpr Real E10CI_ = 3.261e-15;
  static constexpr Real E20CI_ = 8.624e-15;
  static constexpr Real E21CI_ = 5.363e-15;
  // ----OI, 3 level system---
  static constexpr Real g0OI_ = 5;
  static constexpr Real g1OI_ = 3;
  static constexpr Real g2OI_ = 1;
  static constexpr Real A10OI_ = 8.910e-05;
  static constexpr Real A20OI_ = 1.340e-10;
  static constexpr Real A21OI_ = 1.750e-05;
  static constexpr Real E10OI_ = 3.144e-14;
  static constexpr Real E20OI_ = 4.509e-14;
  static constexpr Real E21OI_ = 1.365e-14;

  // -----CO cooling table data, from Omukai+2010, ApJ, 722:1793–1815-----
  static constexpr size_t lenTCO_ = 11;
  static constexpr size_t lenNeffCO_ = 11;
  // gas temperature in K
  static constexpr Kokkos::Array<Real, lenTCO_> TCO_ = {
      10, 20, 30, 50, 80, 100, 300, 600, 1000, 1500, 2000};
  // effective column density of CO in cm^-2
  static constexpr Kokkos::Array<Real, lenNeffCO_> NeffCO_ = {
      14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0};
  // L0CO_, LLTECO_, nhalfCO_, alphaCO_ are fitting coefficients
  // in Omukai+2010 equation B1 and Table 2
  static constexpr Kokkos::Array<Real, lenTCO_> L0CO_ = {
      24.77, 24.38, 24.21, 24.03, 23.89, 23.82,
      // values from despotic, behaves better at high temperature
      23.34238089, 22.99832519, 22.75384686, 22.56640625, 22.43740866};
  //                                       23.42, 23.13, 22.91, 22.63, 22.28};
  static constexpr Kokkos::Array<Real, lenNeffCO_ * lenTCO_> LLTECO_ = {
      21.08, 20.35, 19.94, 19.45, 19.01, 18.80, 17.81, 17.23, 16.86, 16.66,
      16.55, 21.09, 20.35, 19.95, 19.45, 19.01, 18.80, 17.81, 17.23, 16.86,
      16.66, 16.55, 21.11, 20.37, 19.96, 19.46, 19.01, 18.80, 17.81, 17.23,
      16.86, 16.66, 16.55, 21.18, 20.40, 19.98, 19.47, 19.02, 18.81, 17.82,
      17.23, 16.87, 16.66, 16.55, 21.37, 20.51, 20.05, 19.52, 19.05, 18.83,
      17.82, 17.23, 16.87, 16.66, 16.55, 21.67, 20.73, 20.23, 19.64, 19.13,
      18.90, 17.85, 17.25, 16.88, 16.67, 16.56, 22.04, 21.05, 20.52, 19.87,
      19.32, 19.06, 17.92, 17.28, 16.90, 16.69, 16.58, 22.44, 21.42, 20.86,
      20.19, 19.60, 19.33, 18.08, 17.38, 16.97, 16.75, 16.63, 22.87, 21.82,
      21.24, 20.55, 19.95, 19.66, 18.34, 17.59, 17.15, 16.91, 16.78, 23.30,
      22.23, 21.65, 20.94, 20.32, 20.03, 18.67, 17.89, 17.48, 17.26, 17.12,
      23.76, 22.66, 22.06, 21.35, 20.71, 20.42, 19.03, 18.26, 17.93, 17.74,
      17.61};
  static constexpr Kokkos::Array<Real, lenNeffCO_ * lenTCO_> nhalfCO_ = {
      3.29,   3.49,  3.67,  3.97, 4.30, 4.46, 5.17, 5.47, 5.53, 5.30, 4.70,
      3.27,   3.48,  3.66,  3.96, 4.30, 4.45, 5.16, 5.47, 5.53, 5.30, 4.70,
      3.22,   3.45,  3.64,  3.94, 4.29, 4.45, 5.16, 5.47, 5.53, 5.30, 4.70,
      3.07,   3.34,  3.56,  3.89, 4.26, 4.42, 5.15, 5.46, 5.52, 5.30, 4.70,
      2.72,   3.09,  3.35,  3.74, 4.16, 4.34, 5.13, 5.45, 5.51, 5.29, 4.68,
      2.24,   2.65,  2.95,  3.42, 3.92, 4.14, 5.06, 5.41, 5.48, 5.26, 4.64,
      1.74,   2.15,  2.47,  2.95, 3.49, 3.74, 4.86, 5.30, 5.39, 5.17, 4.53,
      1.24,   1.65,  1.97,  2.45, 3.00, 3.25, 4.47, 5.02, 5.16, 4.94, 4.27,
      0.742,  1.15,  1.47,  1.95, 2.50, 2.75, 3.98, 4.57, 4.73, 4.52, 3.84,
      0.242,  0.652, 0.966, 1.45, 2.00, 2.25, 3.48, 4.07, 4.24, 4.03, 3.35,
      -0.258, 0.152, 0.466, 0.95, 1.50, 1.75, 2.98, 3.57, 3.74, 3.53, 2.85};
  static constexpr Kokkos::Array<Real, lenNeffCO_ * lenTCO_> alphaCO_ = {
      0.439, 0.409, 0.392, 0.370, 0.361, 0.357, 0.385, 0.437, 0.428, 0.354,
      0.322, 0.436, 0.407, 0.391, 0.368, 0.359, 0.356, 0.385, 0.437, 0.427,
      0.354, 0.322, 0.428, 0.401, 0.385, 0.364, 0.356, 0.352, 0.383, 0.436,
      0.427, 0.352, 0.320, 0.416, 0.388, 0.373, 0.353, 0.347, 0.345, 0.380,
      0.434, 0.425, 0.349, 0.316, 0.416, 0.378, 0.360, 0.338, 0.332, 0.330,
      0.371, 0.429, 0.421, 0.341, 0.307, 0.450, 0.396, 0.367, 0.334, 0.322,
      0.317, 0.355, 0.419, 0.414, 0.329, 0.292, 0.492, 0.435, 0.403, 0.362,
      0.339, 0.329, 0.343, 0.406, 0.401, 0.317, 0.276, 0.529, 0.473, 0.441,
      0.404, 0.381, 0.370, 0.362, 0.410, 0.392, 0.316, 0.272, 0.555, 0.503,
      0.473, 0.440, 0.423, 0.414, 0.418, 0.446, 0.404, 0.335, 0.289, 0.582,
      0.528, 0.499, 0.469, 0.457, 0.451, 0.470, 0.487, 0.432, 0.364, 0.310,
      0.596, 0.546, 0.519, 0.492, 0.483, 0.479, 0.510, 0.516, 0.448, 0.372,
      0.313};

  // --- dust cooling tabulated from DESPOTIC, not used ---------------
  static constexpr size_t lenTg_ = 10;
  static constexpr size_t lennH_ = 15;
  static constexpr Kokkos::Array<Real, lenTg_> logTg_ = {
      0.5,        0.88888889, 1.27777778, 1.66666667, 2.05555556,
      2.44444444, 2.83333333, 3.22222222, 3.61111111, 4.};
  static constexpr Kokkos::Array<Real, lennH_> lognH_ = {
      0.,         0.42857143, 0.85714286, 1.28571429, 1.71428571,
      2.14285714, 2.57142857, 3.,         3.42857143, 3.85714286,
      4.28571429, 4.71428571, 5.14285714, 5.57142857, 6.};
  static constexpr Kokkos::Array<Real, lennH_ * lenTg_> logps_ = {
      33.60923439, 32.35048647, 31.6458604,  31.02132235, 30.42222289,
      29.83261,    29.24673384, 28.66235604, 28.0785789,  27.49504472,
      33.18091039, 31.92216147, 31.21753302, 30.59298934, 29.99387726,
      29.40423904, 28.8183211,  28.23389204, 27.65007024, 27.06650687,
      32.75300149, 31.49424423, 30.78959638, 30.16501048, 29.5658181,
      28.97605855, 28.39000596, 27.80546791, 27.22157705, 26.6379757,
      32.32619855, 31.06738031, 30.36260198, 29.73777592, 29.13823862,
      28.54811853, 27.96178914, 27.37708183, 26.79309991, 26.20945204,
      31.90230918, 30.64308622, 29.93758718, 29.3117745,  28.71125923,
      28.12042115, 27.53366605, 26.94873467, 26.36464058, 25.78093706,
      31.48587783, 30.22433325, 29.51586354, 28.88730605, 28.28488476,
      27.69295575, 27.10563836, 26.52043033, 25.93620173, 25.35243225,
      31.0872853,  29.81525158, 29.09829079, 28.46439304, 27.85909326,
      27.26572717, 26.67771529, 26.09217504, 25.50778679, 24.92393938,
      30.72591396, 29.41932635, 28.68506438, 28.0430074,  27.4339034,
      26.83875916, 26.24991203, 25.66397699, 25.07939997, 24.49546057,
      30.42733764, 29.03861249, 28.27630163, 27.62323376, 27.00938204,
      26.41209045, 25.82224861, 25.23584618, 24.65104624, 24.06699834,
      30.21146872, 28.67535786, 27.87248634, 27.20529029, 26.58563564,
      25.9857724,  25.39474974, 24.80779463, 24.22273154, 23.63855567,
      30.08004755, 28.33375429, 27.47457781, 26.78951505, 26.16280582,
      25.55986866, 24.96744513, 24.37983659, 23.79446288, 23.21013604,
      30.0136033,  28.02095875, 27.08405993, 26.37636621, 25.74107062,
      25.13445646, 24.54037027, 23.95198897, 23.36624852, 22.7817436,
      29.98466705, 27.74775444, 26.70307122, 25.96644194, 25.32065037,
      24.70962891, 24.1135674,  23.52427179, 22.93809825, 22.35338321,
      29.97312226, 27.52792637, 26.3346648,  25.56052005, 24.90181737,
      24.28549826, 23.68708687, 23.09670875, 22.51002362, 21.92506064,
      29.96870043, 27.37312455, 25.98324711, 25.15962125, 24.48490972,
      23.86220014, 23.26098873, 22.66932799, 22.08203828, 21.49678269};

  //-----radiative cooling from Schure 2009, A&A 508, 751–757, not used -------
  static constexpr size_t len_rad_cool_ = 110;
  static constexpr Kokkos::Array<Real, len_rad_cool_> log_Trad_ = {
      3.80, 3.84, 3.88, 3.92, 3.96, 4.00, 4.04, 4.08, 4.12, 4.16, 4.20,
      4.24, 4.28, 4.32, 4.36, 4.40, 4.44, 4.48, 4.52, 4.56, 4.60, 4.64,
      4.68, 4.72, 4.76, 4.80, 4.84, 4.88, 4.92, 4.96, 5.00, 5.04, 5.08,
      5.12, 5.16, 5.20, 5.24, 5.28, 5.32, 5.36, 5.40, 5.44, 5.48, 5.52,
      5.56, 5.60, 5.64, 5.68, 5.72, 5.76, 5.80, 5.84, 5.88, 5.92, 5.96,
      6.00, 6.04, 6.08, 6.12, 6.16, 6.20, 6.24, 6.28, 6.32, 6.36, 6.40,
      6.44, 6.48, 6.52, 6.56, 6.60, 6.64, 6.68, 6.72, 6.76, 6.80, 6.84,
      6.88, 6.92, 6.96, 7.00, 7.04, 7.08, 7.12, 7.16, 7.20, 7.24, 7.28,
      7.32, 7.36, 7.40, 7.44, 7.48, 7.52, 7.56, 7.60, 7.64, 7.68, 7.72,
      7.76, 7.80, 7.84, 7.88, 7.92, 7.96, 8.00, 8.04, 8.08, 8.12, 8.16};
  static constexpr Kokkos::Array<Real, len_rad_cool_> log_gamma_H_He_ = {
      -30.61, -29.41, -28.46, -27.57, -26.38, -25.29, -24.27, -23.38, -22.60,
      -21.97, -21.61, -21.49, -21.52, -21.61, -21.71, -21.82, -21.91, -22.00,
      -22.08, -22.17, -22.25, -22.32, -22.36, -22.34, -22.25, -22.10, -21.95,
      -21.85, -21.82, -21.86, -21.94, -22.02, -22.11, -22.21, -22.29, -22.38,
      -22.46, -22.53, -22.60, -22.67, -22.73, -22.78, -22.84, -22.89, -22.94,
      -22.98, -23.01, -23.05, -23.07, -23.10, -23.12, -23.14, -23.16, -23.17,
      -23.19, -23.19, -23.20, -23.21, -23.21, -23.21, -23.21, -23.21, -23.20,
      -23.20, -23.19, -23.18, -23.18, -23.17, -23.15, -23.15, -23.13, -23.12,
      -23.11, -23.10, -23.08, -23.07, -23.06, -23.04, -23.03, -23.01, -22.99,
      -22.98, -22.96, -22.95, -22.93, -22.92, -22.90, -22.88, -22.86, -22.84,
      -22.83, -22.81, -22.79, -22.77, -22.76, -22.74, -22.72, -22.71, -22.69,
      -22.67, -22.65, -22.63, -22.61, -22.59, -22.57, -22.55, -22.53, -22.51,
      -22.49, -22.47};
  static constexpr Kokkos::Array<Real, len_rad_cool_> log_gamma_Z_ = {
      -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0,
      -100.0, -23.06, -22.73, -22.54, -22.44, -22.33, -22.17, -21.98, -21.77,
      -21.60, -21.45, -21.32, -21.20, -21.09, -21.00, -20.92, -20.85, -20.80,
      -20.76, -20.72, -20.70, -20.71, -20.73, -20.74, -20.74, -20.72, -20.70,
      -20.69, -20.68, -20.68, -20.68, -20.71, -20.81, -20.97, -21.16, -21.30,
      -21.39, -21.42, -21.44, -21.46, -21.52, -21.59, -21.64, -21.68, -21.69,
      -21.70, -21.72, -21.75, -21.76, -21.78, -21.79, -21.80, -21.84, -21.91,
      -22.00, -22.10, -22.20, -22.28, -22.34, -22.39, -22.42, -22.44, -22.44,
      -22.44, -22.44, -22.43, -22.42, -22.42, -22.43, -22.45, -22.47, -22.50,
      -22.53, -22.58, -22.63, -22.70, -22.77, -22.84, -22.90, -22.96, -23.01,
      -23.04, -23.07, -23.10, -23.12, -23.13, -23.15, -23.16, -23.14, -23.15,
      -23.15, -23.15, -23.15, -23.15, -23.15, -23.16, -23.16, -23.15, -23.16,
      -23.17, -23.17};

  // WD2001: Weingartner and Draine 2001, ApJ, 134:263-281
  // PE heating coefficients from WD2001 Table 2, second last line
  static constexpr Kokkos::Array<Real, 7> CPE_ = {
      5.22, 2.25, 0.04996, 0.00430, 0.147, 0.431, 0.692};
  // Collisional cooling included in PE heating, WD2001 Table3, second last line
  static constexpr Kokkos::Array<Real, 5> DPE_ = {0.4535, 2.234, -6.266, 1.442,
                                                  0.05089};
};
}  // namespace chemistry

#endif  // CHEMISTRY_THERMO_THERMO_HPP_
