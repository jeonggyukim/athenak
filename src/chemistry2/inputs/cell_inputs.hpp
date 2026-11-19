#ifndef CHEMISTRY2_INPUTS_CELL_INPUTS_HPP_
#define CHEMISTRY2_INPUTS_CELL_INPUTS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cell_inputs.hpp
//! \brief Quantities from other physics a chemical network reads in one cell, and the
//!        source that fills them.
//!
//!   CellInputs is named by physical quantity, not by the module that produces it. An
//!   InputSource fills it inside the chemistry kernel from data the producing module
//!   already owns; this version has one source, constants from the input file.

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"

namespace chemistry2 {

struct CellInputs {
  Real g0;      // FUV field [Draine 1978 units]
  Real xi_cr;   // cosmic-ray ionization rate per H [s^-1]
  Real grad_v;  // |grad v| for the LVG escape of CO lines [s^-1]
};

struct InputSource {
  Real g0;
  Real xi_cr;

  //! Same keys and defaults as <chemistry>
  static InputSource Read(ParameterInput *pin, const std::string &block) {
    return InputSource{pin->GetOrAddReal(block, "radiation_G0", 1.0e-6),
                       pin->GetOrAddReal(block, "radiation_CR", 2.0e-16)};
  }

  KOKKOS_INLINE_FUNCTION CellInputs Load(int, int, int, int) const {
    return CellInputs{g0, xi_cr, 0.0};
  }
};

}  // namespace chemistry2

#endif  // CHEMISTRY2_INPUTS_CELL_INPUTS_HPP_
