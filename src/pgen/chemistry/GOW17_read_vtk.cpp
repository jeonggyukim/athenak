//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file GOW17_read_vtk.cpp
//! \brief Initial state read from a legacy VTK file, for the GOW17 network.
//!
//! The file is the Athena 4.2 flavour of legacy VTK that Athena++'s `read_vtk`
//! problem generator consumes: BINARY, DATASET STRUCTURED_POINTS, CELL_DATA,
//! big-endian float32, with `SCALARS density`, `SCALARS pressure` and
//! `VECTORS velocity`. Both codes reading the same file is what makes a
//! cell-by-cell comparison between the semi-implicit method and CVODE possible:
//! the turbulence drivers differ between the codes, so a shared realization can
//! only come from a shared file, not from shared input parameters.
//!
//! Abundances start uniform at `<problem> r_init`, matching Athena++'s reader,
//! so the two runs also share their chemical initial condition.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \fn SwapFloat
//! \brief Legacy VTK is big-endian; every float in the file needs its bytes
//! reversed on the little-endian machines this code runs on.

float SwapFloat(float value) {
  char bytes[4];
  std::memcpy(bytes, &value, 4);
  char swapped[4] = {bytes[3], bytes[2], bytes[1], bytes[0]};
  float out;
  std::memcpy(&out, swapped, 4);
  return out;
}

//----------------------------------------------------------------------------------------
//! \struct VTKField
//! \brief One named field of the file, flattened in VTK order (i fastest).

struct VTKField {
  std::string name;
  int ncomp = 1;
  std::vector<float> data;
};

//----------------------------------------------------------------------------------------
//! \fn AbortVTK
//! \brief A malformed file is a fatal error: continuing would run a simulation
//! on whatever happened to be in memory.

void AbortVTK(const std::string &fname, const std::string &what) {
  std::cout << "### FATAL ERROR in GOW17_read_vtk: " << what << " in file '"
            << fname << "'" << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn ReadVTKFile
//! \brief Parse the header and every SCALARS/VECTORS block into `fields`, and
//! return the cell counts through nx1, nx2, nx3.

void ReadVTKFile(const std::string &fname, int *nx1, int *nx2, int *nx3,
                 std::vector<VTKField> *fields) {
  std::ifstream file(fname, std::ios::binary);
  if (!file.is_open()) {
    AbortVTK(fname, "cannot open");
  }

  // Header lines run up to CELL_DATA; DIMENSIONS counts points, so each cell
  // count is one less than the corresponding entry.
  std::int64_t ncells = 0;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string key;
    iss >> key;
    if (key == "DIMENSIONS") {
      int d1, d2, d3;
      iss >> d1 >> d2 >> d3;
      *nx1 = d1 - 1;
      *nx2 = d2 - 1;
      *nx3 = d3 - 1;
    } else if (key == "CELL_DATA") {
      iss >> ncells;
      break;
    }
  }
  if (ncells <= 0) {
    AbortVTK(fname, "no CELL_DATA record");
  }
  if (ncells != static_cast<std::int64_t>(*nx1) * (*nx2) * (*nx3)) {
    AbortVTK(fname, "CELL_DATA disagrees with DIMENSIONS");
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string key, name, type;
    iss >> key >> name >> type;
    if (key != "SCALARS" && key != "VECTORS") continue;

    VTKField field;
    field.name = name;
    field.ncomp = (key == "VECTORS") ? 3 : 1;
    if (key == "SCALARS") {
      // The LOOKUP_TABLE line sits between the SCALARS line and its data.
      std::getline(file, line);
    }
    field.data.resize(field.ncomp * ncells);
    file.read(reinterpret_cast<char *>(field.data.data()),
              field.data.size() * sizeof(float));
    if (!file) {
      AbortVTK(fname, "truncated data for field '" + name + "'");
    }
    for (std::size_t n = 0; n < field.data.size(); ++n) {
      field.data[n] = SwapFloat(field.data[n]);
    }
    fields->push_back(field);
    // A trailing newline separates this block from the next header line.
    if (file.peek() == '\n') file.get();
  }
}

//----------------------------------------------------------------------------------------
//! \fn FindField
//! \brief Locate a field by name, aborting when the file lacks it.

const VTKField &FindField(const std::vector<VTKField> &fields,
                          const std::string &name, const std::string &fname) {
  for (const auto &field : fields) {
    if (field.name == name) return field;
  }
  AbortVTK(fname, "no field named '" + name + "'");
  return fields[0];  // unreachable; silences a compiler warning
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GOW17ReadVTK()
//! \brief Fill the primitives from a legacy VTK file and set uniform abundances.

void ProblemGenerator::GOW17ReadVTK(ParameterInput *pin, const bool restart) {
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;

  const std::string fname = pin->GetString("problem", "vtkfile");
  int fnx1 = 0, fnx2 = 0, fnx3 = 0;
  std::vector<VTKField> fields;
  ReadVTKFile(fname, &fnx1, &fnx2, &fnx3, &fields);

  // The file has to cover the whole mesh at the mesh's own resolution: this
  // pgen places cells by index, so a mismatch would silently shift the field.
  auto &mindcs = pmy_mesh_->mesh_indcs;
  if (fnx1 != mindcs.nx1 || fnx2 != mindcs.nx2 || fnx3 != mindcs.nx3) {
    std::cout << "### FATAL ERROR in GOW17_read_vtk: file grid " << fnx1 << "x"
              << fnx2 << "x" << fnx3 << " does not match mesh " << mindcs.nx1
              << "x" << mindcs.nx2 << "x" << mindcs.nx3 << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const VTKField &dens_f = FindField(fields, "density", fname);
  const VTKField &pres_f = FindField(fields, "pressure", fname);
  const VTKField &vel_f = FindField(fields, "velocity", fname);

  const bool chemistry_on = (pmbp->pchemistry != nullptr);
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  // Copy the file into a device array laid out per MeshBlock. Each block takes
  // the slab of the global grid its lower corner points at.
  const int nmb = pmbp->nmb_thispack;
  const int nk = indcs.nx3;
  const int nj = indcs.nx2;
  const int ni = indcs.nx1;
  DualArray5D<Real> fld("vtk_field", nmb, 5, nk, nj, ni);

  auto &size = pmbp->pmb->mb_size;
  const Real x1min = pmy_mesh_->mesh_size.x1min;
  const Real x2min = pmy_mesh_->mesh_size.x2min;
  const Real x3min = pmy_mesh_->mesh_size.x3min;

  for (int m = 0; m < nmb; ++m) {
    const int i0 = static_cast<int>(
        std::lround((size.h_view(m).x1min - x1min) / size.h_view(m).dx1));
    const int j0 = static_cast<int>(
        std::lround((size.h_view(m).x2min - x2min) / size.h_view(m).dx2));
    const int k0 = static_cast<int>(
        std::lround((size.h_view(m).x3min - x3min) / size.h_view(m).dx3));

    for (int k = 0; k < nk; ++k) {
      for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
          const std::int64_t idx =
              (static_cast<std::int64_t>(k0 + k) * fnx2 + (j0 + j)) * fnx1 +
              (i0 + i);
          fld.h_view(m, 0, k, j, i) = static_cast<Real>(dens_f.data[idx]);
          fld.h_view(m, 1, k, j, i) = static_cast<Real>(vel_f.data[3 * idx]);
          fld.h_view(m, 2, k, j, i) = static_cast<Real>(vel_f.data[3 * idx + 1]);
          fld.h_view(m, 3, k, j, i) = static_cast<Real>(vel_f.data[3 * idx + 2]);
          fld.h_view(m, 4, k, j, i) = static_cast<Real>(pres_f.data[idx]) / gm1;
        }
      }
    }
  }
  fld.template modify<HostMemSpace>();
  fld.template sync<DevMemSpace>();
  auto fld_d = fld.d_view;

  DualArray1D<Real> initial_chemistry("initial_chemistry",
                                      chemistry::GOW17Network::neqs - 1);
  if (chemistry_on) {
    // Athena++'s read_vtk calls this r_init; the per-species keys of the other
    // GOW17 problems still override it, so the two readers agree by default.
    const Real r_init = pin->GetOrAddReal("problem", "r_init", 0.0);
    for (size_t i = 0; i < chemistry::GOW17Network::neqs - 1; i++) {
      const auto name = chemistry::GOW17Network::species_names[i];
      const auto init_name = std::string("init_") + std::string(name);
      initial_chemistry.view_host()(i) =
          pin->GetOrAddReal("problem", init_name, r_init);
    }
  }
  initial_chemistry.modify_host();
  initial_chemistry.sync_device();
  auto initial_chemistry_d = initial_chemistry.view_device();

  const int chem_start =
      chemistry_on ? pmbp->pchemistry->get_chemistry_scalars_first_idx() : 0;

  par_for(
      "pgen_GOW17_read_vtk", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const int kf = k - ks;
        const int jf = j - js;
        const int if_ = i - is;
        w0(m, IDN, k, j, i) = fld_d(m, 0, kf, jf, if_);
        w0(m, IVX, k, j, i) = fld_d(m, 1, kf, jf, if_);
        w0(m, IVY, k, j, i) = fld_d(m, 2, kf, jf, if_);
        w0(m, IVZ, k, j, i) = fld_d(m, 3, kf, jf, if_);
        w0(m, IEN, k, j, i) = fld_d(m, 4, kf, jf, if_);

        if (chemistry_on) {
          for (size_t s = 0; s < chemistry::GOW17Network::neqs - 1; s++) {
            w0(m, chem_start + s, k, j, i) = initial_chemistry_d(s);
          }
        }
      });

  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
