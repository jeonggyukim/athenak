#ifndef CHEMISTRY2_DATA_PATH_HPP_
#define CHEMISTRY2_DATA_PATH_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file data_path.hpp
//! \brief Default path of a data file under the source tree (rel, e.g.
//!        "inputs/chemistry2/ions/"), so that a run finds it from any working
//!        directory. ATHENAK_SOURCE_DIR is set by src/CMakeLists.txt; without it the
//!        path stays relative.

#include <string>

namespace chemistry2 {

inline std::string DataPath(const std::string &rel) {
#ifdef ATHENAK_SOURCE_DIR
  return std::string(ATHENAK_SOURCE_DIR) + rel;
#else
  return rel;
#endif
}

}  // namespace chemistry2

#endif  // CHEMISTRY2_DATA_PATH_HPP_
