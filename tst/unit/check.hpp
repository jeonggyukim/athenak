#ifndef TST_UNIT_CHECK_HPP_
#define TST_UNIT_CHECK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file check.hpp
//! \brief Checks for the programs in tst/unit.
//!
//!   A unit test records its checks in a unit::Report and returns report.ExitCode() from
//!   main(); ctest counts a nonzero exit as a failed test.

#include <cmath>
#include <cstdio>
#include <string_view>

namespace unit {

class Report {
 public:
  bool Expect(bool ok, std::string_view what) {
    Record(ok);
    if (!ok) std::printf("  FAIL  %.*s\n", static_cast<int>(what.size()), what.data());
    return ok;
  }

  //! Passes when |got - want| <= rtol*|want| + atol
  bool ExpectNear(double got, double want, double rtol, double atol,
                  std::string_view what) {
    const double bound = rtol*std::fabs(want) + atol;
    const double diff = std::fabs(got - want);
    const bool ok = std::isfinite(got) && diff <= bound;
    Record(ok);
    std::printf("  %s  %.*s: got %.9e, want %.9e, diff %.2e, bound %.2e\n",
                ok ? "pass" : "FAIL", static_cast<int>(what.size()), what.data(), got,
                want, diff, bound);
    return ok;
  }

  int ExitCode() const {
    std::printf("%d checks, %d failed\n", n_checks_, n_failed_);
    return n_failed_ == 0 ? 0 : 1;
  }

 private:
  void Record(bool ok) {
    ++n_checks_;
    if (!ok) ++n_failed_;
  }

  int n_checks_ = 0;
  int n_failed_ = 0;
};

}  // namespace unit

#endif  // TST_UNIT_CHECK_HPP_
