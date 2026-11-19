# Unit tests: one executable per tst/unit/<module>/test_<name>.cpp, run by
#   ctest -L unit
# from the build directory. A test calls src/ code directly; no simulation runs.

file(GLOB_RECURSE UNIT_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/test_*.cpp)

# Benchmarks (bench_*.cpp) are built but not registered with ctest; run them by hand.
file(GLOB_RECURSE UNIT_BENCHES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/bench_*.cpp)
foreach(src ${UNIT_BENCHES})
  get_filename_component(dir ${src} DIRECTORY)
  get_filename_component(module ${dir} NAME)
  get_filename_component(name ${src} NAME_WE)
  set(target ${module}_${name})
  add_executable(${target} ${src})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR})
  target_link_libraries(${target} PRIVATE Kokkos::kokkos Kokkos::kokkoskernels
                        Threads::Threads)
  if (ENABLE_MPI)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
  endif()
  if (ENABLE_OPENMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()

foreach(src ${UNIT_TESTS})
  get_filename_component(dir ${src} DIRECTORY)
  get_filename_component(module ${dir} NAME)
  get_filename_component(name ${src} NAME_WE)
  set(target unit_${module}_${name})
  add_executable(${target} ${src})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR})
  target_link_libraries(${target} PRIVATE Kokkos::kokkos Kokkos::kokkoskernels
                        Threads::Threads)
  if (ENABLE_MPI)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
  endif()
  if (ENABLE_OPENMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  # Host and device results are compared bit for bit: no fused multiply-adds on either
  # side (nvcc_wrapper passes -ffp-contract=off to the host compiler).
  target_compile_options(${target} PRIVATE -ffp-contract=off)
  if (Kokkos_ENABLE_CUDA)
    target_compile_options(${target} PRIVATE -fmad=false)
  endif()
  add_test(NAME ${module}/${name} COMMAND ${target})
  set_tests_properties(${module}/${name} PROPERTIES LABELS unit TIMEOUT 60)
endforeach()
