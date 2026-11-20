# Tools: executables that call src/ code directly without a mesh, built with the unit
# tests. tst/tools/<module>/<name>.cpp gives the target <module>_<name>.

file(GLOB_RECURSE TOOLS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp)
foreach(src ${TOOLS})
  get_filename_component(dir ${src} DIRECTORY)
  get_filename_component(module ${dir} NAME)
  get_filename_component(name ${src} NAME_WE)
  set(target ${module}_${name})
  # ParameterInput, so a tool reads an athinput file as athena does
  add_executable(${target} ${src}
    ${PROJECT_SOURCE_DIR}/src/parameter_input.cpp
    ${PROJECT_SOURCE_DIR}/src/globals.cpp
    ${PROJECT_SOURCE_DIR}/src/outputs/io_wrapper.cpp)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR})
  target_compile_definitions(${target} PRIVATE
    ATHENAK_SOURCE_DIR="${PROJECT_SOURCE_DIR}/")
  target_link_libraries(${target} PRIVATE Kokkos::kokkos Kokkos::kokkoskernels
                        Threads::Threads)
  if (ENABLE_MPI)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
  endif()
  if (ENABLE_OPENMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()
