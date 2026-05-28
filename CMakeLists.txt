cmake_minimum_required(VERSION 3.20)
project(ifgf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(SYCL_PARAMS "-fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86")
set(SYCL_PARAMS "-fsycl-targets=spir64")

add_compile_options(
  "$<$<CONFIG:Release>:-O3;-march=native;-funroll-loops;-fsycl;${SYCL_PARAMS}>"
  "$<$<CONFIG:RelWithDebInfo>:-O2;-g>"
  "$<$<CONFIG:Debug>:-O0;-g>"
)

add_compile_definitions(
  EIGEN_FAST_MATH=1
  EIGEN_DONT_PARALLELIZE
)

find_package(Eigen3 CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(IntelSYCL REQUIRED)

add_library(ifgf_core SHARED)

target_sources(ifgf_core
  PRIVATE
    core/ifgf_library.cpp
)


target_include_directories(ifgf_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ifgf>
)

target_link_libraries(ifgf_core
  PUBLIC Eigen3::Eigen
  PRIVATE TBB::tbb
)

add_sycl_to_target(
  TARGET ifgf_core
  SOURCES core/ifgf_library.cpp
)

set_target_properties(ifgf_core PROPERTIES
  INTERFACE_LINK_OPTIONS ""
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ifgf_core
  EXPORT ifgfTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# *** ADDED: install the public header ***
install(FILES
  core/ifgf_library.hpp
  core/config.hpp
  core/unordered_dense.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ifgf
)

install(EXPORT ifgfTargets
  FILE ifgfTargets.cmake
  NAMESPACE ifgf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ifgf
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ifgfConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/ifgfConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/ifgfConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ifgf
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ifgfConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ifgfConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ifgf
)

option(IFGF_BUILD_TESTS "Build ifgf tests" ON)

if(IFGF_BUILD_TESTS)
  add_executable(ifgf_test ./test/ifgf_test.cpp)
  target_link_libraries(ifgf_test
  PUBLIC Eigen3::Eigen
  PRIVATE TBB::tbb
  )
  target_link_libraries(ifgf_test PRIVATE ifgf_core)
  
  add_sycl_to_target(
    TARGET ifgf_test
    SOURCES ./test/ifgf_test.cpp
  )
endif()