
if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()


cmake_minimum_required(VERSION 3.4...3.18)
project(ifgf CXX)
#


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(SYCL_PARAMS "-fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86")
set(SYCL_PARAMS "-fsycl-targets=spir64")
string(APPEND CMAKE_CXX_FLAGS_RELEASE "-march=native -Ofast -O3 -funroll-loops  -UUSE_NGSOLVE -flto=auto -fsycl ${SYCL_PARAMS}")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2  -g -flto=auto -O  -fsycl ${SYCL_PARAMS}")
#string(APPEND CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsycl ${SYCL_PARAMS}")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O2 -g -O -fsycl ${SYCL_PARAMS} -D_GLIBCXX_ASSERTIONS=0")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address -fno-omit-frame-pointer")
string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG " -fsanitize=address")
#add_definitions(-DEIGEN_FAST_MATH=1 -DEIGEN_DONT_PARALLELIZE   )

# find_package(unordered_dense CONFIG )



find_package (Eigen3 CONFIG REQUIRED)
find_package (TBB REQUIRED)
find_package(IntelSYCL REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR} ${TBB_INCLUDE_DIR})

# find_package(pybind11 REQUIRED)
# pybind11_add_module(pyifgf SHARED py_ifgf.cpp )
# target_link_libraries (pyifgf PRIVATE tbb)





set(TARGET_NAME "test_ifgf")
set(SOURCE_FILE test_ifgf.cpp)



add_executable(${TARGET_NAME} ${SOURCE_FILE})
add_sycl_to_target(TARGET ${TARGET_NAME}  SOURCES ${SOURCE_FILE} )
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)

target_link_libraries(${TARGET_NAME} PRIVATE
    TBB::tbb
    Eigen3::Eigen
)


add_library(ifgflib SHARED
    ifgf_library.cpp)
add_sycl_to_target(TARGET ifgflib ifgf_library.cpp)
set_property(TARGET ifgflib PROPERTY CXX_STANDARD 20)
target_link_libraries (ifgflib PRIVATE tbb)
target_link_libraries(ifgflib PRIVATE
    TBB::tbb
    Eigen3::Eigen
)
