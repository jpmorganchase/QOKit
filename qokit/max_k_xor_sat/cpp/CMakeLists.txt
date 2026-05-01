cmake_minimum_required(VERSION 3.16)
project(qaoa_contract CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP: optional on macOS (AppleClang doesn't ship it by default)
find_package(OpenMP QUIET)

# Float64 sources: aggressive optimization + FMA
set(F64_SOURCES
    primitives.cpp
    branch.cpp
    root.cpp
    contract.cpp
    grad.cpp
    alloc.cpp
)
add_library(qaoa_f64 OBJECT ${F64_SOURCES})
target_compile_options(qaoa_f64 PRIVATE -O3)
# -march=native may not work on all platforms; use if GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(qaoa_f64 PRIVATE -march=native)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(qaoa_f64 PRIVATE OpenMP::OpenMP_CXX)
endif()

# DD sources: strict IEEE 754 (no FMA for Dekker splitting correctness)
# Adjoint files instantiate both C128 and DDComplex templates, so they
# must compile with DD-safe flags to ensure correct Dekker splitting.
set(DD_SOURCES
    dd.cpp
    dd_fused.cpp
    adjoint.cpp
    adjoint_branch.cpp
    adjoint_root.cpp
    adjoint_primitives.cpp
)
add_library(qaoa_dd OBJECT ${DD_SOURCES})
target_compile_options(qaoa_dd PRIVATE -O3 -ffp-contract=off)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(qaoa_dd PRIVATE -march=native)
    # Disable FMA to ensure exact IEEE 754 rounding for Dekker splitting
    # Check if -mno-fma is supported (it's not on ARM macOS)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-mno-fma HAS_MNO_FMA)
    if(HAS_MNO_FMA)
        target_compile_options(qaoa_dd PRIVATE -mno-fma)
    endif()
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(qaoa_dd PRIVATE OpenMP::OpenMP_CXX)
endif()

# CLI executable
add_executable(qaoa_contract
    main.cpp
    $<TARGET_OBJECTS:qaoa_f64>
    $<TARGET_OBJECTS:qaoa_dd>
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qaoa_contract PRIVATE OpenMP::OpenMP_CXX)
endif()

# Shared library for Python ctypes binding (zero subprocess overhead)
# exports.cpp provides extern "C" wrappers to avoid C++ name mangling
add_library(qaoa_contract_shared SHARED
    exports.cpp
    $<TARGET_OBJECTS:qaoa_f64>
    $<TARGET_OBJECTS:qaoa_dd>
)
set_target_properties(qaoa_contract_shared PROPERTIES
    OUTPUT_NAME qaoa_contract
    POSITION_INDEPENDENT_CODE ON
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qaoa_contract_shared PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ensure OBJECT libraries are compiled with -fPIC for the shared lib
set_target_properties(qaoa_f64 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(qaoa_dd PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Recommended runtime environment for 91-core multi-socket systems:
#   export OMP_NUM_THREADS=91
#   export OMP_PROC_BIND=spread     # spread threads across NUMA nodes
#   export OMP_PLACES=cores         # one thread per physical core
#   export OMP_WAIT_POLICY=active   # busy-wait for low latency

# Include nlohmann/json if available, otherwise use built-in minimal parser
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(qaoa_contract PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(qaoa_contract PRIVATE HAS_NLOHMANN_JSON)
endif()
