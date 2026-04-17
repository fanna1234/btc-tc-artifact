cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES "120")

project(cusparse_tc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)


add_executable(cusparse_tc_lxlt cusparse_tc_lxlt.cu)
target_link_libraries(cusparse_tc_lxlt PRIVATE
    CUDA::cusparse
    CUDA::cudart
    OpenMP::OpenMP_CXX
    btc_reorder
)

target_compile_options(cusparse_tc_lxlt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

if(OpenMP_CXX_FOUND)
    target_compile_options(cusparse_tc_lxlt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    target_compile_options(cusparse_tc_lxlt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
endif()

# Naive A^3 SpGEMM Baseline
add_executable(cusparse_A3_tc cusparse_A3_tc.cu)
target_link_libraries(cusparse_A3_tc PRIVATE 
    CUDA::cusparse 
    CUDA::cudart
    OpenMP::OpenMP_CXX
)
target_include_directories(cusparse_A3_tc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../" "${CMAKE_CURRENT_SOURCE_DIR}/../../btc")

if(OpenMP_CXX_FOUND)
    target_compile_options(cusparse_A3_tc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    target_compile_options(cusparse_A3_tc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
endif()

target_include_directories(cusparse_tc_lxlt PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../btc
)

# Set output directory to simulate consistent structure

set_target_properties(cusparse_tc_lxlt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/baselines/cusparse_tc/apps"
)
