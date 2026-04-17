cmake_minimum_required(VERSION 3.22)

# Set architecture for Tensor Cores
set(CMAKE_CUDA_ARCHITECTURES "90")

project(cublas_tc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cublas_tc cublas_tc.cu)
target_link_libraries(cublas_tc PRIVATE cublas)

target_compile_options(cublas_tc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(cublas_A3_tc cublas_A3_tc.cu)
target_link_libraries(cublas_A3_tc PRIVATE cublas)

target_include_directories(cublas_tc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../" "${CMAKE_CURRENT_SOURCE_DIR}/../../btc")
target_include_directories(cublas_A3_tc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../" "${CMAKE_CURRENT_SOURCE_DIR}/../../btc")


target_link_libraries(cublas_tc PRIVATE
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
    btc_reorder
)

target_link_libraries(cublas_A3_tc PRIVATE
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
    btc_reorder
)

if(OpenMP_CXX_FOUND)

    target_compile_options(cublas_tc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    # Also add standard flags if CMake doesn't add them automatically for CUDA host code
    target_compile_options(cublas_tc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)

    target_compile_options(cublas_A3_tc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    target_compile_options(cublas_A3_tc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
endif()

target_include_directories(cublas_tc PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../btc
)

# Set output directory to simulate consistent structure
set_target_properties(cublas_tc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/baselines/cublas_tc/apps"
)

set_target_properties(cublas_A3_tc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/baselines/cublas_tc/apps"
)
