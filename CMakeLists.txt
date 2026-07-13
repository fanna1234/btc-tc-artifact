cmake_minimum_required(VERSION 3.22)
project(btc_tc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # "native" requires CMake >= 3.24. Fall back to auto-detect via nvidia-smi.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    execute_process(
      COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _gpu_cc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_gpu_cc)
      string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _m "${_gpu_cc}")
      if(_m)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        message(STATUS "Auto-detected GPU arch: sm_${CMAKE_CUDA_ARCHITECTURES}")
      endif()
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 80)
      message(STATUS "Defaulting to sm_80 (Ampere)")
    endif()
  endif()
endif()

# Prefer latest CUDA toolchain when available (avoids older NVCC defaults such as sm_52).
# Try 13.3, 13.2 (paper toolkit), 13.1, 13.0, 12.9, 12.8 in order
# Adaptive: probe versioned toolkits newest-first, then the generic /usr/local/cuda
# symlink, then any nvcc on PATH. Supports CUDA 12.1 through 13.3+ so a reviewer with
# 12.x (e.g. Chameleon's CC-Ubuntu-CUDA = 12.6) builds without hand-setting a path.
set(_btc_cuda_vers "13.3" "13.2" "13.1" "13.0" "12.9" "12.8" "12.6" "12.5" "12.4" "12.3" "12.2" "12.1")
foreach(_cuda_ver ${_btc_cuda_vers})
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-${_cuda_ver}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${_cuda_ver}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    message(STATUS "Using CUDA ${_cuda_ver}")
    break()
  endif()
endforeach()
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  message(STATUS "Using CUDA at /usr/local/cuda (generic symlink)")
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(_BTC_NVCC nvcc)
  if(_BTC_NVCC)
    set(CMAKE_CUDA_COMPILER "${_BTC_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
    message(STATUS "Using nvcc from PATH: ${_BTC_NVCC}")
  endif()
endif()
foreach(_cuda_ver ${_btc_cuda_vers})
  if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda-${_cuda_ver}")
    set(CUDAToolkit_ROOT "/usr/local/cuda-${_cuda_ver}" CACHE PATH "CUDA toolkit root" FORCE)
    break()
  endif()
endforeach()
if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda")
  set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root" FORCE)
endif()

# CMake's CUDA compiler identification step may invoke NVCC without an explicit
# GPU arch, and some toolchains default to deprecated targets (e.g. sm_52),
# causing configuration to fail before it even reaches your requested arch.
# Provide a safe arch for the identification step via CUDAFLAGS if the user
# hasn't set it already.
set(_btc_restore_cudaflags FALSE)
if(NOT DEFINED ENV{CUDAFLAGS} AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  set(_btc_cuda_arch_list ${CMAKE_CUDA_ARCHITECTURES})
  list(GET _btc_cuda_arch_list 0 _btc_cuda_arch_for_id)
  if(_btc_cuda_arch_for_id MATCHES "^[0-9]+$")
    set(ENV{CUDAFLAGS} "-arch=sm_${_btc_cuda_arch_for_id}")
    set(_btc_restore_cudaflags TRUE)
  endif()
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(_btc_restore_cudaflags)
  unset(ENV{CUDAFLAGS})
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Guard: if conda injects a sysroot librt, drop it (glibc 2.34+ has rt in libc).
if(CUDAToolkit_rt_LIBRARY AND CUDAToolkit_rt_LIBRARY MATCHES "conda")
  unset(CUDAToolkit_rt_LIBRARY CACHE)
endif()

# Rabbit Order + Gorder reordering libraries (CPU-only, compiled by host CXX compiler)
add_library(btc_reorder STATIC
  btc/operations/reorder_rabbit.cpp
  btc/operations/reorder_gorder.cpp
  btc/operations/reorder_hashorder.cpp
  btc/operations/reorder_rcm.cpp
)
target_include_directories(btc_reorder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(btc_reorder PRIVATE -O3 -fopenmp -DNDEBUG)
target_link_libraries(btc_reorder PRIVATE numa OpenMP::OpenMP_CXX)

add_library(btclib INTERFACE)
target_include_directories(btclib
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/btc>
)
target_link_libraries(btclib INTERFACE OpenMP::OpenMP_CXX btc_reorder)
target_compile_options(btclib INTERFACE
  $<$<CONFIG:Debug>:-G -g>
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-fopenmp
    --extended-lambda
    --expt-relaxed-constexpr
  >
)

add_subdirectory(apps)

option(BTC_BUILD_BASELINES "Build baseline implementations" ON)
if(BTC_BUILD_BASELINES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/baselines/CMakeLists.txt")
  add_subdirectory(baselines)
elseif(BTC_BUILD_BASELINES)
  message(STATUS "Baselines directory has no CMakeLists.txt, skipping")
endif()
