# cuBLAS triangle counting baseline (inherits CUDA from parent project)

add_executable(cublas_tc cublas_tc.cu)
target_link_libraries(cublas_tc PRIVATE
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
    btc_reorder
)
target_compile_options(cublas_tc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-fopenmp>
)
target_include_directories(cublas_tc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../btc
)
set_target_properties(cublas_tc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/baselines/cublas_tc/apps"
)

# Naive A^3 SpGEMM baseline
add_executable(cublas_A3_tc cublas_A3_tc.cu)
target_link_libraries(cublas_A3_tc PRIVATE
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
    btc_reorder
)
target_compile_options(cublas_A3_tc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-fopenmp>
)
target_include_directories(cublas_A3_tc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../btc
)
set_target_properties(cublas_A3_tc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/baselines/cublas_tc/apps"
)
