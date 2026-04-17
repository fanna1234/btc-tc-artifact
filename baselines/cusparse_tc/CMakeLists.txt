# cuSPARSE triangle counting baseline (inherits CUDA from parent project)

add_executable(cusparse_tc_lxlt cusparse_tc_lxlt.cu)
target_link_libraries(cusparse_tc_lxlt PRIVATE
    CUDA::cusparse
    CUDA::cudart
    OpenMP::OpenMP_CXX
    btc_reorder
)
target_compile_options(cusparse_tc_lxlt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-fopenmp>
)
target_include_directories(cusparse_tc_lxlt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../btc
)
set_target_properties(cusparse_tc_lxlt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/baselines/cusparse_tc/apps"
)

# Naive A^3 SpGEMM baseline
add_executable(cusparse_A3_tc cusparse_A3_tc.cu)
target_link_libraries(cusparse_A3_tc PRIVATE
    CUDA::cusparse
    CUDA::cudart
    OpenMP::OpenMP_CXX
    btc_reorder
)
target_compile_options(cusparse_A3_tc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-fopenmp>
)
target_include_directories(cusparse_A3_tc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../btc
)
