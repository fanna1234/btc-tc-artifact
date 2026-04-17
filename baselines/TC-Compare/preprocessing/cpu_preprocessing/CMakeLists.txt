cmake_minimum_required(VERSION 3.22)

# TC-Compare preprocessing tools for converting between graph formats
# These tools are needed for Green, HIndex, and GroupTC baselines

# XXX2CSR tools: Convert various formats to CSR
add_executable(MTX2CSR
    XXX2CSR/MTX2CSR.cpp
)
target_compile_options(MTX2CSR PRIVATE -O3)

add_executable(SNAP2CSR
    XXX2CSR/SNAP2CSR.cpp
)
target_compile_options(SNAP2CSR PRIVATE -O3)

# CSR2XXX tools: Convert CSR to various formats
add_executable(CSR2DCSR
    CSR2XXX/CSR2DCSR.cpp
)
target_compile_options(CSR2DCSR PRIVATE -O3)

add_executable(CSR2RidDCSR
    CSR2XXX/CSR2RidDCSR.cpp
)
target_compile_options(CSR2RidDCSR PRIVATE -O3)

add_executable(CSR2HuEdgeList
    CSR2XXX/CSR2HuEdgeList.cpp
)
target_compile_options(CSR2HuEdgeList PRIVATE -O3)

add_executable(CSR2PolakEdgeList
    CSR2XXX/CSR2PolakEdgeList.cpp
)
target_compile_options(CSR2PolakEdgeList PRIVATE -O3)

add_executable(CSR2TrustCSR
    CSR2XXX/CSR2TrustCSR.cpp
)
target_compile_options(CSR2TrustCSR PRIVATE -O3)

add_executable(CSR2TrustSortCSR
    CSR2XXX/CSR2TrustSortCSR.cpp
)
target_compile_options(CSR2TrustSortCSR PRIVATE -O3)

# Install targets to build directory for easy access
install(TARGETS
    MTX2CSR SNAP2CSR
    CSR2DCSR CSR2RidDCSR CSR2HuEdgeList CSR2PolakEdgeList
    CSR2TrustCSR CSR2TrustSortCSR
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/preprocessing_tools
)
