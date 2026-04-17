cmake_minimum_required(VERSION 3.22)

# Baseline implementations (for benchmarking comparisons).
# Add new baselines as subdirectories under `baselines/`.

add_subdirectory(ToT-TPDS25)
add_subdirectory(cusparse_tc)
add_subdirectory(cublas_tc)

# TC-Compare preprocessing tools (for Green, HIndex, GroupTC baselines)
add_subdirectory(TC-Compare)

# Official TRUST baseline (TPDS 2021).
# TRUST is built via its original scripts for baseline-faithfulness (see scripts/build_trust.sh).
