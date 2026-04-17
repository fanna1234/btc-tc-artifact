# Paper applications (11 targets)

# Main executables
add_executable(btc_tc_lite btc_tc_lite.cu)
target_link_libraries(btc_tc_lite PRIVATE btclib)

add_executable(btc_tc_adaptive_16x128 btc_tc_adaptive_16x128.cu)
target_link_libraries(btc_tc_adaptive_16x128 PRIVATE btclib)

add_executable(btc_tc_adaptive_16x32 btc_tc_adaptive_16x32.cu)
target_link_libraries(btc_tc_adaptive_16x32 PRIVATE btclib)

# Ablation variants (Fig 9a)
add_executable(btc_tc_v3 btc_tc_v3.cu)
target_link_libraries(btc_tc_v3 PRIVATE btclib)

add_executable(btc_tc_v3_16x32 btc_tc_v3_16x32.cu)
target_link_libraries(btc_tc_v3_16x32 PRIVATE btclib)

add_executable(btc_tc_v5_16x128 btc_tc_v5_16x128.cu)
target_link_libraries(btc_tc_v5_16x128 PRIVATE btclib)

add_executable(btc_tc_v6_16x32 btc_tc_v6_16x32.cu)
target_link_libraries(btc_tc_v6_16x32 PRIVATE btclib)

# Experiments
add_executable(btc_blocksize_bench btc_blocksize_bench.cu)
target_link_libraries(btc_blocksize_bench PRIVATE btclib)

add_executable(tau_sweep tau_sweep.cu)
target_link_libraries(tau_sweep PRIVATE btclib)

add_executable(hybrid_ablation hybrid_ablation.cu)
target_link_libraries(hybrid_ablation PRIVATE btclib)

# Utility (baseline format conversion)
add_executable(mtx_to_tc_compare mtx_to_tc_compare.cu)
target_link_libraries(mtx_to_tc_compare PRIVATE btclib)
