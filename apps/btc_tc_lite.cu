#include <btc/btc.h>
#include <btc/operations/tc_16x32_adaptive.h>
#include <btc/operations/tc_16x128_adaptive.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main(int argc, char** argv)
{
    btc::CsrMatrix<int, float, btc::device_memory> A_csr;
    btc::CooMatrix<int, float, btc::device_memory> A_coo;

    btc::Config config = btc::program_options(argc, argv);

    // Parse manual mode override via Env Var: BTC_MODE=0,1,2
    int mode = 0;
    const char* env_mode = std::getenv("BTC_MODE");
    if (env_mode) {
        mode = std::atoi(env_mode);
    }

    // Read Input
    fprintf(stderr, "[DBG] Reading MTX file...\n");
    btc::read_matrix_file(A_csr, config.input_file);
    fprintf(stderr, "[DBG] CSR: rows=%d, entries=%d\n", A_csr.num_rows, A_csr.num_entries);
    btc::convert_csr_to_coo(A_coo, A_csr);
    fprintf(stderr, "[DBG] COO ready, entries=%d\n", A_coo.num_entries);
    A_csr.free();
    fprintf(stderr, "[DBG] CSR freed\n");

    btc::CUDATimer timer;

    // Preprocessing (Sorting & Lower Triangular)
    timer.start();
    fprintf(stderr, "[DBG] Starting preprocess...\n");
    btc::preprocess_for_triangle_counting(A_coo, false, true, config.input_file);
    timer.stop();
    fprintf(stderr, "[DBG] Preprocess done, entries=%d\n", A_coo.num_entries);
    float t_pre = timer.elapsed();

    std::printf("\n--------------btc Lite-Adaptive (O(1) Heuristic)----------------\n");

    // 1. O(1) Heuristic Analysis
    float avg_degree = (float)A_coo.num_entries / (float)A_coo.num_rows;
    std::printf("Stats: Rows=%d, NNZ=%zu, AvgDegree=%.2f\n", A_coo.num_rows, A_coo.num_entries, avg_degree);

    // 2. Select Strategy
    bool use_128 = true;

    if (mode == 1) {
        use_128 = false;
        std::printf("Decision: USE 32 (Reason: Forced Mode)\n");
    } else if (mode == 2) {
        use_128 = true;
        std::printf("Decision: USE 128 (Reason: Forced Mode)\n");
    } else {
        if (avg_degree < 15.0f || A_coo.num_rows < 50000) {
            use_128 = false;
            std::printf("Decision: USE 32 (Reason: Lite V3 - Deg < 15 or Rows < 50k)\n");
        } else {
            use_128 = true;
            std::printf("Decision: USE 128 (Reason: Lite V3 - Dense & Large)\n");
        }
    }

    // 3. Execute Selected Strategy
    unsigned long long count = 0;
    float kernel_ms = 0.0f;
    std::vector<float> kernel_times(config.repeat);

    btc::CUDATimer timer_conv;

    timer.start(); // E2E starts

    if (use_128) {
        btc::BCSR_16x128_Device d_bcsr;

        timer_conv.start();
        fprintf(stderr, "[DBG] Starting BCSR 16x128 conversion...\n");
        btc::convert_coo_to_bcsr_16x128_gpu(d_bcsr, A_coo);
        cudaDeviceSynchronize();
        fprintf(stderr, "[DBG] BCSR 16x128 done, blocks=%u\n", d_bcsr.num_blocks);
        timer_conv.stop();

        // Warmup
        for (int w = 0; w < config.warmup; w++) {
            float tmp_ms = 0.0f;
            btc::count_triangles_16x128_adaptive(d_bcsr, 2048, &tmp_ms);
        }
        cudaDeviceSynchronize();

        // Timed runs
        for (int r = 0; r < config.repeat; r++) {
            float run_ms = 0.0f;
            count = btc::count_triangles_16x128_adaptive(d_bcsr, 2048, &run_ms);
            kernel_times[r] = run_ms;
        }
        cudaDeviceSynchronize();

        std::sort(kernel_times.begin(), kernel_times.end());
        kernel_ms = kernel_times[config.repeat / 2];

        d_bcsr.free();
    } else {
        btc::BCSR_16x32_Device d_bcsr;

        timer_conv.start();
        fprintf(stderr, "[DBG] Starting BCSR 16x32 conversion...\n");
        btc::convert_coo_to_bcsr_16x32_gpu(d_bcsr, A_coo);
        cudaDeviceSynchronize();
        fprintf(stderr, "[DBG] BCSR 16x32 done, blocks=%u\n", d_bcsr.num_blocks);
        timer_conv.stop();

        // Warmup
        for (int w = 0; w < config.warmup; w++) {
            float tmp_ms = 0.0f;
            btc::count_triangles_16x32_adaptive(d_bcsr, 40000, &tmp_ms);
        }
        cudaDeviceSynchronize();

        // Timed runs
        for (int r = 0; r < config.repeat; r++) {
            float run_ms = 0.0f;
            count = btc::count_triangles_16x32_adaptive(d_bcsr, 40000, &run_ms);
            kernel_times[r] = run_ms;
        }
        cudaDeviceSynchronize();

        std::sort(kernel_times.begin(), kernel_times.end());
        kernel_ms = kernel_times[config.repeat / 2];

        d_bcsr.free();
    }

    timer.stop(); // E2E stops

    float t_conv = timer_conv.elapsed();
    float t_e2e = timer.elapsed();

    std::printf("[Preprocessing] time: %.4f ms\n", t_pre);
    std::printf("[Time Breakdown] Convert: %.4f ms, Compute (Kernel): %.4f ms\n", t_conv, kernel_ms);
    std::printf("[Total Time (Convert+Compute)] time: %f ms\n", t_e2e);
    std::printf("[Total Time (Preprocess+Convert+Compute)] time: %f ms\n", t_pre + t_e2e);
    std::printf("Triangles (GPU): %llu\n", count);
    std::printf("Timing: warmup=%d, repeat=%d, median kernel reported\n", config.warmup, config.repeat);
    std::printf("Kernel_runs:");
    for (int r = 0; r < config.repeat; r++) std::printf(" %.6f", kernel_times[r]);
    std::printf("\n");

    return 0;
}
