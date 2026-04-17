#include <btc/btc.h>
#include <btc/operations/tc_16x32_adaptive.h>
#include <cstdio>
#include <vector>
#include <algorithm>

int main(int argc, char** argv)
{
    btc::CsrMatrix<int, float, btc::device_memory> A_csr;
    btc::CooMatrix<int, float, btc::device_memory> A_coo;

    btc::Config config = btc::program_options(argc, argv);

    btc::read_matrix_file(A_csr, config.input_file);

    btc::convert_csr_to_coo(A_coo, A_csr);
    A_csr.free();

    // GPU-side preprocessing (sorting, triangular filtering, etc.)
    btc::CUDATimer timer_pre;
    timer_pre.start();
    btc::preprocess_for_triangle_counting(A_coo);
    timer_pre.stop();

    std::printf("\n--------------btc 16x32 Adaptive (V5/V6)----------------\n");
    {
        btc::CUDATimer timer_conv;
        btc::CUDATimer timer_total;

        timer_total.start();  // Convert+Compute total (GPU-side)

        btc::BCSR_16x32_Device d_bcsr;

        timer_conv.start();
        btc::convert_coo_to_bcsr_16x32_gpu(d_bcsr, A_coo);
        timer_conv.stop();
        std::printf("Num Blocks: %u\n", d_bcsr.num_blocks);

        // Warmup
        for (int w = 0; w < config.warmup; w++) {
            float tmp_ms = 0.0f;
            btc::count_triangles_16x32_adaptive(d_bcsr, 40000, &tmp_ms);
        }
        cudaDeviceSynchronize();

        // Timed runs
        unsigned long long count = 0;
        std::vector<float> kernel_times(config.repeat);
        for (int r = 0; r < config.repeat; r++) {
            float run_ms = 0.0f;
            count = btc::count_triangles_16x32_adaptive(d_bcsr, 40000, &run_ms);
            kernel_times[r] = run_ms;
        }
        cudaDeviceSynchronize();

        std::sort(kernel_times.begin(), kernel_times.end());
        float kernel_ms = kernel_times[config.repeat / 2];

        timer_total.stop();

        float t_pre = timer_pre.elapsed();
        float t_conv = timer_conv.elapsed();
        float t_total = timer_total.elapsed();
        std::printf("[Preprocessing] time: %.4f ms\n", t_pre);
        std::printf("[Time Breakdown] Convert: %.4f ms, Compute (Kernel): %.4f ms\n", t_conv, kernel_ms);
        std::printf("[Total Time (Convert+Compute)] time: %f ms\n", t_total);
        std::printf("Triangles (GPU): %llu\n", count);
        std::printf("Timing: warmup=%d, repeat=%d, median kernel reported\n", config.warmup, config.repeat);
        std::printf("Kernel_runs:");
        for (int r = 0; r < config.repeat; r++) std::printf(" %.6f", kernel_times[r]);
        std::printf("\n");

        if (config.verify) {
            std::printf("\n----------------cpu verify-------------\n");
            btc::CsrMatrix<int, float, btc::device_memory> A_csr_verify;
            btc::convert_coo_to_csr(A_csr_verify, A_coo);

            btc::CsrMatrix<int, float, btc::host_memory> A_csr_h;
            A_csr_h = A_csr_verify;

            auto count_cpu = btc::cpu_tc_intersection_lower_triangular(A_csr_h);
            std::printf("Triangles (CPU): %lu\n", (unsigned long)count_cpu);

            A_csr_verify.free();
            A_csr_h.free();
        }

        d_bcsr.free();
    }

    return 0;
}
