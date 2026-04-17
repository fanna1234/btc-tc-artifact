#include <btc/btc.h>

#include <cstdio>

int main(int argc, char** argv)
{
    btc::CsrMatrix<int, float, btc::device_memory> A_csr;
    btc::CooMatrix<int, float, btc::device_memory> A_coo;

    btc::Config config = btc::program_options(argc, argv);

    btc::read_matrix_file(A_csr, config.input_file);

    btc::convert_csr_to_coo(A_coo, A_csr);
    A_csr.free();

    btc::CUDATimer timer;

    timer.start();
    btc::preprocess_for_triangle_counting(A_coo);
    timer.stop();
    std::printf("[Preprocessing (Symmetrize & Lower Triangular)] time: %f ms\n", timer.elapsed());

    std::printf("\n--------------btc 16x128 v5 (no binsearch)----------------\n");
    {
        timer.start();
        btc::BCSR_16x128_Device d_bcsr;
        btc::convert_coo_to_bcsr_16x128_gpu(d_bcsr, A_coo);
        timer.stop();
        std::printf("[Converting to BCSR 16x128] time: %lf ms\n", timer.elapsed());
        std::printf("  Num Blocks: %u\n", d_bcsr.num_blocks);

        timer.start();
        auto count = btc::count_triangles_16x128_v5(d_bcsr);
        timer.stop();
        std::printf("[Counting Triangles (16x128 v5)] time: %f ms\n", timer.elapsed());
        std::printf("Triangles (GPU): %llu\n", count);

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
