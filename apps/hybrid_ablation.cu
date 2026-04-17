#include <btc/btc.h>
#include <btc/operations/tc_16x128_mma_v3.h>
#include <btc/operations/tc_16x128_mma_v4.h>
#include <btc/operations/tc_16x128_mma_v5.h>
#include <btc/operations/tc_16x32_mma_v3.h>
#include <btc/operations/tc_16x32_mma_v4.h>
#include <btc/operations/tc_16x32_mma_v5.h>
#include <btc/operations/tc_16x32_mma_v6.h>
#include <cstdio>

// Wrapper: time a kernel that doesn't have kernel_ms param
template<typename Func>
float time_kernel(Func fn) {
    cudaDeviceSynchronize();
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);
    fn();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start); cudaEventDestroy(end);
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <input.mtx>\n", argv[0]);
        return 1;
    }

    btc::CsrMatrix<int, float, btc::device_memory> A_csr;
    btc::CooMatrix<int, float, btc::device_memory> A_coo;
    btc::read_matrix_file(A_csr, argv[1]);
    btc::convert_csr_to_coo(A_coo, A_csr);
    A_csr.free();
    btc::preprocess_for_triangle_counting(A_coo);

    std::string dataset = argv[1];
    size_t pos = dataset.find_last_of("/\\");
    if (pos != std::string::npos) dataset = dataset.substr(pos + 1);
    pos = dataset.find(".mtx");
    if (pos != std::string::npos) dataset = dataset.substr(0, pos);

    // === 16x128 ===
    float t128_v3, t128_v4, t128_v5;
    {
        btc::BCSR_16x128_Device d;
        btc::convert_coo_to_bcsr_16x128_gpu(d, A_coo);

        // v3: pure MMA
        t128_v3 = time_kernel([&]{ btc::count_triangles_16x128_v3(d); });
        // v4: adaptive hybrid (+ binary search)
        btc::count_triangles_16x128_v4(d, &t128_v4);
        // v5: adaptive hybrid (+ O(1) lookup)
        btc::count_triangles_16x128_v5(d, &t128_v5);

        d.free();
    }

    // === 16x32 ===
    float t32_v3, t32_v4, t32_v5, t32_v6;
    {
        btc::BCSR_16x32_Device d;
        btc::convert_coo_to_bcsr_16x32_gpu(d, A_coo);

        // v3: pure MMA (discrete K combination)
        t32_v3 = time_kernel([&]{ btc::count_triangles_16x32_v3(d); });
        // v4: pure MMA (dual accum)
        t32_v4 = time_kernel([&]{ btc::count_triangles_16x32_v4(d); });
        // v5: adaptive hybrid (+ binary search)
        btc::count_triangles_16x32_v5(d, &t32_v5);
        // v6: adaptive hybrid (+ O(1) lookup + SR)
        btc::count_triangles_16x32_v6(d, &t32_v6);

        d.free();
    }

    // CSV output
    std::printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                dataset.c_str(),
                t128_v3, t128_v4, t128_v5,
                t32_v3, t32_v4, t32_v5, t32_v6);

    A_coo.free();
    return 0;
}
