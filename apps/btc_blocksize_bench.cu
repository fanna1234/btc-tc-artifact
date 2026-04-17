#include <btc/btc.h>
#include <btc/operations/blocksize_bench_bcsr.h>
#include <btc/operations/blocksize_bench_kernels.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;

struct Args {
    std::string input_file;
    int runs = 10;
    int warmup = 3;
    bool preprocess = true;
    bool verify = false;
    int max_samples = -1;  // <0 means use all blocks
    bool print_kernel_info = true;
};

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (std::strcmp(a, "-i") == 0 && i + 1 < argc) {
            args.input_file = argv[++i];
            continue;
        }
        if (std::strcmp(a, "--runs") == 0 && i + 1 < argc) {
            args.runs = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(a, "--warmup") == 0 && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(a, "--no-preprocess") == 0) {
            args.preprocess = false;
            continue;
        }
        if (std::strcmp(a, "--verify") == 0) {
            args.verify = true;
            continue;
        }
        if (std::strcmp(a, "--max-samples") == 0 && i + 1 < argc) {
            args.max_samples = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(a, "--no-kernel-info") == 0) {
            args.print_kernel_info = false;
            continue;
        }

        std::fprintf(stderr,
                     "Usage: %s -i <graph.mtx> [--runs N] [--warmup N] [--no-preprocess] [--verify] [--max-samples N] [--no-kernel-info]\n",
                     argv[0]);
        std::exit(2);
    }

    if (args.input_file.empty()) {
        std::fprintf(stderr,
                     "Usage: %s -i <graph.mtx> [--runs N] [--warmup N] [--no-preprocess] [--verify] [--max-samples N] [--no-kernel-info]\n",
                     argv[0]);
        std::exit(2);
    }
    args.runs = std::max(1, args.runs);
    args.warmup = std::max(0, args.warmup);
    return args;
}

template<typename KernelFn>
void print_kernel_attrs(const char* name, KernelFn fn, int threads_per_block)
{
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, fn) != cudaSuccess) return;

    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, threads_per_block, 0);

    std::cout << "  " << name
              << " | regs/thread=" << attr.numRegs
              << " | smem=" << attr.sharedSizeBytes
              << " | blocks/SM=" << blocks_per_sm
              << "\n";
}

template<typename KernelFn>
float benchmark_kernel(KernelFn fn,
                       int n,
                       int n_row_blocks,
                       const int* indptr,
                       const int* indices,
                       const uint32_t* blocks,
                       int num_sample_blocks,
                       unsigned long long* result,
                       int warmup,
                       int runs)
{
    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    const int num_cuda_blocks = (num_sample_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    for (int i = 0; i < warmup; i++) {
        CHECK_CUDA(cudaMemset(result, 0, sizeof(unsigned long long)));
        fn<<<num_cuda_blocks, threads_per_block>>>(n, n_row_blocks, indptr, indices, blocks, num_sample_blocks, result);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < runs; i++) {
        CHECK_CUDA(cudaMemset(result, 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaEventRecord(start));
        fn<<<num_cuda_blocks, threads_per_block>>>(n, n_row_blocks, indptr, indices, blocks, num_sample_blocks, result);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(runs);
}

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);

    int dev = 0;
    {
        const cudaError_t rc = cudaGetDevice(&dev);
        if (rc != cudaSuccess) {
            std::fprintf(stderr,
                         "CUDA Error: %s (cudaGetDevice). Is a CUDA-capable GPU available?\n",
                         cudaGetErrorString(rc));
            return 1;
        }
    }
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")\n";
    std::cout << "Input: " << args.input_file << "\n";

    using IndexType = int;
    using ValueType = float;

    btc::CsrMatrix<IndexType, ValueType, btc::host_memory> host_csr;
    {
        const int rc = btc::read_from_mtx(host_csr, args.input_file);
        if (rc != 0) {
            std::cerr << "Failed to read matrix: " << rc << "\n";
            return 1;
        }
    }

    btc::CooMatrix<IndexType, ValueType, btc::host_memory> host_coo;
    btc::convert_csr_to_coo(host_coo, host_csr);
    host_csr.free();

    if (args.preprocess) {
        btc::preprocess_for_triangle_counting(host_coo, /*force_cpu=*/true, /*lower_triangular=*/true);
    }

    std::vector<std::pair<int, int>> edges_lower;
    edges_lower.reserve(static_cast<size_t>(host_coo.num_entries));
    for (int i = 0; i < host_coo.num_entries; i++) {
        edges_lower.emplace_back(host_coo.row_indices[i], host_coo.column_indices[i]);
    }

    unsigned long long expected = 0;
    if (args.verify && args.max_samples < 0) {
        btc::CsrMatrix<IndexType, ValueType, btc::host_memory> csr_verify;
        btc::convert_coo_to_csr(csr_verify, host_coo);
        expected = static_cast<unsigned long long>(btc::cpu_tc_intersection_lower_triangular(csr_verify));
        std::cout << "CPU triangles: " << expected << "\n";
    } else if (args.verify && args.max_samples >= 0) {
        std::cout << "[WARN] --verify ignored when --max-samples is set (partial sampling).\n";
    }

    struct ResultRow {
        std::string name;
        unsigned long long count = 0;
        float ms = 0.0f;
        int num_blocks = 0;
        size_t bytes = 0;
        bool supported = true;
    };

    std::vector<ResultRow> results;

    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    if (args.print_kernel_info) {
        std::cout << "Kernel attributes (threads/block=" << threads_per_block << "):\n";
        if (prop.major >= 8) print_kernel_attrs("8x128", btc::bench::kernel_8x128_mma_twopointer, threads_per_block);
        if (prop.major >= 8) print_kernel_attrs("16x128", btc::bench::kernel_16x128_mma_twopointer, threads_per_block);
        if (prop.major >= 8) print_kernel_attrs("16x256", btc::bench::kernel_16x256_mma_twopointer, threads_per_block);
    }

    // Build + benchmark each block size.
    {
        ResultRow row;
        row.name = "8x128";
        if (prop.major < 8) {
            row.supported = false;
            results.push_back(row);
        } else {
            btc::bench::BCSRHost<8, 128> h;
            auto t0 = std::chrono::high_resolution_clock::now();
            h.build_from_edges(host_coo.num_rows, edges_lower);
            auto t1 = std::chrono::high_resolution_clock::now();
            (void)t0;
            (void)t1;

            btc::bench::BCSRDevice<btc::bench::BCSRHost<8, 128>::SIZE_U32> d;
            d.allocate_and_copy(h);
            d.reset_result();

            const int samples = (args.max_samples < 0) ? d.num_blocks : std::min(d.num_blocks, args.max_samples);
            const float ms = benchmark_kernel(btc::bench::kernel_8x128_mma_twopointer, d.n, d.n_row_blocks, d.indptr, d.indices, d.blocks, samples, d.result, args.warmup, args.runs);

            row.count = d.get_result();
            row.ms = ms;
            row.num_blocks = d.num_blocks;
            row.bytes = d.bytes_total();

            d.free();
            results.push_back(row);
        }
    }
    {
        ResultRow row;
        row.name = "16x128";
        if (prop.major < 8) {
            row.supported = false;
            results.push_back(row);
        } else {
            btc::bench::BCSRHost<16, 128> h;
            h.build_from_edges(host_coo.num_rows, edges_lower);

            btc::bench::BCSRDevice<btc::bench::BCSRHost<16, 128>::SIZE_U32> d;
            d.allocate_and_copy(h);
            d.reset_result();

            const int samples = (args.max_samples < 0) ? d.num_blocks : std::min(d.num_blocks, args.max_samples);
            const float ms = benchmark_kernel(btc::bench::kernel_16x128_mma_twopointer, d.n, d.n_row_blocks, d.indptr, d.indices, d.blocks, samples, d.result, args.warmup, args.runs);

            row.count = d.get_result();
            row.ms = ms;
            row.num_blocks = d.num_blocks;
            row.bytes = d.bytes_total();

            d.free();
            results.push_back(row);
        }
    }
    {
        ResultRow row;
        row.name = "16x256";
        if (prop.major < 8) {
            row.supported = false;
            results.push_back(row);
        } else {
            btc::bench::BCSRHost<16, 256> h;
            h.build_from_edges(host_coo.num_rows, edges_lower);

            btc::bench::BCSRDevice<btc::bench::BCSRHost<16, 256>::SIZE_U32> d;
            d.allocate_and_copy(h);
            d.reset_result();

            const int samples = (args.max_samples < 0) ? d.num_blocks : std::min(d.num_blocks, args.max_samples);
            const float ms = benchmark_kernel(btc::bench::kernel_16x256_mma_twopointer, d.n, d.n_row_blocks, d.indptr, d.indices, d.blocks, samples, d.result, args.warmup, args.runs);

            row.count = d.get_result();
            row.ms = ms;
            row.num_blocks = d.num_blocks;
            row.bytes = d.bytes_total();

            d.free();
            results.push_back(row);
        }
    }

    std::cout << "\n"
              << std::left << std::setw(10) << "Block"
              << std::right << std::setw(14) << "Blocks"
              << std::setw(14) << "Mem(MB)"
              << std::setw(16) << "Time(ms)"
              << std::setw(18) << "Triangles"
              << std::setw(10) << "Status"
              << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (const auto& r : results) {
        if (!r.supported) {
            std::cout << std::left << std::setw(10) << r.name
                      << std::right << std::setw(14) << "-"
                      << std::setw(14) << "-"
                      << std::setw(16) << "-"
                      << std::setw(18) << "-"
                      << std::setw(10) << "SKIP"
                      << "\n";

            std::cout << "RESULT_CSV," << args.input_file << ","
                      << r.name << ","
                      << 0 << ","
                      << 0 << ","
                      << 0.0f << ","
                      << 0 << ","
                      << "SKIP"
                      << "\n";
            continue;
        }

        const bool ok = (!args.verify || args.max_samples >= 0) ? true : (r.count == expected);
        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setw(14) << r.num_blocks
                  << std::setw(14) << std::fixed << std::setprecision(2) << (r.bytes / (1024.0 * 1024.0))
                  << std::setw(16) << std::fixed << std::setprecision(4) << r.ms
                  << std::setw(18) << r.count
                  << std::setw(10) << (ok ? "PASS" : "FAIL")
                  << "\n";

        std::cout << "RESULT_CSV," << args.input_file << ","
                  << r.name << ","
                  << r.num_blocks << ","
                  << r.bytes << ","
                  << r.ms << ","
                  << r.count << ","
                  << (ok ? "PASS" : "FAIL")
                  << "\n";
    }

    return 0;
}
