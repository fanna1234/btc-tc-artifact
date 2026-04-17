#include <btc/btc.h>

#include <algorithm>
#include <cerrno>
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>
#include <string>
#include <vector>

#include <getopt.h>

namespace fs = std::filesystem;

namespace {

using index_t = long int;
using vertex_t = int;

static_assert(sizeof(index_t) == 8, "TC-Compare expects begin.bin index_t == 64-bit long int");
static_assert(sizeof(vertex_t) == 4, "TC-Compare expects vertex_t == 32-bit int");

struct Args {
    std::string input_mtx;
    fs::path output_dir;
    bool force = false;
    bool force_cpu = false;
};

[[noreturn]] void die_usage(const char* argv0)
{
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s -i <graph.mtx> -o <out_dir> [--force] [--cpu]\n"
        "\n"
        "Outputs (under <out_dir>):\n"
        "  edges_u32.bin              (for Bisson/Fox/Hu/Tricore)\n"
        "  polak_edges.bin            (for Polak)\n"
        "  graph/begin.bin            (index_t offsets)\n"
        "  graph/source.bin           (vertex_t sources)\n"
        "  graph/adjacent.bin         (vertex_t destinations)\n",
        argv0);
    std::exit(EXIT_FAILURE);
}

Args parse_args(int argc, char** argv)
{
    Args args;

    static option longopts[] = {
        {"input", required_argument, nullptr, 'i'},
        {"output", required_argument, nullptr, 'o'},
        {"force", no_argument, nullptr, 'f'},
        {"cpu", no_argument, nullptr, 'c'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:fch", longopts, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                args.input_mtx = optarg;
                break;
            case 'o':
                args.output_dir = fs::path(optarg);
                break;
            case 'f':
                args.force = true;
                break;
            case 'c':
                args.force_cpu = true;
                break;
            case 'h':
            default:
                die_usage(argv[0]);
        }
    }

    if (args.input_mtx.empty() || args.output_dir.empty()) {
        die_usage(argv[0]);
    }

    return args;
}

bool all_outputs_exist(const fs::path& out_dir)
{
    const fs::path graph_dir = out_dir / "graph";
    return fs::is_regular_file(out_dir / "edges_u32.bin") && fs::is_regular_file(out_dir / "polak_edges.bin")
           && fs::is_regular_file(graph_dir / "begin.bin") && fs::is_regular_file(graph_dir / "source.bin")
           && fs::is_regular_file(graph_dir / "adjacent.bin");
}

template<typename T>
void write_binary_file(const fs::path& path, const T* data, size_t count)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "Failed to open for write: %s (errno=%d)\n", path.string().c_str(), errno);
        std::exit(EXIT_FAILURE);
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) {
        std::fprintf(stderr, "Failed to write: %s\n", path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
}

void write_edges_u32_bin(const fs::path& path, const thrust::host_vector<int>& rows, const thrust::host_vector<int>& cols)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "Failed to open for write: %s (errno=%d)\n", path.string().c_str(), errno);
        std::exit(EXIT_FAILURE);
    }

    constexpr size_t kChunkEdges = 1u << 20;  // 1M edges => 8MB
    std::vector<uint32_t> buffer;
    buffer.reserve(kChunkEdges * 2);

    const size_t m = rows.size();
    for (size_t i = 0; i < m; i++) {
        buffer.push_back(static_cast<uint32_t>(rows[i]));
        buffer.push_back(static_cast<uint32_t>(cols[i]));
        if (buffer.size() == kChunkEdges * 2) {
            out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(uint32_t)));
            buffer.clear();
        }
    }

    if (!buffer.empty()) {
        out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(uint32_t)));
    }

    if (!out) {
        std::fprintf(stderr, "Failed to write: %s\n", path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
}

void write_polak_bin(const fs::path& path, const thrust::host_vector<int>& rows, const thrust::host_vector<int>& cols)
{
    const size_t m_undirected = rows.size();
    const unsigned long long m_directed_ull = static_cast<unsigned long long>(m_undirected) * 2ULL;
    if (m_directed_ull > static_cast<unsigned long long>(INT_MAX)) {
        std::fprintf(stderr, "Polak format requires 32-bit edge count header; got %llu directed edges.\n", m_directed_ull);
        std::exit(EXIT_FAILURE);
    }
    const int32_t m_directed = static_cast<int32_t>(m_directed_ull);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "Failed to open for write: %s (errno=%d)\n", path.string().c_str(), errno);
        std::exit(EXIT_FAILURE);
    }

    out.write(reinterpret_cast<const char*>(&m_directed), static_cast<std::streamsize>(sizeof(int32_t)));
    if (!out) {
        std::fprintf(stderr, "Failed to write header: %s\n", path.string().c_str());
        std::exit(EXIT_FAILURE);
    }

    constexpr size_t kChunkUndirectedEdges = 1u << 20;  // 1M undirected edges => 16MB payload
    std::vector<int32_t> buffer;
    buffer.reserve(kChunkUndirectedEdges * 4);

    for (size_t i = 0; i < m_undirected; i++) {
        const int32_t u = static_cast<int32_t>(rows[i]);
        const int32_t v = static_cast<int32_t>(cols[i]);
        buffer.push_back(u);
        buffer.push_back(v);
        buffer.push_back(v);
        buffer.push_back(u);
        if (buffer.size() == kChunkUndirectedEdges * 4) {
            out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(int32_t)));
            buffer.clear();
        }
    }

    if (!buffer.empty()) {
        out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(int32_t)));
    }

    if (!out) {
        std::fprintf(stderr, "Failed to write: %s\n", path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);

    const fs::path out_dir = args.output_dir;
    const fs::path graph_dir = out_dir / "graph";

    if (all_outputs_exist(out_dir) && !args.force) {
        std::printf("[mtx_to_tc_compare] Outputs already exist under %s (skip; use --force to regenerate)\n",
                    out_dir.string().c_str());
        return 0;
    }

    std::error_code ec;
    fs::create_directories(graph_dir, ec);
    if (ec) {
        std::fprintf(stderr, "Failed to create output directory: %s (%s)\n", graph_dir.string().c_str(), ec.message().c_str());
        return 1;
    }

    int n = 0;
    size_t m = 0;
    thrust::host_vector<int> h_rows;
    thrust::host_vector<int> h_cols;

    bool try_gpu = !args.force_cpu;
    if (try_gpu) {
        int device_count = 0;
        const cudaError_t rc = cudaGetDeviceCount(&device_count);
        if (rc != cudaSuccess || device_count <= 0) {
            try_gpu = false;
        }
    }

    if (try_gpu) {
        try {
            btc::CsrMatrix<int, float, btc::device_memory> A_csr;
            btc::CooMatrix<int, float, btc::device_memory> A_coo;

            btc::read_matrix_file(A_csr, args.input_mtx);
            if (A_csr.num_rows != A_csr.num_cols) {
                std::fprintf(stderr, "Input must be a square matrix for graph TC: %s (%d x %d)\n", args.input_mtx.c_str(), A_csr.num_rows,
                             A_csr.num_cols);
                return 1;
            }

            btc::convert_csr_to_coo(A_coo, A_csr);
            A_csr.free();

            // TC-Compare algorithms based on forward/merge intersection typically assume a strict ordering
            // and store each undirected edge once with src < dst (upper-triangular in that order).
            btc::preprocess_for_triangle_counting(A_coo, /*force_cpu=*/false, /*lower_triangular=*/false);

            n = A_coo.num_rows;
            m = static_cast<size_t>(A_coo.num_entries);

            h_rows = A_coo.row_indices;
            h_cols = A_coo.column_indices;
            A_coo.free();

            std::printf("[mtx_to_tc_compare] Preprocess: GPU\n");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[mtx_to_tc_compare] GPU preprocess failed: %s\n", e.what());
            std::fprintf(stderr, "[mtx_to_tc_compare] Falling back to CPU preprocessing.\n");
            try_gpu = false;
        }
    }

    if (!try_gpu) {
        btc::CsrMatrix<int, float, btc::host_memory> A_csr_h;
        btc::read_matrix_file(A_csr_h, args.input_mtx);
        if (A_csr_h.num_rows != A_csr_h.num_cols) {
            std::fprintf(stderr, "Input must be a square matrix for graph TC: %s (%d x %d)\n", args.input_mtx.c_str(), A_csr_h.num_rows,
                         A_csr_h.num_cols);
            return 1;
        }

        n = A_csr_h.num_rows;

        std::vector<std::pair<int, int>> edges;
        edges.reserve(static_cast<size_t>(A_csr_h.num_entries));

        for (int r = 0; r < n; r++) {
            const int row_start = A_csr_h.row_pointers[r];
            const int row_end = A_csr_h.row_pointers[r + 1];
            for (int idx = row_start; idx < row_end; idx++) {
                const int c = A_csr_h.column_indices[idx];
                if (r == c) continue;
                const int u = (r < c) ? r : c;
                const int v = (r < c) ? c : r;
                edges.emplace_back(u, v);
            }
        }

        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

        m = edges.size();
        h_rows.resize(m);
        h_cols.resize(m);
        for (size_t i = 0; i < m; i++) {
            h_rows[i] = edges[i].first;
            h_cols[i] = edges[i].second;
        }

        A_csr_h.free();
        std::printf("[mtx_to_tc_compare] Preprocess: CPU (sort+unique)\n");
    }

    std::vector<index_t> begin(static_cast<size_t>(n) + 1, 0);
    for (size_t i = 0; i < m; i++) {
        const int r = h_rows[i];
        if (r < 0 || r >= n) {
            std::fprintf(stderr, "Invalid row index %d at edge %zu (n=%d)\n", r, i, n);
            return 1;
        }
        begin[static_cast<size_t>(r) + 1] += 1;
    }
    for (int i = 1; i <= n; i++) {
        begin[static_cast<size_t>(i)] += begin[static_cast<size_t>(i) - 1];
    }

    write_binary_file(graph_dir / "begin.bin", begin.data(), begin.size());
    write_binary_file(graph_dir / "source.bin", h_rows.data(), h_rows.size());
    write_binary_file(graph_dir / "adjacent.bin", h_cols.data(), h_cols.size());

    write_edges_u32_bin(out_dir / "edges_u32.bin", h_rows, h_cols);
    write_polak_bin(out_dir / "polak_edges.bin", h_rows, h_cols);

    {
        std::ofstream manifest(out_dir / "manifest.txt");
        if (manifest) {
            manifest << "input=" << args.input_mtx << "\n";
            manifest << "n=" << n << "\n";
            manifest << "m=" << m << "\n";
        }
    }

    std::printf("[mtx_to_tc_compare] Wrote %zu edges (undirected) for n=%d into %s\n", m, n, out_dir.string().c_str());
    return 0;
}
