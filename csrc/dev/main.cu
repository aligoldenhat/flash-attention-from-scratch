// Standalone driver for the kernel, without Python: used by compute-sanitizer and ncu
// (smaller, cleaner traces than a PyTorch process) and by the CMake strict-warning build.
//
//   fa_dev [B H N D causal iters]       defaults: 4 8 4096 128 0 20
//
// Checks a sample of output rows against a double-precision CPU reference, then times
// `iters` launches with CUDA events.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "common.cuh"
#include "flash_fwd.h"

namespace {

// Exact attention for one query row (b*h index bh, row i), in double precision.
std::vector<double> reference_row(const std::vector<half>& q, const std::vector<half>& k,
                                  const std::vector<half>& v, size_t bh, size_t n, size_t d,
                                  size_t i, bool causal) {
    const size_t base = bh * n * d;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    const size_t kv_end = causal ? i + 1 : n;
    auto at = [](const std::vector<half>& m, size_t idx) {
        return static_cast<double>(__half2float(m[idx]));
    };
    std::vector<double> scores(kv_end);
    double max_score = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < kv_end; ++j) {
        double dot = 0.0;
        for (size_t c = 0; c < d; ++c) {
            dot += at(q, base + (i * d) + c) * at(k, base + (j * d) + c);
        }
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
    }
    double sum = 0.0;
    for (double& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
    }
    std::vector<double> out(d, 0.0);
    for (size_t j = 0; j < kv_end; ++j) {
        const double p = scores[j] / sum;
        for (size_t c = 0; c < d; ++c) {
            out[c] += p * at(v, base + (j * d) + c);
        }
    }
    return out;
}

int arg_or(int argc, char** argv, int idx, int fallback) {
    return argc > idx ? std::stoi(argv[idx]) : fallback;
}

int run(int argc, char** argv) {
    const int batch = arg_or(argc, argv, 1, 4);
    const int heads = arg_or(argc, argv, 2, 8);
    const int n = arg_or(argc, argv, 3, 4096);
    const int d = arg_or(argc, argv, 4, 128);
    const bool causal = arg_or(argc, argv, 5, 0) != 0;
    const int iters = arg_or(argc, argv, 6, 20);

    const size_t elems = static_cast<size_t>(batch) * static_cast<size_t>(heads) *
                         static_cast<size_t>(n) * static_cast<size_t>(d);
    const size_t bytes = elems * sizeof(half);

    std::mt19937 rng(42);  // NOLINT(bugprone-random-generator-seed): reproducible inputs
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<half> h_q(elems);
    std::vector<half> h_k(elems);
    std::vector<half> h_v(elems);
    for (size_t i = 0; i < elems; ++i) {
        h_q[i] = __float2half(dist(rng));
        h_k[i] = __float2half(dist(rng));
        h_v[i] = __float2half(dist(rng));
    }

    half* d_q = nullptr;
    half* d_k = nullptr;
    half* d_v = nullptr;
    half* d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_o, bytes));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));

    fa::FlashFwdParams p{};
    p.q = d_q;
    p.k = d_k;
    p.v = d_v;
    p.o = d_o;
    p.batch = batch;
    p.heads = heads;
    p.seqlen = n;
    p.head_dim = d;
    p.softmax_scale = 1.0F / std::sqrt(static_cast<float>(d));
    p.causal = causal;

    fa::flash_fwd(p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Correctness: a handful of rows spread over (b, h) and N, including the last row.
    std::vector<half> h_o(elems);
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    const size_t num_bh = static_cast<size_t>(batch) * static_cast<size_t>(heads);
    const auto un = static_cast<size_t>(n);
    const auto ud = static_cast<size_t>(d);
    for (const size_t bh : {size_t{0}, num_bh - 1}) {
        for (const size_t i : {size_t{0}, un / 3, un - 1}) {
            const auto ref = reference_row(h_q, h_k, h_v, bh, un, ud, i, causal);
            const size_t base = ((bh * un) + i) * ud;
            for (size_t c = 0; c < ud; ++c) {
                const auto got = static_cast<double>(__half2float(h_o[base + c]));
                max_err = std::max(max_err, std::fabs(got - ref[c]));
            }
        }
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaEventRecord(start));
        fa::flash_fwd(p, nullptr);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    std::ranges::sort(times);
    const double median_ms = times.empty() ? 0.0 : static_cast<double>(times[times.size() / 2]);
    const double flops =
        4.0 * static_cast<double>(elems) * static_cast<double>(n) / (causal ? 2.0 : 1.0);

    std::printf("B=%d H=%d N=%d D=%d causal=%d  max_abs_err=%.2e  median=%.3f ms  %.1f TFLOPS\n",
                batch, heads, n, d, causal ? 1 : 0, max_err, median_ms,
                median_ms > 0.0 ? flops / (median_ms * 1e9) : 0.0);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));

    if (max_err > 1e-2) {
        std::printf("FAILED: max_abs_err above 1e-2\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
