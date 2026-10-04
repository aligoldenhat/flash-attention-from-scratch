// Tests of the public C++ entry point fa::flash_fwd (csrc/flash_fwd.h), without Python:
//   * argument validation throws std::invalid_argument (pytest sees these as ValueError),
//   * a few end-to-end shapes against a double-precision CPU reference, covering partial
//     tiles, both head dims and causal masking. The full shape sweep lives in pytest.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.cuh"
#include "flash_fwd.h"

namespace {

struct Tensors {
    explicit Tensors(size_t count) : elems(count) {
        for (half** p : {&q, &k, &v, &o}) {
            CUDA_CHECK(cudaMalloc(p, count * sizeof(half)));
        }
    }
    Tensors(const Tensors&) = delete;
    Tensors& operator=(const Tensors&) = delete;
    Tensors(Tensors&&) = delete;
    Tensors& operator=(Tensors&&) = delete;
    ~Tensors() {
        for (half* p : {q, k, v, o}) {
            cudaFree(p);
        }
    }
    size_t elems;
    half* q = nullptr;
    half* k = nullptr;
    half* v = nullptr;
    half* o = nullptr;
};

fa::FlashFwdParams make_params(const Tensors& t, int batch, int heads, int n, int d, bool causal) {
    fa::FlashFwdParams p{};
    p.q = t.q;
    p.k = t.k;
    p.v = t.v;
    p.o = t.o;
    p.batch = batch;
    p.heads = heads;
    p.seqlen = n;
    p.head_dim = d;
    p.softmax_scale = 1.0F / std::sqrt(static_cast<float>(d));
    p.causal = causal;
    return p;
}

// ---- Argument validation -------------------------------------------------------------------

TEST(FlashFwdArgs, RejectsUnsupportedHeadDim) {
    const Tensors t(64 * 32);
    EXPECT_THROW(fa::flash_fwd(make_params(t, 1, 1, 64, 32, false), nullptr),
                 std::invalid_argument);
}

TEST(FlashFwdArgs, RejectsEmptyShapes) {
    const Tensors t(64 * 64);
    EXPECT_THROW(fa::flash_fwd(make_params(t, 1, 1, 0, 64, false), nullptr), std::invalid_argument);
    EXPECT_THROW(fa::flash_fwd(make_params(t, 0, 1, 64, 64, false), nullptr),
                 std::invalid_argument);
}

TEST(FlashFwdArgs, RejectsMisalignedPointers) {
    // cp.async moves 16 bytes at a time, so every row start must be 16-byte aligned.
    const Tensors t((64 * 64) + 8);
    auto p = make_params(t, 1, 1, 64, 64, false);
    p.q = t.q + 1;  // 2-byte offset
    EXPECT_THROW(fa::flash_fwd(p, nullptr), std::invalid_argument);
}

// ---- End to end against a CPU reference ------------------------------------------------------

struct Shape {
    int batch;
    int heads;
    int n;
    int d;
    bool causal;
    fa::Variant variant = fa::Variant::kOpt;
};

const char* variant_name(fa::Variant v) {
    switch (v) {
        case fa::Variant::kBaseline:
            return "baseline";
        case fa::Variant::kOpt:
            return "opt";
        case fa::Variant::kFp16Acc:
            return "fp16acc";
    }
    return "?";
}

// How GoogleTest (and ctest's test names) print a Shape, instead of a raw byte dump.
void PrintTo(const Shape& s, std::ostream* os) {
    *os << "B" << s.batch << "_H" << s.heads << "_N" << s.n << "_D" << s.d
        << (s.causal ? "_causal_" : "_full_") << variant_name(s.variant);
}

// Exact attention in double precision for all rows. Small shapes only.
std::vector<double> cpu_attention(const std::vector<half>& q, const std::vector<half>& k,
                                  const std::vector<half>& v, const Shape& s) {
    const auto n = static_cast<size_t>(s.n);
    const auto d = static_cast<size_t>(s.d);
    const size_t bh_count = static_cast<size_t>(s.batch) * static_cast<size_t>(s.heads);
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    auto at = [](const std::vector<half>& m, size_t i) {
        return static_cast<double>(__half2float(m[i]));
    };
    std::vector<double> out(bh_count * n * d, 0.0);
    std::vector<double> p(n);
    for (size_t bh = 0; bh < bh_count; ++bh) {
        const size_t base = bh * n * d;
        for (size_t i = 0; i < n; ++i) {
            const size_t kv_end = s.causal ? i + 1 : n;
            double max_s = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < kv_end; ++j) {
                double dot = 0.0;
                for (size_t c = 0; c < d; ++c) {
                    dot += at(q, base + (i * d) + c) * at(k, base + (j * d) + c);
                }
                p[j] = dot * scale;
                max_s = std::max(max_s, p[j]);
            }
            double sum = 0.0;
            for (size_t j = 0; j < kv_end; ++j) {
                p[j] = std::exp(p[j] - max_s);
                sum += p[j];
            }
            for (size_t j = 0; j < kv_end; ++j) {
                for (size_t c = 0; c < d; ++c) {
                    out[base + (i * d) + c] += (p[j] / sum) * at(v, base + (j * d) + c);
                }
            }
        }
    }
    return out;
}

class FlashFwdVsCpu : public ::testing::TestWithParam<Shape> {};

TEST_P(FlashFwdVsCpu, MatchesReference) {
    const Shape s = GetParam();
    const size_t elems = static_cast<size_t>(s.batch) * static_cast<size_t>(s.heads) *
                         static_cast<size_t>(s.n) * static_cast<size_t>(s.d);
    std::mt19937 rng(1234);  // NOLINT(bugprone-random-generator-seed): reproducible inputs
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<half> q(elems);
    std::vector<half> k(elems);
    std::vector<half> v(elems);
    for (size_t i = 0; i < elems; ++i) {
        q[i] = __float2half(dist(rng));
        k[i] = __float2half(dist(rng));
        v[i] = __float2half(dist(rng));
    }

    const Tensors t(elems);
    CUDA_CHECK(cudaMemcpy(t.q, q.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t.k, k.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t.v, v.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
    auto params = make_params(t, s.batch, s.heads, s.n, s.d, s.causal);
    params.variant = s.variant;
    fa::flash_fwd(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> o(elems);
    CUDA_CHECK(cudaMemcpy(o.data(), t.o, elems * sizeof(half), cudaMemcpyDeviceToHost));

    const auto ref = cpu_attention(q, k, v, s);
    double max_err = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(__half2float(o[i])) - ref[i]));
    }
    EXPECT_LT(max_err, 1e-2);
}

// N chosen to hit: a single row, a partial Q tile and partial K/V tile, exact multiples of
// the tiles, and several tiles with a partial last one; each for both head dims and causal.
// Every shape on every kernel build (baseline, opt, fp16acc).
std::vector<Shape> all_shapes() {
    // Positional rows read better than designated initializers in a table like this.
    // NOLINTBEGIN(modernize-use-designated-initializers)
    const Shape base[] = {{1, 1, 1, 64, false},  {1, 1, 1, 128, true},    {1, 2, 65, 64, false},
                          {1, 2, 65, 64, true},  {2, 1, 128, 128, false}, {2, 1, 128, 128, true},
                          {1, 2, 257, 64, true}, {1, 2, 300, 128, false}, {1, 2, 300, 128, true}};
    // NOLINTEND(modernize-use-designated-initializers)
    std::vector<Shape> out;
    for (const auto v : {fa::Variant::kBaseline, fa::Variant::kOpt, fa::Variant::kFp16Acc}) {
        for (Shape s : base) {
            s.variant = v;
            out.push_back(s);
        }
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(Shapes, FlashFwdVsCpu, ::testing::ValuesIn(all_shapes()));

}  // namespace
