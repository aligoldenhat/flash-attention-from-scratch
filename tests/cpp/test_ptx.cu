// Unit tests of the kernel's building blocks (csrc/common.cuh), one tiny kernel each.
//
// pytest checks the final attention output; these tests check the pieces it is built from,
// so a failure points at the exact piece that is wrong. Each test is also an executable
// statement of a layout described in docs/optimizations.md:
//   * the mma.sync m16n8k16 fragment layout (which lane holds which element),
//   * what ldmatrix / ldmatrix.trans put in each lane's registers,
//   * the swizzle is a permutation and makes ldmatrix bank-conflict-free,
//   * cp.async with src-size 0 writes zeros.
//
// Values are small integers, which fp16 and fp32 represent exactly, so results are compared
// for exact equality: any layout mistake shows up as a wrong number, not as rounding noise.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "common.cuh"

namespace {

constexpr int kWarp = 32;

// Small RAII wrapper for a device buffer, so a failing ASSERT never leaks memory.
template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;
    ~DeviceBuffer() { cudaFree(ptr_); }

    [[nodiscard]] T* get() const { return ptr_; }
    [[nodiscard]] std::vector<T> to_host() const {
        std::vector<T> host(count_);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }

private:
    T* ptr_ = nullptr;
    size_t count_;
};

// Row-major index (row * stride + col), computed in size_t so nothing overflows in int first.
size_t idx(int row, int stride, int col) {
    return (static_cast<size_t>(row) * static_cast<size_t>(stride)) + static_cast<size_t>(col);
}

void sync_and_check() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Splits a 32-bit register holding a half2 into its two halfs, as floats.
float lo_half(uint32_t r) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(r & 0xFFFFU)));
}
float hi_half(uint32_t r) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(r >> 16U)));
}

__device__ uint32_t pack(half lo, half hi) {
    const __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// ---------------------------------------------------------------------------------------------
// mma.sync.m16n8k16: build the A and B fragments by hand from the layout table, multiply,
// scatter C by the same table, and compare with a CPU matmul. If the table in common.cuh /
// docs were wrong, C would come out permuted.
//   g = lane / 4, t = lane % 4
//   a0 = A[g][2t..2t+1]  a1 = A[g+8][2t..]  a2 = A[g][2t+8..]  a3 = A[g+8][2t+8..]
//   b0 = B[2t..2t+1][g]  b1 = B[2t+8..2t+9][g]
//   c0,c1 = C[g][2t..2t+1]  c2,c3 = C[g+8][2t..2t+1]
// ---------------------------------------------------------------------------------------------
// F16ACC: the fp16-accumulate mma (C as 2 half2 registers: row g and row g+8).
template <bool F16ACC>
__global__ void mma_from_layout_table(const half* a_mat /*16x16*/, const half* b_mat /*16x8*/,
                                      float* c_mat /*16x8*/) {
    const int lane = static_cast<int>(threadIdx.x);
    const int g = lane / 4;
    const int t = lane % 4;
    auto A = [&](int r, int c) { return a_mat[(r * 16) + c]; };
    auto B = [&](int k, int n) { return b_mat[(k * 8) + n]; };
    const uint32_t a[4] = {
        pack(A(g, 2 * t), A(g, (2 * t) + 1)),
        pack(A(g + 8, 2 * t), A(g + 8, (2 * t) + 1)),
        pack(A(g, (2 * t) + 8), A(g, (2 * t) + 9)),
        pack(A(g + 8, (2 * t) + 8), A(g + 8, (2 * t) + 9)),
    };
    const uint32_t b0 = pack(B(2 * t, g), B((2 * t) + 1, g));
    const uint32_t b1 = pack(B((2 * t) + 8, g), B((2 * t) + 9, g));
    float d[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    if constexpr (F16ACC) {
        uint32_t d16[2] = {0U, 0U};
        fa::ptx::mma_16816_f16acc(d16, a, b0, b1);
        const float2 row_g = fa::ptx::unpack_half2(d16[0]);
        const float2 row_g8 = fa::ptx::unpack_half2(d16[1]);
        d[0] = row_g.x;
        d[1] = row_g.y;
        d[2] = row_g8.x;
        d[3] = row_g8.y;
    } else {
        fa::ptx::mma_16816(d, a, b0, b1);
    }
    c_mat[(g * 8) + (2 * t)] = d[0];
    c_mat[(g * 8) + (2 * t) + 1] = d[1];
    c_mat[((g + 8) * 8) + (2 * t)] = d[2];
    c_mat[((g + 8) * 8) + (2 * t) + 1] = d[3];
}

template <bool F16ACC>
void check_mma_layout() {
    std::vector<half> a(16 * 16);
    std::vector<half> b(16 * 8);
    std::vector<float> a_f(a.size());
    std::vector<float> b_f(b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a_f[i] = static_cast<float>(static_cast<int>(i % 7) - 3);  // -3..3
        a[i] = __float2half(a_f[i]);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b_f[i] = static_cast<float>(static_cast<int>(i % 5) - 2);  // -2..2
        b[i] = __float2half(b_f[i]);
    }
    const DeviceBuffer<half> d_a(a);
    const DeviceBuffer<half> d_b(b);
    const DeviceBuffer<float> d_c(16 * 8);
    mma_from_layout_table<F16ACC><<<1, kWarp>>>(d_a.get(), d_b.get(), d_c.get());
    sync_and_check();

    const auto c = d_c.to_host();
    for (int r = 0; r < 16; ++r) {
        for (int n = 0; n < 8; ++n) {
            float expected = 0.0F;
            for (int k = 0; k < 16; ++k) {
                expected += a_f[idx(r, 16, k)] * b_f[idx(k, 8, n)];
            }
            EXPECT_EQ(c[idx(r, 8, n)], expected) << "C[" << r << "][" << n << "]";
        }
    }
}

TEST(Mma16816, FragmentLayoutMatchesCpuMatmul) {
    check_mma_layout<false>();
}

// Same products (|C| <= 96, exact in fp16), through the fp16-accumulate instruction.
TEST(Mma16816, F16AccumulateFragmentLayoutMatchesCpuMatmul) {
    check_mma_layout<true>();
}

// ---------------------------------------------------------------------------------------------
// ldmatrix: put a 16x16 matrix M[r][c] = r*16 + c in shared memory, load it with the same
// lane -> address mapping as the kernel, and check every register of every lane.
// ---------------------------------------------------------------------------------------------
enum class LdMode : std::uint8_t { kQ = 0, kVTrans = 1 };

__global__ void ldmatrix_16x16(const half* m /*16x16*/, uint32_t* regs /*32 lanes x 4*/,
                               LdMode mode) {
    __shared__ __align__(16) half s_m[16 * 16];
    const int lane = static_cast<int>(threadIdx.x);
    for (int i = lane; i < 16 * 16; i += kWarp) {
        s_m[i] = m[i];
    }
    __syncwarp();
    uint32_t r[4];
    if (mode == LdMode::kQ) {
        // As for Q (A operand): lanes 0-15 -> rows 0-15 col 0, lanes 16-31 -> rows 0-15 col 8.
        const int row = lane & 15;
        const int col = (lane >> 4) * 8;
        fa::ptx::ldmatrix_x4(r, fa::ptx::smem_addr(&s_m[(row * 16) + col]));
    } else {
        // As for V (B operand of P V, rows = keys): matrices (k0,n0) (k8,n0) (k0,n8) (k8,n8).
        const int row = (lane & 7) + (((lane >> 3) & 1) * 8);
        const int col = (lane >> 4) * 8;
        fa::ptx::ldmatrix_x4_trans(r, fa::ptx::smem_addr(&s_m[(row * 16) + col]));
    }
    for (int i = 0; i < 4; ++i) {
        regs[(lane * 4) + i] = r[i];
    }
}

std::vector<uint32_t> run_ldmatrix(LdMode mode) {
    std::vector<half> m(16 * 16);
    for (size_t i = 0; i < m.size(); ++i) {
        m[i] = __float2half(static_cast<float>(i));  // M[r][c] = r*16 + c, exact in fp16
    }
    const DeviceBuffer<half> d_m(m);
    const DeviceBuffer<uint32_t> d_regs(kWarp * 4);
    ldmatrix_16x16<<<1, kWarp>>>(d_m.get(), d_regs.get(), mode);
    sync_and_check();
    return d_regs.to_host();
}

float m_at(int r, int c) {
    return static_cast<float>((r * 16) + c);
}

TEST(Ldmatrix, X4GivesMmaAFragment) {
    const auto regs = run_ldmatrix(LdMode::kQ);
    for (int lane = 0; lane < kWarp; ++lane) {
        const int g = lane / 4;
        const int t = lane % 4;
        // Expected = the A-fragment rows of the layout table.
        const int rows[4] = {g, g + 8, g, g + 8};
        const int cols[4] = {2 * t, 2 * t, (2 * t) + 8, (2 * t) + 8};
        for (int i = 0; i < 4; ++i) {
            const uint32_t r = regs[idx(lane, 4, i)];
            EXPECT_EQ(lo_half(r), m_at(rows[i], cols[i])) << "lane " << lane << " a" << i << " lo";
            EXPECT_EQ(hi_half(r), m_at(rows[i], cols[i] + 1))
                << "lane " << lane << " a" << i << " hi";
        }
    }
}

TEST(Ldmatrix, X4TransGivesMmaBFragmentOfRowMajorV) {
    // V is [key][d] row-major; the mma B operand wants B[k = key][n = d] with a lane holding
    // two consecutive *keys*: b0 = (V[2t][n], V[2t+1][n]). .trans provides exactly that.
    const auto regs = run_ldmatrix(LdMode::kVTrans);
    for (int lane = 0; lane < kWarp; ++lane) {
        const int g = lane / 4;
        const int t = lane % 4;
        // r0,r1 = b0,b1 of n-tile 0 (d = g); r2,r3 = b0,b1 of n-tile 1 (d = 8 + g).
        const int keys[4] = {2 * t, (2 * t) + 8, 2 * t, (2 * t) + 8};
        const int dims[4] = {g, g, 8 + g, 8 + g};
        for (int i = 0; i < 4; ++i) {
            const uint32_t r = regs[idx(lane, 4, i)];
            EXPECT_EQ(lo_half(r), m_at(keys[i], dims[i])) << "lane " << lane << " r" << i << " lo";
            EXPECT_EQ(hi_half(r), m_at(keys[i] + 1, dims[i]))
                << "lane " << lane << " r" << i << " hi";
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Swizzle (host-side: swizzle<D> is __host__ __device__).
// ---------------------------------------------------------------------------------------------
// The 4-bank group (0..7) a 16-byte chunk at element offset `off` lands in.
int bank_group(int off) {
    return ((off * 2) / 16) % 8;
}

template <int D>
void check_swizzle_is_permutation(int rows) {
    constexpr int kChunks = D / 8;
    std::set<int> seen;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < kChunks; ++c) {
            const int off = fa::swizzle<D>(r, c);
            ASSERT_EQ(off % 8, 0) << "chunk not 16-byte aligned";
            ASSERT_GE(off, r * D) << "chunk left its row";
            ASSERT_LT(off, (r + 1) * D) << "chunk left its row";
            ASSERT_TRUE(seen.insert(off).second) << "two chunks collide at offset " << off;
        }
    }
}

TEST(Swizzle, IsAPermutationWithinEachRow) {
    check_swizzle_is_permutation<64>(64);
    check_swizzle_is_permutation<128>(64);
}

template <int D>
void check_ldmatrix_conflict_free() {
    // One ldmatrix phase reads the same logical chunk of 8 consecutive rows. Conflict-free
    // means the 8 chunks are in 8 different bank groups.
    for (int row0 = 0; row0 < 64; row0 += 8) {
        for (int c = 0; c < D / 8; ++c) {
            std::set<int> groups;
            std::set<int> groups_unswizzled;
            for (int r = row0; r < row0 + 8; ++r) {
                groups.insert(bank_group(fa::swizzle<D>(r, c)));
                groups_unswizzled.insert(bank_group((r * D) + (c * 8)));
            }
            EXPECT_EQ(groups.size(), 8U) << "bank conflict at rows " << row0 << "+, chunk " << c;
            // The reason the swizzle exists: without it all 8 rows hit the same group.
            EXPECT_EQ(groups_unswizzled.size(), 1U);
        }
    }
}

TEST(Swizzle, LdmatrixColumnReadsAreBankConflictFree) {
    check_ldmatrix_conflict_free<64>();
    check_ldmatrix_conflict_free<128>();
}

TEST(Swizzle, RowWritesAreBankConflictFree) {
    // cp.async: 8 consecutive threads write chunks 0..7 of one row.
    for (int r = 0; r < 64; ++r) {
        std::set<int> groups;
        for (int c = 0; c < 8; ++c) {
            groups.insert(bank_group(fa::swizzle<64>(r, c)));
        }
        EXPECT_EQ(groups.size(), 8U) << "row " << r;
    }
}

// ---------------------------------------------------------------------------------------------
// cp.async: src-size 16 copies, src-size 0 zero-fills (how the kernel pads rows >= N).
// ---------------------------------------------------------------------------------------------
__global__ void cp_async_zero_fill(const half* src /*8 halfs*/, half* out /*16 halfs*/) {
    __shared__ __align__(16) half s_buf[16];
    if (threadIdx.x == 0) {
        for (half& h : s_buf) {
            h = __float2half(-1.0F);  // garbage that must be overwritten
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        fa::ptx::cp_async_16(fa::ptx::smem_addr(&s_buf[0]), src, 16);  // real copy
        fa::ptx::cp_async_16(fa::ptx::smem_addr(&s_buf[8]), src, 0);   // zero fill
        fa::ptx::cp_async_commit();
        fa::ptx::cp_async_wait<0>();
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        out[threadIdx.x] = s_buf[threadIdx.x];
    }
}

TEST(CpAsync, SrcSizeZeroWritesZeros) {
    std::vector<half> src(8);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = __float2half(static_cast<float>(i + 1));
    }
    const DeviceBuffer<half> d_src(src);
    const DeviceBuffer<half> d_out(16);
    cp_async_zero_fill<<<1, kWarp>>>(d_src.get(), d_out.get());
    sync_and_check();
    const auto out = d_out.to_host();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(__half2float(out[i]), static_cast<float>(i + 1)) << "copied element " << i;
        EXPECT_EQ(__half2float(out[8 + i]), 0.0F) << "zero-filled element " << i;
    }
}

// ---------------------------------------------------------------------------------------------
// Exponentials: the FA4-style polynomial exp2 (FMA pipe) and ex2.approx.f16x2 (SFU, packed).
// ---------------------------------------------------------------------------------------------
__global__ void exp2_kernels(const float* x, float* poly, float* f16x2, int n) {
    const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (2 * i + 1 < n) {
        poly[2 * i] = fa::ptx::exp2_poly3(x[2 * i]);
        poly[(2 * i) + 1] = fa::ptx::exp2_poly3(x[(2 * i) + 1]);
        const float2 r = fa::ptx::unpack_half2(
            fa::ptx::ex2_f16x2(fa::ptx::pack_half2(x[2 * i], x[(2 * i) + 1])));
        f16x2[2 * i] = r.x;
        f16x2[(2 * i) + 1] = r.y;
    }
}

struct Exp2Results {
    std::vector<float> x;
    std::vector<float> poly;
    std::vector<float> f16x2;
};

// x from -30 to +8 (lazy rescaling lets p reach 2^8), plus -inf (masked scores).
Exp2Results run_exp2() {
    Exp2Results r;
    for (int i = 0; i <= 3800; ++i) {
        r.x.push_back(-30.0F + (static_cast<float>(i) * 0.01F));
    }
    r.x.push_back(-std::numeric_limits<float>::infinity());
    r.x.push_back(-1000.0F);
    if (r.x.size() % 2 != 0) {
        r.x.push_back(0.0F);
    }
    const auto n = static_cast<int>(r.x.size());
    const DeviceBuffer<float> d_x(r.x);
    const DeviceBuffer<float> d_poly(r.x.size());
    const DeviceBuffer<float> d_f16(r.x.size());
    const auto blocks = static_cast<unsigned>(((n / 2) + 255) / 256);
    exp2_kernels<<<blocks, 256>>>(d_x.get(), d_poly.get(), d_f16.get(), n);
    sync_and_check();
    r.poly = d_poly.to_host();
    r.f16x2 = d_f16.to_host();
    return r;
}

TEST(Exp2, PolynomialRelativeErrorBelowFp16Resolution) {
    const auto r = run_exp2();
    double worst = 0.0;
    for (size_t i = 0; i < r.x.size(); ++i) {
        const double expected = std::exp2(static_cast<double>(r.x[i]));
        if (expected == 0.0 || expected < 1e-30) {
            EXPECT_LT(r.poly[i], 1e-30F) << "x = " << r.x[i];
            continue;
        }
        const double rel = std::fabs(static_cast<double>(r.poly[i]) - expected) / expected;
        worst = std::max(worst, rel);
    }
    // fp16 resolves 2^-11 = 4.9e-4; P is rounded to fp16 right after, so 2e-4 is invisible.
    EXPECT_LT(worst, 2e-4) << "worst relative error " << worst;
}

TEST(Exp2, F16x2MatchesExp2WithinFp16Precision) {
    const auto r = run_exp2();
    for (size_t i = 0; i < r.x.size(); ++i) {
        const double expected = std::exp2(static_cast<double>(r.x[i]));
        const auto got = static_cast<double>(r.f16x2[i]);
        // Relative error near fp16 resolution for normal results; absolute for tiny ones
        // (the input x itself is rounded to fp16, which costs up to ~2e-3 relative at |x|=8).
        const double tol = std::max(4e-3 * expected, 1e-6);
        EXPECT_NEAR(got, expected, tol) << "x = " << r.x[i];
    }
}

}  // namespace
