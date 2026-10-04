#pragma once
// Shared helpers: error checking and the small inline-PTX wrappers used by the kernel.
// Deliberately free of any PyTorch include, so the kernel also builds in the standalone
// CMake target (csrc/dev/main.cu) used for compute-sanitizer and ncu.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// Throws instead of exit() so a failure inside the Python extension surfaces as a Python
// RuntimeError instead of killing the interpreter. A macro, so __FILE__/__LINE__ are the caller's.
#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                                \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +             \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(err_)); \
        }                                                                                         \
    } while (0)

// After a <<<...>>> launch: catches bad launch configurations (too much smem, bad grid, ...).
// Errors raised *while* the kernel runs show up at the next synchronizing call.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace fa {

// Shared-memory layout of a [rows][D] fp16 tile: element offset of 16-byte chunk `chunk`
// (8 halfs) of row `row`.
// A row is D*2 = 128 or 256 bytes, a multiple of the 128-byte bank window, so without a
// swizzle every row starts at bank 0: ldmatrix reads the same 16-byte column of 8 rows,
// which is an 8-way bank conflict. XORing the chunk index with (row % 8) puts those 8
// chunks in 8 different 4-bank groups. Writes (cp.async: 8 consecutive chunks of a row)
// stay conflict-free because XOR with a constant is a permutation.
// __host__ too, so tests/cpp can check these properties on the CPU.
template <int D>
__host__ __device__ __forceinline__ constexpr int swizzle(int row, int chunk) {
    static_assert(D / 8 >= 8, "the XOR uses 3 bits of the chunk index: needs >= 8 chunks/row");
    return (row * D) + ((chunk ^ (row & 7)) << 3);
}

}  // namespace fa

namespace fa::ptx {

// Converts a generic pointer to shared memory into the 32-bit shared-window address that
// ldmatrix / cp.async expect.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cp.async: 16-byte global -> shared copy that bypasses registers (and L1, with .cg).
// src_bytes < 16 zero-fills the rest; src_bytes == 0 writes 16 zero bytes without reading
// global memory. That is how out-of-range rows (N not a multiple of the tile) become zeros.
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
// Wait until at most N committed groups are still in flight.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ldmatrix .x4: loads four 8x8 b16 matrices. Lanes 8i..8i+7 supply the 8 row addresses of
// matrix i; afterwards register i of every lane holds 2 elements of matrix i, in exactly the
// layout an mma.sync operand fragment needs (lane holds row lane/4, cols 2*(lane%4)+{0,1}).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
// Same, but each 8x8 matrix is transposed on the way into registers (used for V, whose
// rows are keys = the mma "k" dimension).
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// D = A * B + D, A 16x16 (row-major), B 16x8 (col-major), fp16 inputs, fp32 accumulate.
// Fragment layout (g = lane/4, t = lane%4):
//   a[0]: A[g][2t..2t+1]    a[1]: A[g+8][2t..]    a[2]: A[g][2t+8..]    a[3]: A[g+8][2t+8..]
//   b[0]: B[2t..2t+1][g]    b[1]: B[2t+8..][g]
//   d[0..1]: D[g][2t..2t+1] d[2..3]: D[g+8][2t..2t+1]
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Same shape, fp16 accumulate: C/D are 2 registers of half2 (d[0]: row g, d[1]: row g+8,
// columns 2t..2t+1). GeForce GPUs (RTX 30/40/50) run this at twice the rate of the fp32-
// accumulate version; datacenter GPUs run both at the same rate.
__device__ __forceinline__ void mma_16816_f16acc(uint32_t (&d)[2], const uint32_t (&a)[4],
                                                 uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Packs two floats into one register holding a half2 (lo in the low 16 bits).
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Unpacks a register holding a half2 into two floats.
__device__ __forceinline__ float2 unpack_half2(uint32_t r) {
    return __half22float2(*reinterpret_cast<const __half2*>(&r));
}

// Two 2^x in one SFU (special function unit) instruction, on a packed half2.
__device__ __forceinline__ uint32_t ex2_f16x2(uint32_t x) {
    uint32_t y = 0;  // NOLINT(misc-const-correctness): written by the asm statement
    asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(y) : "r"(x));
    return y;
}

// 2^x on the FMA pipe instead of the SFU (the FlashAttention-4 trick): the SFU does only 16
// exp2 per clock per SM, the FMA units 128 FMAs. 2^x = 2^floor(x) * 2^frac(x), with 2^frac by
// a degree-3 polynomial (max relative error ~1e-4, below fp16's resolution of 4.9e-4, and P is
// rounded to fp16 anyway); 2^floor(x) is added straight into the exponent bits.
__device__ __forceinline__ float exp2_poly3(float x) {
    x = fmaxf(x, -127.0F);  // 2^-127 underflows to 0; also maps -inf (masked) to 0
    const float xi = floorf(x);
    const float f = x - xi;  // in [0, 1)
    float p = fmaf(f, 0.07944023841053369F, 0.224494337302845F);
    p = fmaf(p, f, 0.6960656421638072F);
    p = fmaf(p, f, 1.0F);
    return __int_as_float(__float_as_int(p) + (static_cast<int>(xi) << 23));
}

}  // namespace fa::ptx
