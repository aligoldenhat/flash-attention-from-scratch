// FlashAttention-2 forward pass, written from scratch for Ampere / Ada / consumer Blackwell
// (sm_80+): mma.sync tensor cores, ldmatrix, cp.async, XOR-swizzled shared memory.
//
//   O = softmax(Q K^T * scale) V        Q, K, V, O: [B, H, N, D] fp16, fp32 accumulation
//
// Work decomposition (FA2):
//   * One thread block = one (batch, head) pair x BR = 16 * MT * WARPS consecutive query rows.
//   * Each warp owns MT * 16 query rows for the whole kernel, so the softmax statistics of a
//     row live in one warp and no inter-warp reduction is ever needed (the FA2 "split Q"
//     scheme, versus FA1's "split K", which needs a reduction through shared memory).
//   * The block streams over the keys in tiles of BC rows: S = Q K_j^T, online softmax,
//     O += P V_j. S and P never leave registers.
//
// Optimizations, and where to find them:
//   1. Tensor cores via inline-PTX mma.sync.m16n8k16 (fp16 in, fp32 accumulate).   -> mma_16816
//   2. ldmatrix.x4 to build operand fragments in one instruction (.trans for V).    -> ldmatrix_x4*
//   3. MT m-tiles per warp: every K / V fragment loaded from shared memory feeds MT
//      mma instructions, cutting shared-memory traffic per FLOP by MT.             -> MT loops
//   4. Q optionally held in registers for the whole kernel (Q_IN_REGS), else re-read
//      from shared memory each tile (fewer registers, used when MT * D is large).  -> q_frag
//   5. The S accumulator is re-packed in registers as the A operand of P V: the
//      C-fragment layout of mma m16n8 matches the A-fragment layout, so P never
//      touches shared memory.                                                      -> p_h
//   6. cp.async with the FA2 pipeline: V_j loads while S = Q K_j^T runs, K_{j+1}
//      loads while softmax and P V_j run.                                          -> tile()
//   7. XOR swizzle of 16-byte chunks in shared memory: ldmatrix and cp.async are
//      both bank-conflict-free with no padding.                                    -> swizzle()
//   8. Softmax in base 2: exp(x*scale - m) == exp2(x*scale*log2e - m'), and
//      x*c - m is one FMA feeding ex2.approx.                                      -> softmax
//   9. The row sum l stays a per-thread partial sum; the 4-lane reduction happens
//      once at the end instead of once per tile.                                   -> epilogue
//  10. Masking only on tiles that need it (the last tile when N % BC != 0, and the
//      diagonal tiles when causal). Causal blocks also skip tiles above the diagonal
//      and run in reverse order, so the longest blocks start first.                -> need_mask
//  11. Output is staged through shared memory and written with 16-byte coalesced
//      stores.                                                                     -> epilogue
//
// Beyond FA2: optional features selected per variant by the Opt template parameter
// (docs/optimizations.md, "Beyond FA2", has the measurements):
//  12. PEEL:   masked tiles run in a separate loop, so the main loop has no mask code.
//  13. STAGES: 2 = double-buffered K and V; the whole next tile is prefetched while the
//              current one is computed (2 barriers per tile instead of 3).
//  14. LAZY:   FA4's conditional rescaling: O is only rescaled when a row max grows by more
//              than 2^8; otherwise the stale max is kept (exact: l uses the same max).
//  15. EXP16:  ex2.approx.f16x2 computes two probabilities per SFU instruction and gives
//              P already packed as the fp16 mma operand.
//  16. EMU:    FA4's software exp2: half of the exponentials run as a polynomial on the FMA
//              pipe, splitting the load between the FMA units and the SFU.
//  17. ACC16:  P V accumulated in fp16 (2x tensor throughput on GeForce GPUs) for one tile,
//              then added into the fp32 O accumulator: only BC/16 fp16 roundings per tile,
//              and no fp16 overflow however long the sequence.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common.cuh"
#include "../flash_fwd.h"

namespace fa {
namespace {

// Tile shape.
template <int D_, int WARPS_, int MT_, int BC_, bool Q_IN_REGS_, int STAGES_ = 1>
struct Config {
    static constexpr int D = D_;          // head dim
    static constexpr int WARPS = WARPS_;  // warps per block
    static constexpr int MT = MT_;        // 16-row m-tiles per warp
    static constexpr int THREADS = 32 * WARPS;
    static constexpr int WARP_ROWS = 16 * MT;      // query rows per warp
    static constexpr int BR = WARP_ROWS * WARPS;   // query rows per block
    static constexpr int BC = BC_;                 // key rows per tile
    static constexpr bool Q_IN_REGS = Q_IN_REGS_;  // keep Q fragments in registers
    static constexpr int CHUNKS = D / 8;           // 16-byte chunks (8 halfs) per row
    // Pipeline depth: part of the tile config because it costs shared memory, and whether
    // that is affordable depends on the tile (it halves occupancy at d = 128, BR = 128).
    static constexpr int STAGES = STAGES_;
    static_assert(STAGES == 1 || STAGES == 2, "1 (FA2 pipeline) or 2 (double-buffered K/V)");
    static_assert(D == 64 || D == 128, "head dim must be 64 or 128");
    static_assert(BC % 16 == 0, "BC must be a multiple of the mma K/N tile");
};

// Optional features (numbered 12-17 in the header comment).
template <bool PEEL_, bool LAZY_, bool EXP16_, bool EMU_, bool ACC16_>
struct Opt {
    static constexpr bool PEEL = PEEL_;
    static constexpr bool LAZY = LAZY_;
    static constexpr bool EXP16 = EXP16_;
    static constexpr bool EMU = EMU_;
    static constexpr bool ACC16 = ACC16_;
    // LAZY lets p grow to 2^8; a 64-term fp16 sum of p * v could then overflow fp16.
    static_assert(!(LAZY && ACC16), "lazy rescaling needs fp32 accumulation");
};

template <class Cfg, class O>
constexpr int smem_bytes() {
    return (Cfg::BR + (2 * Cfg::STAGES * Cfg::BC)) * Cfg::D * static_cast<int>(sizeof(half));
}

// The swizzled shared-memory layout, swizzle<D>(row, chunk), is in common.cuh.

// Asynchronously copies rows [row0, row0 + ROWS) of a [n][D] matrix into a swizzled shared
// tile. Consecutive threads take consecutive 16-byte chunks: fully coalesced global reads.
// Rows >= n are zero-filled (cp.async with src-size 0), which keeps padded K rows from
// producing garbage scores and padded V rows from turning 0 * garbage into NaN.
template <int ROWS, int D, int THREADS>
__device__ __forceinline__ void load_tile(half* smem, const half* gmem, int row0, int n, int tid) {
    constexpr int CHUNKS = D / 8;
    constexpr int TOTAL = ROWS * CHUNKS;
    static_assert(TOTAL % THREADS == 0, "tile must split evenly over the threads");
#pragma unroll
    for (int i = 0; i < TOTAL / THREADS; ++i) {
        const int idx = tid + (i * THREADS);
        const int r = idx / CHUNKS;
        const int c = idx % CHUNKS;
        const bool valid = row0 + r < n;
        // An in-bounds address even when nothing is read, to be safe.
        const half* src = gmem + (valid ? (static_cast<size_t>(row0 + r) * D) + (c * 8) : 0);
        ptx::cp_async_16(ptx::smem_addr(smem + swizzle<D>(r, c)), src, valid ? 16 : 0);
    }
}

// A fragment (16 x 16) of Q rows [row0, row0+16), columns [16kk, 16kk+16).
// ldmatrix lane -> row address: lanes 0-15 rows 0-15 at chunk 2kk, lanes 16-31 rows 0-15 at
// chunk 2kk+1; the four 8x8 matrices come out as a[0..3] of the mma A fragment.
template <int D>
__device__ __forceinline__ void load_q_frag(uint32_t (&a)[4], const half* s_q, int row0, int kk,
                                            int lane) {
    const int row = row0 + (lane & 15);
    const int chunk = (kk * 2) + (lane >> 4);
    ptx::ldmatrix_x4(a, ptx::smem_addr(s_q + swizzle<D>(row, chunk)));
}

// B fragments of V for keys [16kk, 16kk+16) and head-dim columns [16dp, 16dp+16): V is
// [key][d] row-major, the mma wants k = key, so ldmatrix.trans. b[0..1]: n-tile 2dp,
// b[2..3]: n-tile 2dp+1.
template <int D>
__device__ __forceinline__ void load_v_frag(uint32_t (&b)[4], const half* s_v, int kk, int dp,
                                            int lane) {
    const int row = (kk * 16) + (lane & 7) + (((lane >> 3) & 1) << 3);
    const int chunk = (dp * 2) + (lane >> 4);
    ptx::ldmatrix_x4_trans(b, ptx::smem_addr(s_v + swizzle<D>(row, chunk)));
}

// One long kernel on purpose: the loops are fully unrolled into straight-line register code,
// and splitting it into functions would only hide the data flow between the stages.
template <class Cfg, class O, bool CAUSAL>
__global__ void __launch_bounds__(Cfg::THREADS)
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    flash_fwd_kernel(const half* __restrict__ q, const half* __restrict__ k,
                     const half* __restrict__ v, half* __restrict__ o, int n, float scale_log2) {
    constexpr int D = Cfg::D;
    constexpr int MT = Cfg::MT;
    constexpr int BR = Cfg::BR;
    constexpr int BC = Cfg::BC;
    constexpr int THREADS = Cfg::THREADS;
    constexpr int CHUNKS = Cfg::CHUNKS;
    constexpr int WARP_ROWS = Cfg::WARP_ROWS;
    constexpr int STAGES = Cfg::STAGES;

    // Dynamic shared memory is declared this way in CUDA (size set at launch); it is per
    // block, not a real global, and the kernel writes it, so it cannot be const.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    extern __shared__ __align__(128) unsigned char smem_raw[];
    half* s_q = reinterpret_cast<half*>(smem_raw);  // [BR][D], later reused for the output
    half* s_k0 = s_q + (BR * D);                    // [STAGES][BC][D]
    half* s_v0 = s_k0 + (STAGES * BC * D);          // [STAGES][BC][D]

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g = lane >> 2;  // "group": row within an 8-row half of the mma tile
    const int t = lane & 3;   // thread in group: selects a column pair

    // Causal: block x does (x+1) tiles of work, so launch the biggest ones first.
    const int m_block =
        CAUSAL ? static_cast<int>(gridDim.x - 1 - blockIdx.x) : static_cast<int>(blockIdx.x);
    const int q0 = m_block * BR;
    const size_t bh_offset = static_cast<size_t>(blockIdx.y) * static_cast<size_t>(n) * D;
    q += bh_offset;
    k += bh_offset;
    v += bh_offset;
    o += bh_offset;

    // Causal: keys beyond the last query row of this block are fully masked; skip them.
    const int kv_end = CAUSAL ? min(n, q0 + BR) : n;
    const int n_tiles = (kv_end + BC - 1) / BC;

    // ---- Prologue: Q and K_0 (and V_0 when double-buffered) -> shared, Q -> registers ----
    load_tile<BR, D, THREADS>(s_q, q, q0, n, tid);
    load_tile<BC, D, THREADS>(s_k0, k, 0, n, tid);
    if constexpr (STAGES == 2) {
        load_tile<BC, D, THREADS>(s_v0, v, 0, n, tid);
    }
    ptx::cp_async_commit();
    ptx::cp_async_wait<0>();
    __syncthreads();

    const int warp_row0 = warp * WARP_ROWS;  // first row of this warp inside the block tile

    // Q A fragments. With Q_IN_REGS they are loaded once here; otherwise this array is
    // dead code and the fragments are re-read from shared memory in the S loop.
    uint32_t q_frag[Cfg::Q_IN_REGS ? MT : 1][Cfg::Q_IN_REGS ? D / 16 : 1][4];
    if constexpr (Cfg::Q_IN_REGS) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk) {
                load_q_frag<D>(q_frag[mt][kk], s_q, warp_row0 + (mt * 16), kk, lane);
            }
        }
    }

    // O accumulator: per m-tile, 16 x D as D/8 mma C fragments (rows g and g+8).
    float acc_o[MT][D / 8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int dn = 0; dn < D / 8; ++dn) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc_o[mt][dn][e] = 0.0F;
            }
        }
    }
    // Running max (already in log2 units, i.e. multiplied by scale*log2e) and the running
    // per-thread partial row sum, for rows g ([.][0]) and g+8 ([.][1]) of each m-tile.
    float row_max[MT][2];
    float row_sum[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        row_max[mt][0] = row_max[mt][1] = -INFINITY;
        row_sum[mt][0] = row_sum[mt][1] = 0.0F;
    }

    // ---- One key/value tile -------------------------------------------------------------
    // mask_tag is std::true_type / std::false_type: with PEEL the unmasked tiles get a copy
    // of this body with no mask code at all. need_mask is the runtime check otherwise.
    // NOLINTNEXTLINE(readability-function-cognitive-complexity): same reason as the kernel
    auto tile = [&](int j, auto mask_tag, bool need_mask) {
        constexpr bool MASK_CODE = decltype(mask_tag)::value;
        const int k0 = j * BC;
        const int stage = (STAGES == 2) ? (j & 1) : 0;
        const half* s_k = s_k0 + (stage * BC * D);
        const half* s_v = s_v0 + (stage * BC * D);

        if constexpr (STAGES == 1) {
            // V_j starts loading now and overlaps with S = Q K_j^T.
            load_tile<BC, D, THREADS>(s_v0, v, k0, n, tid);
            ptx::cp_async_commit();
        } else {
            // Prefetch the whole next tile into the other buffer, then wait for this one
            // (issued one iteration ago, or in the prologue).
            if (j + 1 < n_tiles) {
                const int next = (j + 1) & 1;
                load_tile<BC, D, THREADS>(s_k0 + (next * BC * D), k, k0 + BC, n, tid);
                load_tile<BC, D, THREADS>(s_v0 + (next * BC * D), v, k0 + BC, n, tid);
            }
            ptx::cp_async_commit();
            ptx::cp_async_wait<1>();
            __syncthreads();
        }

        // S = Q K_j^T: per m-tile 16 x BC, BC/8 C fragments.
        // B operand = K^T, i.e. "col-major B" == K row-major, so ldmatrix (no .trans) on K.
        // One ldmatrix.x4 covers two 8-key n-tiles x the two 8-wide halves of a k step,
        // and is reused by all MT m-tiles.
        float s[MT][BC / 8][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < BC / 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    s[mt][nt][e] = 0.0F;
                }
            }
        }
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            // A fragments for this k step (register copies are free after unrolling).
            uint32_t a_q[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if constexpr (Cfg::Q_IN_REGS) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        a_q[mt][e] = q_frag[mt][kk][e];
                    }
                } else {
                    load_q_frag<D>(a_q[mt], s_q, warp_row0 + (mt * 16), kk, lane);
                }
            }
#pragma unroll
            for (int np = 0; np < BC / 16; ++np) {
                uint32_t b[4];
                const int row = (np * 16) + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = (kk * 2) + ((lane >> 3) & 1);
                ptx::ldmatrix_x4(b, ptx::smem_addr(s_k + swizzle<D>(row, chunk)));
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    ptx::mma_16816(s[mt][2 * np], a_q[mt], b[0], b[1]);
                    ptx::mma_16816(s[mt][(2 * np) + 1], a_q[mt], b[2], b[3]);
                }
            }
        }

        if constexpr (STAGES == 1) {
            // All warps are done reading K_j: K_{j+1} can start loading and overlap with the
            // softmax and P V_j. An empty group is committed on the last tile so the
            // wait_group counts below are the same every iteration.
            __syncthreads();
            if (j + 1 < n_tiles) {
                load_tile<BC, D, THREADS>(s_k0, k, k0 + BC, n, tid);
            }
            ptx::cp_async_commit();
        }

        // Masking: keys >= n (last tile) and, when causal, keys after the query.
        if constexpr (MASK_CODE) {
            if (need_mask) {
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const int row_a = q0 + warp_row0 + (mt * 16) + g;  // global query index
#pragma unroll
                    for (int nt = 0; nt < BC / 8; ++nt) {
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int col = k0 + (nt * 8) + (2 * t) + (e & 1);
                            const int row = row_a + ((e >> 1) * 8);
                            if (col >= n || (CAUSAL && col > row)) {
                                s[mt][nt][e] = -INFINITY;
                            }
                        }
                    }
                }
            }
        }

        // Online softmax. A row is spread over the 4 lanes of a group (t = 0..3), so the
        // max needs two butterfly shuffles; the sum can wait (see the epilogue).
        // P comes out packed as fp16 pairs: p_h[mt][nt][0] = row g, [1] = row g+8.
        uint32_t p_h[MT][BC / 8][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int nt = 0; nt < BC / 8; ++nt) {
                tile_max[0] = fmaxf(tile_max[0], fmaxf(s[mt][nt][0], s[mt][nt][1]));
                tile_max[1] = fmaxf(tile_max[1], fmaxf(s[mt][nt][2], s[mt][nt][3]));
            }
#pragma unroll
            for (float& m : tile_max) {
                m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, 2));
            }

            // FA4 conditional rescaling: if no row of this warp grows its max by more than
            // 2^8 (8 in log2 units), keep the old max: p may then reach 2^8, which fp32 O
            // and fp16 P hold easily, and skipping saves the rescale of all of O.
            const bool rescale = [&] {
                if constexpr (O::LAZY) {
                    const bool grow = (tile_max[0] * scale_log2 > row_max[mt][0] + 8.0F) ||
                                      (tile_max[1] * scale_log2 > row_max[mt][1] + 8.0F);
                    return __any_sync(0xffffffffU, grow) != 0;
                } else {
                    return true;
                }
            }();
            if (rescale) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const float new_max = fmaxf(row_max[mt][i], tile_max[i] * scale_log2);
                    // A row that has seen only masked keys keeps max = -inf; use 0 so that
                    // (-inf) - (-inf) = NaN never happens. Its p values are exp2(-inf) = 0.
                    const float alpha =
                        exp2f(row_max[mt][i] - ((new_max == -INFINITY) ? 0.0F : new_max));
                    row_max[mt][i] = new_max;
                    row_sum[mt][i] *= alpha;
#pragma unroll
                    for (int dn = 0; dn < D / 8; ++dn) {
                        acc_o[mt][dn][2 * i] *= alpha;
                        acc_o[mt][dn][(2 * i) + 1] *= alpha;
                    }
                }
            }
            const float max_used[2] = {(row_max[mt][0] == -INFINITY) ? 0.0F : row_max[mt][0],
                                       (row_max[mt][1] == -INFINITY) ? 0.0F : row_max[mt][1]};

#pragma unroll
            for (int nt = 0; nt < BC / 8; ++nt) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const float x0 = fmaf(s[mt][nt][2 * i], scale_log2, -max_used[i]);
                    const float x1 = fmaf(s[mt][nt][(2 * i) + 1], scale_log2, -max_used[i]);
                    if (O::EMU && (nt & 1) == 1) {
                        // Every other n-tile on the FMA pipe instead of the SFU.
                        const float p0 = ptx::exp2_poly3(x0);
                        const float p1 = ptx::exp2_poly3(x1);
                        row_sum[mt][i] += p0 + p1;
                        p_h[mt][nt][i] = ptx::pack_half2(p0, p1);
                    } else if constexpr (O::EXP16) {
                        // Two exponentials in one SFU instruction, result already packed.
                        p_h[mt][nt][i] = ptx::ex2_f16x2(ptx::pack_half2(x0, x1));
                        const float2 p = ptx::unpack_half2(p_h[mt][nt][i]);
                        row_sum[mt][i] += p.x + p.y;
                    } else {
                        const float p0 = exp2f(x0);
                        const float p1 = exp2f(x1);
                        row_sum[mt][i] += p0 + p1;
                        p_h[mt][nt][i] = ptx::pack_half2(p0, p1);
                    }
                }
            }
        }

        if constexpr (STAGES == 1) {
            // Wait for V_j (the newest group, K_{j+1}, may stay in flight).
            ptx::cp_async_wait<1>();
            __syncthreads();
        }

        // O += P V_j. P as the A operand: the C fragments of key n-tiles 2kk and 2kk+1 are
        // exactly the a[0..1] and a[2..3] halves of a 16x16 A fragment over keys
        // 16kk..16kk+15.
        auto a_of = [&](int mt, int kk, uint32_t (&a)[4]) {
            a[0] = p_h[mt][2 * kk][0];
            a[1] = p_h[mt][2 * kk][1];
            a[2] = p_h[mt][(2 * kk) + 1][0];
            a[3] = p_h[mt][(2 * kk) + 1][1];
        };
        if constexpr (O::ACC16) {
            // dp outer, kk inner: the fp16 partial sums of one 16-column slice live only
            // for BC/16 mma steps, then get added into fp32 O (few extra registers).
#pragma unroll
            for (int dp = 0; dp < D / 16; ++dp) {
                uint32_t part[MT][2][2] = {};
#pragma unroll
                for (int kk = 0; kk < BC / 16; ++kk) {
                    uint32_t b[4];
                    load_v_frag<D>(b, s_v, kk, dp, lane);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        uint32_t a[4];
                        a_of(mt, kk, a);
                        ptx::mma_16816_f16acc(part[mt][0], a, b[0], b[1]);
                        ptx::mma_16816_f16acc(part[mt][1], a, b[2], b[3]);
                    }
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 lo = ptx::unpack_half2(part[mt][h][0]);  // row g
                        const float2 hi = ptx::unpack_half2(part[mt][h][1]);  // row g+8
                        float (&acc)[4] = acc_o[mt][(2 * dp) + h];
                        acc[0] += lo.x;
                        acc[1] += lo.y;
                        acc[2] += hi.x;
                        acc[3] += hi.y;
                    }
                }
            }
        } else {
#pragma unroll
            for (int kk = 0; kk < BC / 16; ++kk) {
                uint32_t a[MT][4];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    a_of(mt, kk, a[mt]);
                }
#pragma unroll
                for (int dp = 0; dp < D / 16; ++dp) {
                    uint32_t b[4];
                    load_v_frag<D>(b, s_v, kk, dp, lane);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        ptx::mma_16816(acc_o[mt][2 * dp], a[mt], b[0], b[1]);
                        ptx::mma_16816(acc_o[mt][(2 * dp) + 1], a[mt], b[2], b[3]);
                    }
                }
            }
        }

        if constexpr (STAGES == 1) {
            // K_{j+1} has landed, and every warp is done with V_j before it is overwritten.
            ptx::cp_async_wait<0>();
        }
        // STAGES == 2: every warp is done with this stage before the next tile's prefetch
        // overwrites it.
        __syncthreads();
    };

    // ---- Main loop over key/value tiles ---------------------------------------------------
    auto needs_mask = [&](int j) {
        const int k0 = j * BC;
        return (k0 + BC > n) || (CAUSAL && (k0 + BC - 1 > q0));
    };
    if constexpr (O::PEEL) {
        // Tiles [0, first_masked) need no mask, the rest (the last tile when n % BC != 0,
        // the diagonal tiles when causal) do. (q0 + 1) / BC is the first tile with a key
        // after query q0, i.e. the first one crossing the diagonal.
        int first_masked = (n % BC != 0) ? n_tiles - 1 : n_tiles;
        if constexpr (CAUSAL) {
            first_masked = min(first_masked, (q0 + 1) / BC);
        }
        for (int j = 0; j < first_masked; ++j) {
            tile(j, std::false_type{}, false);
        }
        for (int j = first_masked; j < n_tiles; ++j) {
            tile(j, std::true_type{}, true);
        }
    } else {
        for (int j = 0; j < n_tiles; ++j) {
            tile(j, std::true_type{}, needs_mask(j));
        }
    }

    // ---- Epilogue: normalize, stage through shared memory, 16-byte stores ---------------
    // Each warp writes only its own rows of s_q, which no other warp ever reads, so a
    // __syncwarp (not __syncthreads) is enough before reading them back.
    half* s_o = s_q;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        float inv_sum[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float l = row_sum[mt][i];
            l += __shfl_xor_sync(0xffffffffU, l, 1);
            l += __shfl_xor_sync(0xffffffffU, l, 2);
            inv_sum[i] = (l > 0.0F) ? (1.0F / l) : 0.0F;
        }
        const int r_local = warp_row0 + (mt * 16) + g;
#pragma unroll
        for (int dn = 0; dn < D / 8; ++dn) {
            *reinterpret_cast<uint32_t*>(s_o + swizzle<D>(r_local, dn) + (2 * t)) =
                ptx::pack_half2(acc_o[mt][dn][0] * inv_sum[0], acc_o[mt][dn][1] * inv_sum[0]);
            *reinterpret_cast<uint32_t*>(s_o + swizzle<D>(r_local + 8, dn) + (2 * t)) =
                ptx::pack_half2(acc_o[mt][dn][2] * inv_sum[1], acc_o[mt][dn][3] * inv_sum[1]);
        }
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < (WARP_ROWS * CHUNKS) / 32; ++i) {
        const int idx = lane + (i * 32);
        const int r = idx / CHUNKS;
        const int c = idx % CHUNKS;
        const int row = q0 + warp_row0 + r;
        if (row < n) {
            *reinterpret_cast<uint4*>(o + (static_cast<size_t>(row) * D) + (c * 8)) =
                *reinterpret_cast<const uint4*>(s_o + swizzle<D>(warp_row0 + r, c));
        }
    }
}

template <class Cfg, class O, bool CAUSAL>
void launch(const FlashFwdParams& p, cudaStream_t stream) {
    auto* kernel = flash_fwd_kernel<Cfg, O, CAUSAL>;
    constexpr int smem = smem_bytes<Cfg, O>();
    // More than 48 KB of dynamic shared memory must be opted into explicitly.
    if constexpr (smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    const dim3 grid(static_cast<unsigned>((p.seqlen + Cfg::BR - 1) / Cfg::BR),
                    static_cast<unsigned>(p.batch * p.heads));
    const float scale_log2 = p.softmax_scale * std::numbers::log2e_v<float>;
    kernel<<<grid, Cfg::THREADS, smem, stream>>>(
        static_cast<const half*>(p.q), static_cast<const half*>(p.k), static_cast<const half*>(p.v),
        static_cast<half*>(p.o), p.seqlen, scale_log2);
    CUDA_CHECK_LAUNCH();
}

// A kernel set = optional features + one tile shape per (head dim, causal). Non-causal and causal
// use different tiles: causal blocks waste work on the diagonal tile, which favours smaller BR.
template <class O, class D64, class D64Causal, class D128, class D128Causal>
struct KernelSet {
    static void run(const FlashFwdParams& p, cudaStream_t stream) {
        if (p.head_dim == 64) {
            p.causal ? launch<D64Causal, O, true>(p, stream) : launch<D64, O, false>(p, stream);
        } else {
            p.causal ? launch<D128Causal, O, true>(p, stream) : launch<D128, O, false>(p, stream);
        }
    }
};

// Tile configurations, chosen by benchmark (bench/tune.sh); see docs/benchmarking.md.
// clang-format off
// Features: measured one at a time with bench/ablate.sh (docs/fa3_fa4_techniques.md).
//                    PEEL   LAZY   EXP16  EMU    ACC16
using BaseOpt = Opt<false, false, false, false, false>;
using OptOpt  = Opt<false, true,  false, false, false>;
using FastOpt = Opt<false, false, true,  false, true>;
//                                D  warps MT  BC  Q in regs  stages
using BaseD64        = Config<64,  4, 2, 64, true,  1>;
using BaseD64Causal  = Config<64,  4, 2, 64, true,  1>;
using BaseD128       = Config<128, 4, 2, 32, false, 1>;
using BaseD128Causal = Config<128, 4, 1, 64, true,  1>;
using OptD64         = Config<64,  4, 2, 64, true,  1>;
using OptD64Causal   = Config<64,  4, 2, 64, true,  1>;
using OptD128        = Config<128, 4, 2, 32, false, 1>;
using OptD128Causal  = Config<128, 4, 1, 64, true,  1>;
using FastD64        = Config<64,  4, 2, 32, true,  2>;
using FastD64Causal  = Config<64,  4, 2, 32, true,  2>;
using FastD128       = Config<128, 4, 1, 64, true,  1>;
using FastD128Causal = Config<128, 4, 1, 64, true,  1>;
// clang-format on

using BaseSet = KernelSet<BaseOpt, BaseD64, BaseD64Causal, BaseD128, BaseD128Causal>;
using OptSet = KernelSet<OptOpt, OptD64, OptD64Causal, OptD128, OptD128Causal>;
using FastSet = KernelSet<FastOpt, FastD64, FastD64Causal, FastD128, FastD128Causal>;

bool aligned16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

}  // namespace

void flash_fwd(const FlashFwdParams& p, cudaStream_t stream) {
    if (p.batch <= 0 || p.heads <= 0 || p.seqlen <= 0) {
        throw std::invalid_argument("flash_fwd: batch, heads and seqlen must be positive");
    }
    if (p.batch * p.heads > 65535) {
        throw std::invalid_argument("flash_fwd: batch * heads must be <= 65535 (grid.y limit)");
    }
    if (!aligned16(p.q) || !aligned16(p.k) || !aligned16(p.v) || !aligned16(p.o)) {
        throw std::invalid_argument("flash_fwd: tensors must be 16-byte aligned (cp.async)");
    }
    if (p.head_dim != 64 && p.head_dim != 128) {
        throw std::invalid_argument("flash_fwd: head_dim must be 64 or 128, got " +
                                    std::to_string(p.head_dim));
    }
    switch (p.variant) {
        case Variant::kBaseline:
            BaseSet::run(p, stream);
            break;
        case Variant::kOpt:
            OptSet::run(p, stream);
            break;
        case Variant::kFp16Acc:
            FastSet::run(p, stream);
            break;
        default:
            throw std::invalid_argument("flash_fwd: unknown variant");
    }
}

}  // namespace fa
