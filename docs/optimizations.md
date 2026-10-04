# How the kernel works, and why each optimization is there

Source: [`csrc/kernels/flash_fwd.cu`](../csrc/kernels/flash_fwd.cu), PTX wrappers in
[`csrc/common.cuh`](../csrc/common.cuh). This page walks through the kernel in the order it
executes, and for each design choice gives the problem it solves.

## 1. The problem: attention is memory-bound if you write it naively

`O = softmax(Q Kᵀ / √d) V`, with Q, K, V of shape `[B, H, N, d]`.

The naive version (three kernels: matmul, softmax, matmul) writes the `N × N` score matrix S
and the probabilities P to DRAM and reads them back. For N = 4096, B·H = 64 heads that is
64 · 4096² · 2 bytes = **2 GB per matrix** — more than the 3050's 4 GB can hold twice, which is
why `torch naive` runs out of memory in the benchmark from N = 2048. Its arithmetic intensity
is low: every score is written once and read twice for only O(d) FLOPs of work.

FlashAttention never materialises S. It tiles Q into blocks of rows, streams K and V through
shared memory, and keeps a **running (online) softmax**, so the only DRAM traffic is reading
Q, K, V and writing O: `O(N·d)` instead of `O(N²)`. That turns attention into a
compute-bound kernel, and the job becomes keeping the tensor cores busy.

## 2. Online softmax (the math that makes tiling possible)

For one query row, process keys in tiles. Keep the running max `m` and running sum `l` of
`exp(s - m)`, and the unnormalised output `acc`. For a new tile with scores `s_j`:

```
m_new = max(m, max_j s_j)
alpha = exp(m - m_new)            # rescales everything computed with the old max
p_j   = exp(s_j - m_new)
l     = alpha * l + sum_j p_j
acc   = alpha * acc + sum_j p_j v_j
```

At the end `O = acc / l`. Subtracting the max keeps `exp` from overflowing (a test feeds
scores in the hundreds to check this); `alpha` fixes up previous tiles whenever the max grows.

## 3. Work decomposition (FA2 vs FA1)

- Grid: `(ceil(N / BR), B·H)`. One block handles BR query rows of one head.
- Each **warp owns 16·MT complete query rows** for the whole kernel. All scores of a row are
  computed by that warp, so the row max and row sum need only shuffles between the 4 lanes that
  hold a row — no shared-memory reduction between warps and no `__syncthreads` for the softmax.
  (FA1 split K/V across warps, which needs exactly that reduction. This is the main FA2 change.)

## 4. Tensor cores: `mma.sync.m16n8k16` + `ldmatrix`

Both matmuls (S = Q Kᵀ and O += P V) run on tensor cores through inline PTX
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`: a warp multiplies a 16×16 fp16 tile by a
16×8 fp16 tile and accumulates into 16×8 fp32. The register layout of every operand is fixed
by the PTX ISA (g = lane / 4, t = lane % 4):

| operand | register | holds |
|---|---|---|
| A (16×16) | a0, a1, a2, a3 | A[g][2t..2t+1], A[g+8][2t..], A[g][2t+8..], A[g+8][2t+8..] |
| B (16×8)  | b0, b1 | B[2t..2t+1][g], B[2t+8..2t+9][g] |
| C (16×8)  | c0..c3 | C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1] |

`ldmatrix.x4` loads four 8×8 fp16 matrices from shared memory directly into this layout in a
single instruction: lanes 8i..8i+7 give the row addresses of matrix i.

- **Q (A operand)**: lanes 0–15 point at rows 0–15, lanes 16–31 at the same rows 8 columns
  further: the 4 matrices come out as a0..a3.
- **K (B operand of Q Kᵀ)**: `B = Kᵀ` as a "column-major" 16×8 matrix is just K stored
  row-major, so plain `ldmatrix` works. One `.x4` gives the B fragments of two 8-key tiles.
- **V (B operand of P V)**: V is `[key][d]` row-major but the mma needs it "column-major" over
  keys, so `ldmatrix.trans` transposes each 8×8 matrix on the way into registers.

## 5. P never leaves registers

The C fragment of `S = Q Kᵀ` for key tiles 2kk and 2kk+1 has exactly the register layout of
the A fragment of `P V` for keys 16kk..16kk+15 (compare the A and C rows of the table). So after
the softmax, P is converted to fp16 pairs (`pack_half2`) and fed straight into the second
mma. S and P cost zero shared memory and zero bandwidth.

## 6. Multiple m-tiles per warp (MT)

With MT = 2 each warp owns 32 rows. Every K or V fragment loaded with `ldmatrix` feeds two
mma instructions instead of one, halving shared-memory reads per FLOP. Shared-memory
bandwidth (128 bytes/clock/SM) is the next limit after the tensor cores, so this is worth
about 2–6% here (measured with `bench/tune.sh`). The cost is registers: the O accumulator doubles.
For d = 128 the kernel then re-reads Q from shared memory each tile instead of keeping it in
registers (`Q_IN_REGS = false`), which keeps it at 0 spills.

## 7. `cp.async` pipeline

`cp.async.cg.shared.global` copies 16 bytes from global to shared memory without going
through registers and without blocking the thread. The loop overlaps loads with compute in
the same order as FA2:

```
prologue: load Q, K_0; wait
for each tile j:
    issue V_j                         # loads while S = Q K_jᵀ runs
    S = Q K_jᵀ
    __syncthreads                     # everyone is done reading K_j
    issue K_{j+1}                     # loads while softmax and P V_j run
    mask, online softmax
    wait_group 1 + __syncthreads      # V_j landed (K_{j+1} may still be in flight)
    O += P V_j
    wait_group 0 + __syncthreads      # K_{j+1} landed, everyone done with V_j
```

An empty group is committed on the last tile so `wait_group` counts stay the same every
iteration. This uses only one K buffer and one V buffer (48 KB at d = 128), which keeps two
blocks resident per SM.

Out-of-range rows (when N is not a multiple of the tile) use `cp.async` with a source size of
0, which writes 16 zero bytes without reading memory. Zero K rows are then masked to −∞;
zero V rows matter too: with garbage V, `0 × NaN` would still give NaN.

## 8. Swizzled shared memory: no bank conflicts, no padding

Shared memory has 32 banks of 4 bytes. A row of the K tile is d·2 = 128 or 256 bytes, a
multiple of 128, so every row starts in bank 0. `ldmatrix` reads the same 16-byte column of
8 consecutive rows, so all 8 reads hit the same 4 banks: an **8-way bank conflict**.

The fix is to store 16-byte chunk `c` of row `r` at chunk position `c XOR (r mod 8)`. The same
logical column of 8 consecutive rows now lands in 8 different chunks, in 8 different bank
groups. `cp.async` writes stay conflict-free too, because XOR with a constant is a
permutation of the chunks in a row. Unlike padding, this wastes no shared memory.
`ncu` shows it directly: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` ≈ 0
(see `profile/ncu.sh`).

## 9. Cheap softmax

- **Base-2 exponent**: `exp(x·scale − m) = exp2(x·scale·log2e − m')`. The constant is folded
  into one `scale_log2`, so each probability is one FMA plus one `ex2.approx` (with
  `--use_fast_math`), the SFU (special-function unit) instruction.
- **Lazy row sum**: the 4 lanes that share a row each keep a partial sum; `alpha` is the same
  for all of them, so the 4-lane reduction can be done once in the epilogue, not once per tile.
  The row max does need the reduction every tile (two `__shfl_xor_sync`).
- **Masking only where needed**: the mask loop runs only on the last tile when N % BC ≠ 0 and,
  for causal, on tiles that cross the diagonal. All other tiles skip it.

## 10. Causal attention

- Blocks stop at the diagonal: tiles entirely above it are never loaded (half the work).
- Block indices are reversed, so the blocks with the most tiles (bottom of the matrix) are
  scheduled first and the short ones fill the gaps at the end (better tail balance).
- Causal uses its own tile shape: big blocks waste more work on the partially masked
  diagonal tile, so the best causal config can differ from the non-causal one.

## 11. Epilogue

The output is normalised by `1/l`, converted to fp16, written into this warp's (now unused)
rows of the Q shared-memory tile with the same swizzle, and copied to global memory with
16-byte (`uint4`) fully coalesced stores. Only the owning warp touches those rows, so
`__syncwarp` is enough.

## 12. Tile configuration

`Config<D, WARPS, MT, BC, Q_IN_REGS>`, chosen per (d, causal) with `bench/tune.sh`. See the
tuning table in [benchmarking.md](benchmarking.md#tile-tuning-benchtunesh-make-tune). On a new GPU, re-run `make tune` and update the four `using
ConfigD...` lines at the bottom of the kernel file.

## 13. Beyond FA2: variants and optional features

The kernel template takes a second parameter, `Opt`, a set of compile-time switches (numbered
12–17 in the kernel's header comment). Each switch is an `if constexpr`, so a disabled feature
leaves no trace in the machine code, and every combination is a separately compiled kernel.
Three combinations are built and selectable at runtime:

| Variant | API | Features | Purpose |
|---|---|---|---|
| `baseline` | `fa.forward_variant(..., variant="baseline")` | none | the first tuned FA2 kernel, kept as the reference point |
| `opt` | `fa.forward(...)` (default) | LAZY | exact fp32 accumulation |
| `fp16acc` | `fa.forward(..., fp16_accum=True)` | ACC16 + EXP16, STAGES=2 at d = 64 | P·V accumulated in fp16 per tile: 2× tensor rate on GeForce GPUs |

Every feature was measured on its own with `bench/ablate.sh` (tables in
[fa3_fa4_techniques.md](fa3_fa4_techniques.md#ablation-each-implemented-technique-on-its-own)).
The numbers quoted below are TFLOPS at N = 4096 on the RTX 3050, for
d64 / d64 causal / d128 / d128 causal.

### 13.1 ACC16: fp16 accumulation of P·V (the big win, +25–30%)

**Why it's faster.** GeForce tensor cores (RTX 30/40/50) run `mma.sync` with fp16 inputs at full
rate when the accumulator is fp16 and at half rate when it is fp32. Every fp32-accumulate kernel
on the 3050 (this one, FA2, cuDNN, cuBLAS) tops out around 14.5 TFLOPS for that reason. P·V is
half of the matmul work, so moving it to the fp16-accumulate instruction raises the ceiling.

**Why only P·V, not Q·Kᵀ.** Scores can be large (the large-logits test feeds hundreds), and fp16
near 100 has a spacing of 0.0625. An error in a score becomes a relative error in p after the
exponential. P, by contrast, lies in [0, 1], which bounds every sum P·V can produce.

**Two-level accumulation.** A plain fp16 accumulator over the whole sequence would round
thousands of times and could overflow 65504. Instead, each key tile gets a fresh fp16 partial
sum, which is folded into the fp32 accumulator as soon as the tile is done:

```
for each key tile j:                       (online-softmax rescale of fp32 acc_o happens before)
    part  = 0                              fp16, 2 registers per 8 columns
    part += P_j[:, 16 keys] · V_j          BC/16 fp16-accumulate mma steps
    acc_o += float(part)                   fp32, carries everything across tiles
```

- **Overflow bound.** A partial sum covers BC ≤ 64 keys with p ≤ 1, so
  |part| ≤ 64 · max|V|. That stays below 65504 for any |V| < 1000.
  `test_fp16acc_large_values_no_overflow` uses |V| up to about 400 at N = 4096.
- **Rounding.** The tensor core rounds its fp16 output once per mma step, so each output
  element sees only BC/16 roundings per tile (2 at BC = 32, 4 at BC = 64) before reaching fp32.
  Measured error stays around 5e-5, the same order as the fp32 path, against the 1e-2 tolerance.
- **No LAZY with ACC16.** Lazy rescaling lets p reach 2^8, which would make the bound
  64 · 256 · |V|. A `static_assert` forbids the combination.

**Loop order for registers.** The fp32 path runs keys outside and head-dim columns inside, and
accumulates straight into `acc_o`. With ACC16 that order would keep an fp16 partial for every
column of the tile alive at once (D/8 · 2 · MT registers: 32 for the fp16acc tiles). The ACC16 loop runs a
16-column slice of the head dim outside and the keys inside, so only one slice's partials
(2 · 2 · MT registers) are alive, and they are folded into `acc_o` right after:

```cpp
for (int dp = 0; dp < D / 16; ++dp) {          // 16 head-dim columns at a time
    uint32_t part[MT][2][2] = {};              // fp16 partials of those columns
    for (int kk = 0; kk < BC / 16; ++kk) {     // keys
        load_v_frag(b, s_v, kk, dp, lane);     // each V fragment still loaded exactly once
        mma_16816_f16acc(part[mt][0], a, b[0], b[1]);
        mma_16816_f16acc(part[mt][1], a, b[2], b[3]);
    }
    acc_o[mt][2*dp + h] += unpack_half2(part[mt][h]);   // into fp32
}
```

The cost of this order is a chain of BC/16 dependent mma instructions on one partial. The MT
m-tiles, the two n-tiles and the unrolled `dp` loop give the scheduler independent chains to
interleave.

Code: `if constexpr (O::ACC16)` in the P·V section of `tile()`; `mma_16816_f16acc` in
`common.cuh`.

### 13.2 STAGES=2: double-buffered K and V (+5% at d = 64, −12% at d = 128)

The FA2 pipeline (section 7) uses one K buffer and one V buffer and needs 3 barriers per tile.
With two buffers of each, the whole next tile is prefetched while the current one is computed:

```
prologue: load Q, K_0, V_0 into buffer 0; wait
for each tile j:                              buffer b = j % 2
    issue K_{j+1}, V_{j+1} into buffer 1-b    loads overlap ALL of tile j's compute
    wait_group 1 + __syncthreads              tile j's loads (issued one iteration ago) landed
    S = Q K_jᵀ, softmax, O += P V_j           read buffer b
    __syncthreads                             everyone is done with b before it is refilled
```

That's 2 barriers per tile instead of 3, and a prefetch distance of a full tile.

**Why it helps or hurts depends on occupancy.** Shared memory per block is
(BR + 2 · STAGES · BC) · D · 2 bytes, and an sm_86 SM has 100 KB:

| Tile (variant) | Registers / thread | Blocks/SM allowed by registers | Shared memory, STAGES 1 → 2 | Blocks/SM allowed by shared memory | Result |
|---|---|---|---|---|---|
| d = 64, MT = 2, BC = 32 (fp16acc) | 168 | 3 | 24 → 32 KB | 4 → 3 | **3 → 3: the second buffer is free** |
| d = 128, MT = 1, BC = 64 (fp16acc) | 252 | 2 | 48 → 80 KB | 2 → 1 | **2 → 1: occupancy halves**, −17% |
| d = 128, MT = 2, BC = 32 (opt) | 250 | 2 | 48 → 64 KB | 2 → 1 | **2 → 1**, −12% |

The gain at d = 64 was measured on the MT = 2, BC = 64 fp16acc tile (+2.6 to +5%), before
re-tuning picked BC = 32 with STAGES = 2. (Registers are allocated per warp; 65,536 per SM,
4 warps per block.) At d = 64 the register
file was already the limit, so the extra buffer costs nothing. At d = 128 it costs half the warps
that hide latency. That's why STAGES lives in the tile `Config`, not in `Opt`: whether it pays
off depends on the tile, and `bench/tune.sh` picks it per head dim.

Code: `if constexpr (STAGES == 2)` at the start of `tile()` and in the prologue.

### 13.3 LAZY: FA4 conditional rescaling (neutral here, kept in `opt`)

Every time a row's running max m grows, online softmax multiplies the whole O row and the row sum
by α = 2^(m_old − m_new): MT · D/8 · 4 multiplies per thread per tile, plus an exponential.

**Why a stale max is still exact.** The result is O = Σ p_j v_j / Σ p_j with
p_j = 2^(s_j − m). Any common value of m cancels between numerator and denominator. The max is
only subtracted to keep the numbers in range. So the kernel may keep an *old* max as long as p
can't overflow: it only updates when the new scores exceed the stored max by more than 8 (in log2
units), so p ≤ 2^8 = 256. That's trivial for fp32 sums and for fp16 P (max 65504).

**Why the decision is per warp.** If each thread decided for its own rows, the warp would
diverge and execute both paths, saving nothing. So the decision is made for all 32 threads at once:

```cpp
const bool grow = tile_max[0] * c > row_max[0] + 8.0F || tile_max[1] * c > row_max[1] + 8.0F;
const bool rescale = __any_sync(0xffffffff, grow);   // one vote for the whole warp
if (rescale) { /* normal update: new max, alpha, rescale O and l */ }
// else: keep the stale max, skip the exp2 and every multiply of O
```

After the first few tiles the max rarely jumps by 2^8, so most tiles skip the rescale.

**Measured:** 15.0 / 14.4 / 14.9 / 14.4 vs the baseline's 15.1 / 14.5 / 14.8 / 14.3, which is noise. The
fp32 path is bound by the tensor cores, so saving multiplies doesn't shorten anything. It stays
in `opt` because it can't hurt, and on a GPU with much faster tensor cores the softmax work
matters more (that's why FA4 introduced it).

Code: the `rescale` lambda in the softmax section of `tile()`.

### 13.4 EXP16: two exponentials per SFU instruction (small gain with ACC16)

Exponentials run on the SFU (special function unit), which does only 16 per clock per SM. PTX has
`ex2.approx.f16x2`, which computes two exponentials of a packed fp16 pair in one instruction. Its
output is exactly the register format the P·V mma wants as its A operand:

```cpp
const float x0 = fmaf(s0, scale_log2, -max);     // exponent arguments in fp32
const float x1 = fmaf(s1, scale_log2, -max);
p_h = ex2_f16x2(pack_half2(x0, x1));             // 1 SFU instruction, P already packed
const float2 p = unpack_half2(p_h);              // for the row sum
row_sum += p.x + p.y;
```

- **Precision.** x is rounded to fp16 before the exponential. P is rounded to fp16 anyway for the
  mma, so this costs about the same precision (`Exp2.F16x2MatchesExp2WithinFp16Precision` checks it).
- **Consistent normalization.** The row sum adds the same rounded fp16 values the mma multiplies,
  so the normalization matches the numerator exactly.
- **Measured.** With fp32 accumulation it is slightly *slower* (14.7 / 14.2 / 14.9 / 14.2): the
  pack/unpack conversions cost more than the halved SFU work saves. With ACC16, where the matmul
  part is twice as fast and the softmax a larger share, it gains up to 1.7%, so it's in `fp16acc`.

Code: the `O::EXP16` branch of the exponential loop in `tile()`; `ex2_f16x2` in `common.cuh`.

### 13.5 EMU: FA4's software exp2 (slower here, kept for the 5090)

FA4 found that on Blackwell (B200) the tensor cores are so fast that the SFU's exponentials become
the bottleneck, and moved some exponentials to the FMA units, which are 8× more plentiful
(128 vs 16 per clock per SM on the 3050). It splits 2^x into an integer and a fractional part:

```
2^x = 2^floor(x) · 2^f,   f = x − floor(x) ∈ [0, 1)
2^f ≈ 1 + f·(0.69607 + f·(0.22449 + f·0.07944))     3 FMAs (Horner), max rel. error ~1e-4
2^floor(x): add floor(x) to the exponent bits:  bits(result) = bits(2^f) + (floor(x) << 23)
```

The `<< 23` works because a float's exponent field starts at bit 23, so adding to it multiplies
by a power of two without any arithmetic. x is clamped to ≥ −127 first, which also maps masked
scores (−∞) to 0. `Exp2.PolynomialRelativeErrorBelowFp16Resolution` checks the error is below
2e-4, invisible after rounding P to fp16. The kernel applies it to every other 8-key block (half
the exponentials), sharing the work between FMA units and SFU.

**Measured: slower** (−1 to −5%, both paths). One emulated exponential is about 9 FMA/ALU
instructions against 1 SFU instruction worth about 8 FMA slots. That only wins when the SFU is
saturated and the FMA pipe has idle slots. On the 3050 the SFU isn't the bottleneck, and the
FMA pipe is busy with the rest of the softmax. Worth re-measuring on the 5090.

Code: `exp2_poly3` in `common.cuh`; the `O::EMU` branch in `tile()`.

### 13.6 PEEL: masked tiles in their own loop (slower here)

Only a few tiles need masking: the last one when N isn't a multiple of BC, and, for causal, the
ones that cross the diagonal. PEEL runs the tiles in two loops, and the first one has no mask code at all:

```cpp
int first_masked = (n % BC != 0) ? n_tiles - 1 : n_tiles;
if (CAUSAL) first_masked = min(first_masked, (q0 + 1) / BC);
for (j = 0;            j < first_masked; ++j) tile(j, std::false_type{});  // no mask code
for (j = first_masked; j < n_tiles;      ++j) tile(j, std::true_type{});   // masked
```

`(q0 + 1) / BC` is the first tile whose last key comes after the block's first query row q0:
tile j crosses the diagonal when j·BC + BC − 1 > q0. For example, q0 = 64, BC = 64 gives tile 1.
The per-tile body is a lambda that takes a `std::true_type` / `std::false_type` tag, so
`if constexpr` can delete the mask code from one copy.

**Measured: slower** (−0.5 to −3%). The branch it removes was cheap: one warp-uniform compare
per tile, and all 32 threads always take the same side. The fully unrolled tile body is large,
and PEEL doubles it. The likely cost is instruction-cache pressure (not verified with a profiler).

Code: the `if constexpr (O::PEEL)` main loop after `tile()`.

## What is *not* here (and why)

- **FlashAttention-3** techniques (TMA, WGMMA, warp specialisation, ping-pong scheduling)
  need Hopper (sm_90) hardware; RTX 3050/4090/5090 don't have it.
- **FP8** (Ada sm_89+ supports fp8 mma): different accuracy contract and can't run on the 3050; see [fa3_fa4_techniques.md](fa3_fa4_techniques.md).
- **Split-KV (Flash-Decoding)** for tiny batch × long KV: this kernel targets prefill/training
  shapes where B·H·N/BR already fills the GPU.
- Backward pass.
