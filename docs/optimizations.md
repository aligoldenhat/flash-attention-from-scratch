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

## What is *not* here (and why)

- **FlashAttention-3** techniques (TMA, WGMMA, warp specialisation, ping-pong scheduling)
  need Hopper (sm_90) hardware; RTX 3050/4090/5090 don't have it.
- **FP8** (Ada sm_89+ supports fp8 mma): different accuracy contract; a possible extension.
- **Split-KV (Flash-Decoding)** for tiny batch × long KV: this kernel targets prefill/training
  shapes where B·H·N/BR already fills the GPU.
- Backward pass.
