# FlashAttention-3 / FlashAttention-4 techniques, and what applies to this kernel

FlashAttention-2 (FA2) is the algorithm this kernel implements. FA3 (2024) was written for
Hopper (H100) and FA4 (2025) for datacenter Blackwell (B200). Both get most of their speed from
hardware features that consumer GPUs don't have, but several of their ideas are pure software
and carry over. This page lists each technique, whether it can run on the GPUs this project
targets, and what happened when it was tried. It is written from the papers and the hardware
documentation as I know them; anything marked *(unverified)* was not checked on real hardware here.

Target GPUs: **RTX 3050 Laptop** (Ampere, sm_86, development), **RTX 4090** (Ada, sm_89),
**RTX 5090** (consumer Blackwell, sm_120).

## Techniques

| Technique | From | 3050 | 4090 | 5090 | Status in this repo |
|---|---|---|---|---|---|
| **WGMMA**: asynchronous warpgroup (4-warp) matrix multiply | FA3 (Hopper) | ✗ | ✗ | ✗ | Not possible: Hopper-only instruction. Consumer GPUs only have `mma.sync`. |
| **TMA** (Tensor Memory Accelerator): hardware copies whole tiles global → shared | FA3 | ✗ | ✗ | ✓ *(unverified)* | Not used: `cp.async` is the Ampere/Ada equivalent; TMA can't be tested on the 3050. |
| **Warp specialization** + `setmaxnreg`: producer warps only load, consumer warps only compute, registers moved between them | FA3 | weak | weak | weak | Not used: without TMA the producer warps cost registers for little gain, and `setmaxnreg` is Hopper-only. Every warp issues its own `cp.async` instead. |
| **Ping-pong scheduling** between warpgroups: one does softmax while the other does matmul | FA3 | implicit | implicit | implicit | Happens automatically: with several warps per SM, the warp scheduler interleaves one warp's softmax with another's `mma.sync`. |
| **Intra-warpgroup overlap**: softmax of tile j overlaps the matmul of tile j+1, needs deeper buffering | FA3 | ✓ | ✓ | ✓ | Tried as **STAGES=2** (double-buffered K/V): **+5% at d=64** with fp16acc (used there); −12–17% at d=128 (shared memory halves occupancy). |
| **FP8** attention with block quantization and Hadamard "incoherent processing" | FA3 | ✗ | ✓ | ✓ | Not done: the 3050 has no fp8 tensor cores, so it can't be developed or tested here. Candidate for the 4090/5090. |
| **Software exp2**: compute some exponentials as a polynomial on the FMA units, so the SFU isn't the bottleneck | FA4 | ✓ | ✓ | ✓ | Implemented as **EMU**: **slower here** (−1 to −5%), not used. The SFU isn't the bottleneck on these GPUs. |
| **Conditional (lazy) rescaling**: only rescale O when the row max grows by more than 2^8 | FA4 | ✓ | ✓ | ✓ | Implemented as **LAZY**: neutral on the 3050 (within noise); kept in `opt`. |
| **tcgen05 + TMEM** (tensor memory), 2-CTA matmul | FA4 (B200) | ✗ | ✗ | ✗ | Not possible: datacenter Blackwell (sm_100) only. The 5090 (sm_120) doesn't have them. |
| **fp16 accumulation** of P·V (2× tensor throughput on GeForce cards) | consumer-GPU kernels | ✓ | ✓ | ✓ | Implemented as **ACC16**, the `fp16acc` variant, opt-in: **+25–30%, the biggest win**. |
| **`ex2.approx.f16x2`**: two exponentials per SFU instruction | common trick | ✓ | ✓ | ✓ | Implemented as **EXP16**: small gain with fp16acc (used there), slightly slower with fp32 accumulation. |
| **Peeling masked tiles** out of the main loop | FA2/FA3 | ✓ | ✓ | ✓ | Implemented as **PEEL**: slower (−1 to −3%), not used. |
| Runtime tile selection by sequence length | FA2 heuristics | ✓ | ✓ | ✓ | Not done yet; tiles are fixed per (head dim, causal). |
| Persistent kernel + tile scheduler (causal load balance, L2 reuse) | FA3 | ✓ | ✓ | ✓ | Not done yet. |
| Split-KV (Flash-Decoding) for small batch × long context | FlashDecoding | ✓ | ✓ | ✓ | Not done: targets inference decoding, not these benchmark shapes. |

## Ablation: each implemented technique on its own

Measured with `bench/ablate.sh` on the RTX 3050 Laptop: TFLOPS, best of 3 medians, FA2-paper
shapes (B·N = 16k, H·d = 2048). Noise is about ±0.3 at N = 4096 and several TFLOPS at N = 1024
(laptop clocks), so N = 4096 is the column to trust.

Columns: head dim / causal at N = 4096 (N = 1024 in the raw logs, too noisy to rank by).

**fp32-accumulate path** (`opt` variant), each flag added alone to the baseline:

| Flags | d64 | d64 causal | d128 | d128 causal | Verdict |
|---|---|---|---|---|---|
| baseline | 15.1 | 14.5 | 14.8 | 14.3 | |
| + PEEL | 15.0 | 14.4 | 14.6 | 14.2 | −0.5 to −1.5%: no gain |
| + STAGES=2 | 15.1 | 14.5 | **13.1** | **12.3** | d=128: 64 KB smem → 1 block/SM instead of 2 |
| + LAZY | 15.0 | 14.4 | 14.9 | 14.4 | noise; kept in `opt` (harmless, saves work on faster GPUs) |
| + EXP16 | 14.7 | 14.2 | 14.9 | 14.2 | slightly worse (extra fp32 ↔ fp16 conversions) |
| + EMU | 14.4 | 13.8 | 14.5 | 13.9 | −2 to −5%: the SFU is not the bottleneck |

None of these can help much here: the fp32-accumulate path is already at the GPU's tensor-core
ceiling (a cuBLAS GEMM also reaches 14.5), so removing non-matmul work doesn't shorten anything.

**fp16-accumulate path** (`fp16acc` variant), each flag added to ACC16 alone:

| Flags | d64 | d64 causal | d128 | d128 causal | Verdict |
|---|---|---|---|---|---|
| ACC16 | 18.9 | 18.1 | 18.5 | 17.9 | **+25% over the baseline** |
| + PEEL | 18.6 | 17.7 | 18.5 | 17.4 | worse: two copies of the unrolled loop body (likely instruction cache, not verified) |
| + STAGES=2 | **19.9** | **18.4** | 15.4 | 14.7 | d=64 +5% / +2%; d=128 loses occupancy as above |
| + EXP16 | 19.2 | 18.3 | 18.2 | 18.2 | small gain: kept |
| + EMU | 18.3 | 17.4 | 18.4 | 18.0 | worse |
| + PEEL + STAGES=2 + EXP16 | 19.1 | 17.9 | 15.7 | 14.4 | PEEL cancels the STAGES gain |

Conclusions:

- **fp16 accumulation is the one big win** (+25–30%), the only technique that raises the
  ceiling instead of trimming work under it.
- **STAGES=2 helps only when shared memory allows it**, so it moved from a per-variant flag
  into the tile config (`Config<..., STAGES>`), and `bench/tune.sh` picks it per head dim.
- **FA4's software exp2 doesn't pay off here.** FA4 needs it because Blackwell's tensor cores
  are so fast that the exponentials become the bottleneck. On the 3050 they aren't, and the
  polynomial takes FMA-pipe slots the softmax needs. Worth re-measuring on the 5090.
- **Peeling doesn't help.** The masked-tile branch is warp-uniform and cheap; duplicating the
  body costs more than the branch.

Chosen flags: `opt` = LAZY; `fp16acc` = ACC16 + EXP16, with STAGES=2 at d = 64.

## Number formats, and which GPU can do what

A floating-point number is sign × mantissa × 2^exponent. **Exponent bits set the range** (how
big or small a value can be); **mantissa bits set the precision** (how many significant digits).

| Format | Bits (sign / exponent / mantissa) | Largest value | Relative precision | Typical use |
|---|---|---|---|---|
| **fp32** | 1 / 8 / 23 | 3.4e38 | ~6e-8 | accumulators, softmax statistics |
| **tf32** | 1 / 8 / 10 (stored in 32 bits) | 3.4e38 | ~5e-4 | fp32 matmuls on tensor cores (Ampere+) |
| **fp16** (half) | 1 / 5 / 10 | 65504 | ~5e-4 | inputs here: Q, K, V, P, O |
| **bf16** (bfloat16) | 1 / 8 / 7 | 3.4e38 | ~4e-3 | training: fp32's range, less precision |
| **fp8 e4m3** | 1 / 4 / 3 | 448 | ~6e-2 | inference matmuls, with scale factors |
| **fp8 e5m2** | 1 / 5 / 2 | 57344 | ~1e-1 | gradients (needs range more than precision) |
| **fp4 e2m1** (NVFP4 / MXFP4) | 1 / 2 / 1 | 6 | coarse | inference, always with a scale per small block |

Two things to know about each format on a GPU:

1. **Input format**: what the tensor cores multiply (A and B).
2. **Accumulator format**: what the products are summed into (C and D). Summing many numbers
   loses precision fast in a small format, so accumulators are usually fp32.

### Tensor-core support by architecture

| Input format | Ampere (RTX 3050, sm_86) | Ada (RTX 4090, sm_89) | Consumer Blackwell (RTX 5090, sm_120) | Hopper (H100, sm_90) |
|---|---|---|---|---|
| fp16, bf16 | ✓ | ✓ | ✓ | ✓ |
| tf32 | ✓ | ✓ | ✓ | ✓ |
| int8 | ✓ | ✓ | ✓ | ✓ |
| **fp8** (e4m3, e5m2) | **✗** | ✓ | ✓ | ✓ |
| **fp4** (block-scaled) | ✗ | ✗ | ✓ | ✗ |
| Instruction | `mma.sync` | `mma.sync` | `mma.sync` (+ block-scaled variants) | `mma.sync` + **WGMMA** |

So fp8 needs a 4090 or newer, and fp4 a 5090. The 3050 tops out at fp16/bf16.

### The GeForce accumulator rule (why `fp16acc` is faster)

On **GeForce** cards (RTX 30/40/50), an fp16 × fp16 matmul runs at **full speed with an fp16
accumulator** but at **half speed with an fp32 accumulator**. Datacenter cards (A100, H100) run
both at full speed. NVIDIA's whitepapers list both numbers for every GeForce card.

That's why every implementation on the 3050 hit the same ~14.5 TFLOPS wall: FA2, cuDNN, cuBLAS
and this kernel all accumulate in fp32, so they all run at the half-speed rate. The `fp16acc`
variant goes past it (18+ TFLOPS measured) by doing the P·V matmul with fp16 accumulation:

- **Only P·V, not Q·Kᵀ.** Scores can be large (hundreds) and need fp32 before the exponential.
  P is between 0 and 1, so P·V sums are bounded.
- **Only within one tile.** The fp16 partial sum covers 64 keys (4 tensor-core steps), and is
  then added into the fp32 accumulator. A sum over the whole sequence in fp16 could overflow
  65504 or drift; a 64-key sum can't (unless |V| > 1000).
- **Opt-in** (`fa.forward(..., fp16_accum=True)`), because it changes the accuracy contract
  slightly. Measured error stays far below the 1e-2 test tolerance.

### Where fp8 would come in

On a 4090/5090, fp8 inputs double tensor throughput again over fp16. But fp8 e4m3 has only
3 mantissa bits, so Q, K and V need scale factors (per tensor, per block or per row) to use the
small range well. FA3's "incoherent processing" multiplies Q and K by a random Hadamard matrix
first, which spreads out outliers so a single scale fits better. That's a different accuracy
contract and a separate project step, and it can't run on the 3050.
