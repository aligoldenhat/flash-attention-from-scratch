# Benchmarking

## What we compare against

| Name in the results | What it is |
|---|---|
| **fa fp16-acc (ours)** | this repo's kernel, `fa.forward(..., fp16_accum=True)`: P·V accumulated in fp16 per tile (opt-in) |
| **fa opt (ours)** | this repo's kernel, `fa.forward` (default): exact fp32 accumulation |
| fa baseline (ours) | the first tuned version of the kernel, kept as the reference point |
| **sdpa-flash (FA2)** | `F.scaled_dot_product_attention` forced to the flash backend: FlashAttention-2 code vendored inside PyTorch. The production bar on these GPUs. |
| sdpa-cudnn | the same API on NVIDIA's cuDNN fused-attention backend |
| sdpa-efficient | the memory-efficient backend (xFormers / CUTLASS based) |
| flashinfer | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) batch prefill, the attention library used by vLLM and SGLang, on its native token-major layout `[B·N, H, d]`. Its kernels are JIT-compiled with the local nvcc on first use. |
| torch naive | `softmax(Q Kᵀ / √d) V` written with plain matmuls in fp16, materialising N × N |

Not compared, and why:

- **FlashAttention-3** needs Hopper (H100, sm_90): TMA (Tensor Memory Accelerator) and WGMMA
  (warpgroup MMA) instructions do not exist on RTX 3050 / 4090 / 5090. **FlashAttention-4**
  targets datacenter Blackwell (B200, sm_100) and does not run on the RTX 5090 (sm_120) either.
  On consumer GPUs, FA2 is the state of the art.
- **The official `flash-attn` package** (Dao-AILab): no prebuilt wheel for this torch/CUDA combo,
  and a source build takes hours. SDPA's flash backend is the same FA2 algorithm and kernel
  family; the package can be added on the rented GPUs.
- **xFormers** (0.0.35): its wheel is built for torch 2.10 + CUDA 12.8, so with torch 2.14 its own
  CUDA kernels don't load. The attention paths it still offers (`cutlassF-pt`, `fa2F-pt`) call
  PyTorch's built-in kernels, the same ones as `sdpa-efficient` and `sdpa-flash`, so it would only
  add duplicate lines. Needs a source build to test its real kernels.
- **FlashInfer's fp16 accumulation** (`use_fp16_qk_reduction=True`): its JIT build rejects it unless
  compiled with `-DFP16_QK_REDUCTION_SUPPORTED` and Boost.Math, so only the default (fp32) mode
  is benchmarked.
- **SageAttention** (quantized INT8/fp8 attention): v1 is Triton-based (Triton baselines are on
  hold), v2 needs a source build. A different accuracy contract; a candidate for later.

## Method (`bench/benchmark.py`)

- **Shapes** follow the FlashAttention-2 paper: B·N = 16384 tokens and H·d = 2048, i.e.
  `B = 16384 / N`, `H = 2048 / d`. N ∈ {512 … 16384}, d ∈ {64, 128}, causal on/off.
- **Timing**: CUDA events around each call, 10 warmup calls, then at least 20 and up to 200
  timed calls (about 1.5 s per point); the **median** is reported.
- **FLOPs** = 4·B·H·N²·d (two matmuls of 2·N²·d each), halved for causal.
- **Correctness cross-check**: before timing, every implementation's output is compared with
  ours on the same inputs; a mismatch above 2e-2 is reported instead of a time.
- **Native layouts**: each implementation has a setup step outside the timed region (layout
  conversion, FlashInfer's `plan()`), so each is timed on its preferred input layout.
- Out-of-memory and unsupported backends are recorded as such, not as crashes.

```
make bench                  # sweep -> bench/results/<gpu>.csv, plot -> docs/img/<gpu>.png, table
python bench/benchmark.py --n 4096 --d 128 --causal 1 --impl "fa (ours)" "sdpa-flash (FA2)"
```

## Results: RTX 3050 Laptop (sm_86)

![benchmark](img/NVIDIA_GeForce_RTX_3050_Laptop_GPU.png)

TFLOPS, higher is better. torch 2.14.1+cu130, CUDA 13.1, FlashInfer 0.7.0, driver clocks
uncontrolled (laptop). All columns come from the same run.

| d | causal | N | fa fp16-acc (ours) | fa opt (ours) | fa baseline (ours) | sdpa-flash (FA2) | sdpa-cudnn | flashinfer | sdpa-efficient | torch naive | opt / FA2 | fp16-acc / FA2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0 | 512 | **18.1** | **14.2** | 14.2 | 13.4 | 12.9 | 13.3 | 10.5 | 3.5 | 1.06x | 1.35x |
| 64 | 0 | 1024 | **18.7** | **14.4** | 14.5 | 14.0 | 13.4 | 13.5 | 10.8 | 3.6 | 1.03x | 1.34x |
| 64 | 0 | 2048 | **19.0** | **14.7** | 14.6 | 14.1 | 13.7 | 13.8 | 10.9 | OOM | 1.04x | 1.35x |
| 64 | 0 | 4096 | **19.2** | **14.7** | 14.7 | 14.1 | 13.8 | 13.7 | 11.0 | OOM | 1.04x | 1.36x |
| 64 | 0 | 8192 | **19.1** | **14.6** | 14.6 | 14.1 | 13.9 | 13.8 | 10.9 | OOM | 1.04x | 1.36x |
| 64 | 0 | 16384 | **19.2** | **14.7** | 14.6 | 13.7 | 13.8 | 13.6 | 10.6 | OOM | 1.07x | 1.40x |
| 64 | 1 | 512 | **13.0** | **10.4** | 10.5 | 9.8 | 7.9 | 9.8 | 8.7 | 1.3 | 1.07x | 1.33x |
| 64 | 1 | 1024 | **15.3** | **12.1** | 12.2 | 11.5 | 9.9 | 11.4 | 9.6 | 1.3 | 1.06x | 1.33x |
| 64 | 1 | 2048 | **16.8** | **13.2** | 13.2 | 12.5 | 11.4 | 12.2 | 10.2 | OOM | 1.06x | 1.34x |
| 64 | 1 | 4096 | **17.4** | **13.7** | 13.7 | 12.9 | 12.1 | 12.9 | 10.3 | OOM | 1.07x | 1.35x |
| 64 | 1 | 8192 | **17.7** | **13.9** | 13.8 | 13.1 | 12.5 | 12.8 | 10.0 | OOM | 1.06x | 1.35x |
| 64 | 1 | 16384 | **17.8** | **13.7** | 13.7 | 13.0 | 12.3 | 12.7 | 9.4 | OOM | 1.06x | 1.38x |
| 128 | 0 | 512 | **15.6** | **13.1** | 12.9 | 12.9 | 12.1 | 12.9 | 8.2 | 5.6 | 1.02x | 1.21x |
| 128 | 0 | 1024 | **16.1** | **13.4** | 13.2 | 13.1 | 12.6 | 12.9 | 8.4 | 5.7 | 1.02x | 1.23x |
| 128 | 0 | 2048 | **16.6** | **13.7** | 13.1 | 13.0 | 12.7 | 13.3 | 8.6 | 5.6 | 1.05x | 1.28x |
| 128 | 0 | 4096 | **16.6** | **13.6** | 13.3 | 13.2 | 13.0 | 13.3 | 8.6 | OOM | 1.03x | 1.26x |
| 128 | 0 | 8192 | **16.4** | **13.5** | 13.3 | 13.1 | 12.6 | 13.0 | 8.4 | OOM | 1.03x | 1.25x |
| 128 | 0 | 16384 | **15.9** | **13.2** | 13.0 | 12.9 | 12.7 | 13.0 | 8.5 | OOM | 1.03x | 1.24x |
| 128 | 1 | 512 | **11.9** | **9.9** | 9.9 | 9.8 | 8.4 | 9.2 | 6.2 | 2.2 | 1.01x | 1.22x |
| 128 | 1 | 1024 | **13.9** | **11.3** | 11.2 | 11.1 | 10.2 | 10.7 | 7.1 | 2.1 | 1.02x | 1.25x |
| 128 | 1 | 2048 | **14.6** | **12.2** | 11.9 | 11.2 | 11.3 | 11.6 | 7.7 | 2.0 | 1.08x | 1.30x |
| 128 | 1 | 4096 | **15.1** | **12.6** | 12.4 | 11.7 | 11.7 | 11.9 | 7.8 | OOM | 1.08x | 1.30x |
| 128 | 1 | 8192 | **15.8** | **12.7** | 12.1 | 11.0 | 11.6 | 12.0 | 7.8 | OOM | 1.15x | 1.43x |
| 128 | 1 | 16384 | **15.9** | **12.7** | 12.5 | 11.0 | 11.2 | 12.0 | 7.8 | OOM | 1.16x | 1.45x |

### How to read these numbers honestly

- **The fp32-accumulate ceiling is about 14.5 TFLOPS.** A plain cuBLAS fp16 GEMM (4096³, fp32
  accumulate) measures 14.5 TFLOPS on this laptop. `opt` and `baseline` reach 14.5–14.8:
  attention runs at the speed of a pure matmul, and every fp32-accumulate implementation is
  squeezed against the same hardware limit. That's why opt / FA2 is only 1.01–1.16x, and why
  opt and baseline overlap.
- **FlashInfer** lands next to FA2: about equal at d = 64, slightly ahead of FA2 at d = 128 causal,
  behind ours everywhere.
- **fp16-acc breaks that ceiling: 1.21–1.45x FA2** (up to 19.5 TFLOPS in a cool run). GeForce GPUs run
  fp16-accumulate tensor-core math at twice the fp32-accumulate rate; the P·V half of the work
  uses it. See [fa3_fa4_techniques.md](fa3_fa4_techniques.md) for how it stays accurate.
- Biggest exact-path wins are **causal** (6–13% over FA2): diagonal-only masking, skipping
  fully masked tiles, reversed block order for load balance, and causal-specific tile shapes.
- **Laptop noise**: clocks move with temperature and power, about ±0.3 TFLOPS run to run at
  large N and more at N = 512. Differences under ~2% are within noise. Whole runs shift too: in
  this run every implementation measured 5–10% lower at d = 128 than in the previous one (FA2
  included), so compare columns within one run, not numbers across runs.
- **naive OOM**: the fp16 N × N score matrix needs 2–8 GB at these shapes, more than the 4 GB card.
- The rented RTX 4090 / 5090 have far more tensor throughput per byte of shared-memory and
  DRAM bandwidth, so the gap between implementations there will be larger and more telling.
  **Re-run `make tune` (and `bench/ablate.sh`) first on each new GPU.**

## Tile tuning (`bench/tune.sh`, `make tune`)

`Config<D, WARPS, MT, BC, Q_IN_REGS>`: warps per block, 16-row m-tiles per warp, keys per
tile, Q kept in registers or re-read from shared memory. Each cell is the best of 3 medians on
the standalone driver; each run also checks correctness.

| Slot | Chosen | TFLOPS N = 512 / 1k / 4k / 16k | Previous `4,1,64,true` |
|---|---|---|---|
| d = 64 | `4, 2, 64, true` | 14.3 / 14.6 / 14.7 / 14.8 | 13.7 / 14.1 / 14.4 / 14.2 |
| d = 64 causal | `4, 2, 64, true` | 11.0 / 12.7 / 14.3 / 14.8 | 9.8 / 12.8 / 13.9 / 14.1 |
| d = 128 | `4, 2, 32, false` | 14.0 / 14.4 / 14.6 / 14.6 | 13.7 / 14.1 / 14.3 / 14.2 |
| d = 128 causal | `4, 1, 64, true` | 11.6 / 12.9 / 14.0 / 14.2 | (same) |

`fp16acc` tiles (`Config<..., STAGES>` adds the pipeline depth), re-tuned because fp16
accumulation moves the bottleneck:

| Slot | Chosen | TFLOPS N = 512 / 1k / 4k / 16k |
|---|---|---|
| d = 64 | `4, 2, 32, true, 2` | 16.0 / 19.0 / 19.6 / 19.5 |
| d = 64 causal | `4, 2, 32, true, 2` | 12.1 / 16.0 / 18.5 / 19.1 |
| d = 128 | `4, 1, 64, true, 1` | 14.8 / 18.1 / 18.5 / 18.4 |
| d = 128 causal | `4, 1, 64, true, 1` | 12.6 / 16.2 / 18.1 / 18.4 |

What the sweep showed:

- **MT = 2 (32 rows per warp) wins** in three of four slots: each ldmatrix-loaded K/V fragment
  feeds two mma instructions, halving shared-memory traffic per FLOP.
- **d = 128 with MT = 2 must re-read Q from shared memory**: keeping Q in registers too
  (`4,2,64,true`) collapses to ~9 TFLOPS. The likely cause is register pressure: the O accumulator
  alone is 64 fp32 registers per m-tile at d = 128, times 2 m-tiles, plus 64 for Q (spills not yet
  confirmed with `FA_PTXAS_VERBOSE=1`).
- **d = 128 causal prefers MT = 1** at short N: a 128-row block wastes more work on the half-masked
  diagonal tile than a 64-row block.
- **8 warps per block is always slower** here (cause not profiled yet).

## Reproducing on a rented GPU

```
git clone ... && cd flash-attention-from-scratch
uv venv --python 3.12 .venv && uv pip install --python .venv torch numpy pytest pandas matplotlib ninja setuptools flashinfer-python \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
make build ARCH=8.9          # RTX 4090 (12.0 for RTX 5090)
make test                    # correctness first
make tune                    # pick tiles for this GPU, update the `using <Base|Opt|Fast>D...` lines, rebuild
make bench                   # -> bench/results/NVIDIA_GeForce_RTX_4090.csv + plot
```
