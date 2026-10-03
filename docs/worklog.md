# Work log

What was done, in order, with the decisions and the results that drove them.

## 1. Toolchain setup (2026-10-02)

- Checked the machine: RTX 3050 Laptop (sm_86, 16 SMs, 4 GB), CUDA 13.1, GCC 15.2, CMake 4.2,
  clangd/clang-tidy 22, compute-sanitizer / ncu / nsys installed. ncu counters are root-only.
- Scaffolded the C++/CUDA project template: CMake presets (debug / release / asan), strict
  warnings as errors, ASan/UBSan, `.clangd`, `.clang-format`, `.clang-tidy`, `scripts/lint.sh`,
  VS Code configs. Adapted it from `src/` to the repo's `csrc/` layout.
- Verified it: all three presets build and run, the four compute-sanitizer tools are clean,
  `cuobjdump` shows `sm_86`, lint is clean, `clangd-22 --check` reports 0 errors.
  Negative checks: `const int bad = v.size();` fails the debug build with `-Werror=conversion`;
  removing a bounds check makes memcheck report `Invalid __global__ read`.

## 2. Scope change

The original plan was a v1…v6 progression, with the user writing the kernels. It changed at the
user's request to **one fully optimized kernel**, written by Claude, compared against production
implementations. `CLAUDE.md` was updated to match.

Comparison targets: FlashAttention-3 only runs on Hopper (sm_90) and FA4 on datacenter Blackwell,
so neither runs on RTX 3050/4090/5090. The production bar is **FlashAttention-2**, used through
PyTorch SDPA's flash backend, plus the cuDNN and memory-efficient backends and naive PyTorch.
The official `flash-attn` package was skipped locally (no wheel, hours to build).

Python environment: `uv venv` in `.venv`, PyTorch 2.14.1+cu130.

## 3. The kernel (`csrc/kernels/flash_fwd.cu`)

FlashAttention-2 forward pass in raw CUDA C++ with inline PTX. Design and every optimization
are explained in [optimizations.md](optimizations.md). In short:

- mma.sync.m16n8k16 tensor cores fed by ldmatrix (`.trans` for V); P stays in registers
- each warp owns whole query rows (no inter-warp softmax reduction)
- cp.async loads overlapped with compute (FA2 pipeline), zero-fill for rows ≥ N
- XOR-swizzled shared memory: no bank conflicts, no padding
- exp2 softmax with the scale folded into one FMA, lazy row-sum reduction
- causal: skips masked tiles, masks only diagonal tiles, reversed block order
- output staged through shared memory, 16-byte coalesced stores

First version (4 warps × 16 rows, 64-key tiles) was correct on the first run (max error ~1e-3 on
all shapes, including odd N and causal) and already on par with FA2 (14.5 vs 14.6 TFLOPS at d = 128).

A cuBLAS fp16 GEMM reaches 14.5 TFLOPS on this GPU, so the kernel was already at the
hardware ceiling for large N.

PyTorch 2.14 headers require C++20, so `setup.py` uses `-std=c++20`.

## 4. Infrastructure

- `csrc/bindings.cpp` (`fa.forward(q, k, v, causal, softmax_scale)`), `setup.py`, `pyproject.toml`.
- `csrc/dev/main.cu`: standalone driver with a CPU reference, used for sanitizers, ncu, tuning.
  The strict debug build caught real sign-conversion issues in its indexing; fixed with `size_t`.
- `tests/test_flash_fwd.py` (57 pytest cases), `bench/benchmark.py`, `bench/plot.py`,
  `profile/ncu.sh`, `profile/nsys.sh`, `Makefile`.

## 5. Optimization: multiple m-tiles per warp + tile tuning

The first full benchmark: causal already 2–11% faster than FA2, non-causal 1–3% slower.
The missing piece was FA2's larger per-warp tile: with **MT = 2 m-tiles per warp** each
ldmatrix'd K/V fragment feeds two mma instructions, halving shared-memory traffic. Added `MT` and a
`Q_IN_REGS` switch (re-read Q from shared memory to save registers) as template parameters, plus
separate configs for causal and non-causal, and wrote `bench/tune.sh` to sweep them.

Result (see [benchmarking.md](benchmarking.md#tile-tuning-benchtunesh-make-tune)): MT = 2 wins
in 3 of 4 slots. d = 128 with MT = 2 needs Q in shared memory (with Q in registers it drops to
9 TFLOPS, most likely register pressure). After tuning, **ours is fastest at all 24 benchmark shapes**: 1.01–1.06x FA2
non-causal, 1.02–1.11x causal.

## 6. C++ unit tests (GoogleTest)

`tests/cpp/`: 19 tests. The PTX building blocks (mma fragment layout, ldmatrix and ldmatrix.trans
register contents, swizzle permutation and bank-conflict freedom, cp.async zero-fill), C++
argument validation, and 9 small end-to-end shapes against a CPU reference. `swizzle()` moved to
`common.cuh` as `__host__ __device__` so it can be tested on the CPU. CMake now builds the kernel
as a library (`fa_kernel`) shared by the driver and the tests; GoogleTest is fetched at configure
time.

Mutation check: removing the swizzle XOR is caught **only** by the swizzle unit test (the output
stays correct, only slower); a wrong ldmatrix lane mapping is caught by the ldmatrix test.
All tests pass under memcheck, racecheck, synccheck and initcheck. Details: [testing.md](testing.md).

## 7. Current status

| Check | Result |
|---|---|
| pytest | 57 passed |
| GoogleTest (ctest) | 19 passed |
| `make sanitize` | 0 errors / 0 hazards (driver, GoogleTest, pytest subsets) |
| Register spills | 0 in all four configs |
| Benchmark vs FA2 (RTX 3050) | faster at all 24 shapes |

## 8. Editor and lint cleanup

Errors like `use of undeclared identifier 'CUDA_CHECK'` showed up in every file. Cause: CMake puts
nvcc's include paths in a response file (`--options-file`), and `.clangd` strips that flag, so the
`-I` paths were lost. Fixed with `CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF`. Also: `.clangd`
strips `--use_fast_math`; `bindings.cpp` gets its own flags with the PyTorch headers; VS Code uses
`clangd-22`; `lint.sh` now also checks `tests/cpp/` and `bindings.cpp`; the kernel uses
`std::numbers::log2e_v` and has two documented NOLINTs; everything is clang-formatted.
Result: 0 clangd diagnostics and 0 clang-tidy warnings in all files. See [setup.md](setup.md#editor-setup-clangd).

## Open items

- **ncu reports**: need sudo; run `make profile` in your own terminal.
- Benchmarks on RTX 4090 / 5090 (re-tune first), optionally with the official `flash-attn` package.
