# Setup, build system and repository layout

## Toolchain (development machine)

| Tool | Version | Used for |
|---|---|---|
| GPU | RTX 3050 Laptop, sm_86 (Ampere), 16 SMs, 4 GB | development, first benchmarks |
| CUDA toolkit (nvcc) | 13.1 | compiling kernels |
| GCC | 15.2 | host compiler behind nvcc |
| CMake | 4.2 | standalone driver + C++ tests |
| Python / uv | 3.12, `.venv` | bindings, pytest, benchmark |
| PyTorch | 2.14.1+cu130 | bindings, reference, baselines |
| clangd / clang-tidy | 22 (`clangd-22`, `clang-tidy-22`) | editor diagnostics, lint |
| compute-sanitizer, ncu, nsys | from the CUDA toolkit | memory checking, profiling |

Later benchmarks run on rented RTX 4090 (sm_89) and RTX 5090 (sm_120); the build takes the
architecture as a parameter everywhere (`make build ARCH=8.9`, `make dev ARCH=12.0`).

Python analogies for the C++ tools: clang-format ≈ `ruff format`, clang-tidy ≈ `ruff check`,
clangd ≈ the editor side of `ty`, CMake presets ≈ named profiles in `pyproject.toml`,
GoogleTest ≈ pytest, ctest ≈ the `pytest` command.

## Two builds of the same kernel

The kernel source `csrc/kernels/flash_fwd.cu` is compiled by two build systems:

```
                         csrc/kernels/flash_fwd.cu  (+ common.cuh, flash_fwd.h; no PyTorch)
                          /                                   \
     setup.py (torch.utils.cpp_extension)                 CMake (CMakePresets.json)
     + csrc/bindings.cpp                                    library fa_kernel
     -> fa/_C.so  (import fa)                               -> build/<preset>/fa_dev    (driver)
     used by: pytest, benchmark, nsys                       -> build/<preset>/fa_tests  (GoogleTest)
                                                            used by: sanitizers, ncu, tune.sh
```

Why the kernel knows nothing about PyTorch: `flash_fwd.h` takes raw pointers and sizes. Only
`bindings.cpp` includes torch. That keeps the kernel compiling in seconds under the strict
CMake warning set, lets compute-sanitizer and ncu run on a small process with only our kernel
in it, and makes the C++ tests possible without Python.

### Python extension (`setup.py`)

```
make build                 # = TORCH_CUDA_ARCH_LIST=<this GPU> uv pip install -e . --no-build-isolation
FA_PTXAS_VERBOSE=1 make build   # also print registers / spills per kernel instantiation
```

nvcc flags: `-O3 -std=c++20 --use_fast_math -lineinfo`. `-std=c++20` because PyTorch 2.14's
headers require it. `--use_fast_math` turns `exp2f` into the single `ex2.approx` instruction.
`-lineinfo` maps machine code back to source lines for ncu and compute-sanitizer, at no
runtime cost.

### CMake (`CMakeLists.txt`, `CMakePresets.json`)

| Preset | What it is for |
|---|---|
| `debug` | warnings are errors (`-Werror`), `-G` device debug info for cuda-gdb. Never benchmark it. |
| `release` | optimized + `-lineinfo`; used by sanitizers, ncu and `bench/tune.sh` |
| `asan` | debug + AddressSanitizer / UBSan on host code |

`make dev` builds all three. The GPU architecture is set before `project()` (otherwise nvcc's
default sm_75 wins); check with `cuobjdump --list-elf build/release/fa_dev`.

Targets: `fa_kernel` (static library with the kernel), `fa_dev` (driver), `fa_tests`
(GoogleTest; downloaded automatically at configure time, disable with `-DFA_BUILD_TESTS=OFF`).

### Standalone driver

```
build/release/fa_dev [B H N D causal iters]     # default 4 8 4096 128 0 20
B=4 H=8 N=4096 D=128 causal=0  max_abs_err=3.14e-05  median=18.947 ms  14.5 TFLOPS
```

It checks a sample of rows against a double-precision CPU reference, then times the kernel.

## Repository layout

```
csrc/
  common.cuh              CUDA_CHECK macros, swizzle(), inline-PTX wrappers (mma, ldmatrix, cp.async)
  flash_fwd.h             C++ API: FlashFwdParams + flash_fwd()
  kernels/flash_fwd.cu    the kernel, tile configs, launcher
  bindings.cpp            PyTorch binding: fa.forward(q, k, v, causal=False, softmax_scale=None)
  dev/main.cu             standalone driver
fa/                       Python package (__init__.py, reference.py)
tests/
  test_flash_fwd.py       pytest: correctness vs float32 PyTorch reference
  cpp/                    GoogleTest: PTX building blocks + C++ API
bench/
  benchmark.py            sweep vs torch SDPA backends -> bench/results/<gpu>.csv
  plot.py                 CSV -> docs/img/<gpu>.png + Markdown table
  tune.sh                 tile-configuration sweep
profile/
  ncu.sh, nsys.sh         Nsight Compute / Systems wrappers; reports in profile/reports/
scripts/lint.sh           clang-format + clang-tidy
docs/                     this documentation
Makefile                  entry point for everything above
```

## Editor setup (clangd)

- `.vscode/settings.json` points the clangd extension at `clangd-22`; the plain `clangd` on PATH
  is v21, which can't fully parse CUDA 13 + GCC 15 headers.
- `.clangd` reads `build/debug/compile_commands.json` (written by CMake) and strips the nvcc-only
  flags clang doesn't understand (`--use_fast_math`, `-G`, `--options-file`, ...).
- `CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF` in `CMakeLists.txt`: by default CMake hides
  nvcc's `-I` paths in a response file passed as `--options-file`. Stripping that flag also
  stripped the include paths, so clangd could not find `common.cuh` / `gtest.h` and reported
  `CUDA_CHECK` and `fa` as undeclared in every file.
- `csrc/bindings.cpp` is built by setup.py, not CMake, so `.clangd` has a section giving it its
  own flags (C++20, PyTorch and Python include paths, `-DTORCH_EXTENSION_NAME=_C`). Those paths
  are absolute and machine-specific; regenerate them on a new machine with the command in
  the `.clangd` comment.
- `scripts/lint.sh` runs clang-format and clang-tidy on `csrc/` and `tests/cpp/`, including
  `bindings.cpp` (PyTorch include paths read from `.venv` at run time). Status: 0 warnings.
  The kernel has two documented `NOLINT`s: its cognitive complexity (one long unrolled
  function on purpose) and the `extern __shared__` buffer (the CUDA idiom for dynamic shared
  memory).

After changing `.clangd` or re-running CMake, restart clangd in the editor
(VS Code: "clangd: Restart language server"). Verify from the terminal with
`clangd-22 --check=<file>`. Lines starting with `tweak: ... FAIL` there are refactorings it
tried, not errors.
