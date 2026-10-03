# Testing and memory checking

Three layers, from the outside in:

| Layer | Tool | Checks | Run |
|---|---|---|---|
| End to end | pytest (57 tests) | `fa.forward` vs a float32 PyTorch reference, many shapes | `make test` |
| Building blocks + C++ API | GoogleTest (19 tests) | mma / ldmatrix layouts, swizzle, cp.async, argument validation, small end-to-end | `make test` |
| Memory / races | compute-sanitizer | out-of-bounds, races, barrier misuse, uninitialised reads | `make sanitize` |

## pytest (`tests/test_flash_fwd.py`)

The main correctness suite. Every case compares against `fa.attention_reference`, exact
attention computed in float32, with `atol = rtol = 1e-2` (fp16 inputs, fp32 accumulation;
measured errors are around 1e-3).

- `test_matches_reference`: N ∈ {1, 17, 63, 64, 65, 100, 128, 257, 511, 1000, 2048} × d ∈ {64, 128}
  × causal on/off. The N values hit a single row, partial Q tiles and partial K/V tiles, exact
  tile multiples and several tiles.
- `test_batch_heads`: batch/head indexing (B·H up to 128).
- `test_large_logits_are_stable`: scores in the hundreds; a softmax without max subtraction
  would overflow to inf.
- `test_custom_softmax_scale`, `test_causal_first_row_is_v0` (row 0 sees only key 0, so its
  output must equal V[0]), `test_deterministic` (bitwise equal on repeat).
- `test_input_validation`, `test_unsupported_head_dim`: wrong dtype / device / layout / shape
  raise errors instead of computing garbage.

## GoogleTest (`tests/cpp/`)

pytest only sees the final output: when it fails you learn that "the output is wrong", not
which piece is. The C++ tests launch tiny kernels that check each building block alone.

`test_ptx.cu`, building blocks from `csrc/common.cuh`. Values are small integers (exact in
fp16 and fp32), so results are compared for exact equality:

| Test | What it proves |
|---|---|
| `Mma16816.FragmentLayoutMatchesCpuMatmul` | builds A and B fragments by hand from the layout table (g = lane/4, t = lane%4), runs `mma.sync`, scatters C by the same table, compares with a CPU matmul. A wrong table entry would permute C. |
| `Ldmatrix.X4GivesMmaAFragment` | with the kernel's Q addressing, `ldmatrix.x4` puts exactly the A-fragment elements in every lane's 4 registers |
| `Ldmatrix.X4TransGivesMmaBFragmentOfRowMajorV` | with the kernel's V addressing, `ldmatrix.x4.trans` gives the B fragment (two consecutive *keys* per register) |
| `Swizzle.IsAPermutationWithinEachRow` | no two chunks collide, every chunk stays 16-byte aligned and inside its row |
| `Swizzle.LdmatrixColumnReadsAreBankConflictFree` | the 8 rows of one ldmatrix phase land in 8 different bank groups, and without the swizzle they would all land in one |
| `Swizzle.RowWritesAreBankConflictFree` | cp.async row writes stay conflict-free |
| `CpAsync.SrcSizeZeroWritesZeros` | `cp.async` with src-size 0 writes zeros: the padding used for rows ≥ N |

`test_flash_fwd.cu`, the C++ API `fa::flash_fwd`:

- `FlashFwdArgs.*`: head_dim 32, empty shapes and misaligned pointers throw
  `std::invalid_argument` (Python sees `ValueError`).
- `Shapes/FlashFwdVsCpu.MatchesReference/*`: 9 small shapes against a double-precision CPU
  reference, so `ctest` alone is a meaningful check on a machine without PyTorch.

### Do the tests catch bugs? (mutation check)

A test that never fails proves nothing, so two bugs were planted on purpose:

1. **Swizzle without the XOR** (`(chunk ^ (row & 7)) << 3` → `chunk << 3`): only
   `Swizzle.LdmatrixColumnReadsAreBankConflictFree` failed. pytest and the end-to-end tests
   still passed, because the output is still correct and the kernel just gets slower (8-way
   bank conflicts). Only a unit test can catch this kind of performance bug.
2. **Wrong ldmatrix lane → row mapping**: `Ldmatrix.X4GivesMmaAFragment` failed.

Both were reverted.

### Running

```
make test                                          # ctest + pytest
ctest --test-dir build/release --output-on-failure
build/release/fa_tests --gtest_filter='Swizzle*'   # a subset, with GoogleTest's own output
```

## compute-sanitizer (`make sanitize`)

compute-sanitizer is to CUDA what ASan (AddressSanitizer) is to C++: it instruments the
kernel and reports bad memory accesses with the exact source line (thanks to `-lineinfo`).

| Tool | Finds |
|---|---|
| `memcheck` | out-of-bounds or misaligned global/shared accesses |
| `racecheck` | shared-memory data races (e.g. a missing `__syncthreads` in the cp.async pipeline) |
| `synccheck` | invalid barrier use (e.g. `__syncthreads` in divergent code) |
| `initcheck` | reads of uninitialised global memory |

`make sanitize` runs all four on the standalone driver for six shapes (odd N, both head dims,
causal on/off, N = 1) and on the GoogleTest binary, then memcheck and racecheck on a pytest
subset (`--kernel-name kns=flash_fwd_kernel` limits checking to our kernel, not PyTorch's).
Current result: **0 errors, 0 hazards** everywhere.

Negative check done during setup: removing a kernel's bounds check makes memcheck report
`Invalid __global__ read of size 4 bytes`. So the tool is live, not silently passing.
