# FlashAttention-2 forward, from scratch in CUDA C++

A FlashAttention-2 forward kernel written in raw CUDA C++ and inline PTX, with no CUTLASS,
CuTe, Triton or cuDNN. It is benchmarked against the production implementations shipped in
PyTorch.

- **Tensor cores** via `mma.sync.m16n8k16` + `ldmatrix` / `ldmatrix.trans`. S and P never leave registers.
- **`cp.async` pipeline** overlapping K/V loads with compute. **XOR-swizzled** shared memory
  (bank-conflict-free, no padding).
- Online softmax in base 2. Causal masking only on diagonal tiles, with masked tiles skipped.
  N need not be a multiple of the tile size.
- FP16 in, FP32 accumulate, head dim 64 / 128, causal and non-causal, tile shapes tuned per case.

```python
import fa
o = fa.forward(q, k, v, causal=True)   # q, k, v: [B, H, N, d] fp16 CUDA tensors, d in {64, 128}
```

## Results (RTX 3050 Laptop, sm_86)

![benchmark](docs/img/NVIDIA_GeForce_RTX_3050_Laptop_GPU.png)

Faster than PyTorch SDPA's FlashAttention-2 backend at all 24 benchmarked shapes:
**1.01–1.06x non-causal, 1.02–1.11x causal**. That's 14.5–14.8 TFLOPS at large N, equal to a
cuBLAS fp16 GEMM on this GPU (14.5 TFLOPS). Full table, method and caveats are in
[docs/benchmarking.md](docs/benchmarking.md).

| d=128, N=16k | ours | FA2 (SDPA flash) | cuDNN | mem-efficient | naive torch |
|---|---|---|---|---|---|
| non-causal | **14.5** | 14.3 | 13.7 | 9.3 | OOM |
| causal | **14.0** | 12.7 | 13.3 | 8.9 | OOM |

FlashAttention-3/4 need Hopper/datacenter-Blackwell hardware and don't run on consumer GPUs, so
FA2 is the bar here. RTX 4090 / 5090 numbers are coming.

## Quick start

```
uv venv --python 3.12 .venv
uv pip install --python .venv torch numpy pytest pandas matplotlib ninja setuptools \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
make build        # Python extension (ARCH=8.9 for RTX 4090, 12.0 for RTX 5090)
make test         # GoogleTest + pytest
make sanitize     # compute-sanitizer memcheck / racecheck / synccheck / initcheck
make bench        # benchmark sweep + plot
make profile      # Nsight Compute reports (needs sudo for GPU counters)
```

## Documentation

| Doc | Contents |
|---|---|
| [docs/optimizations.md](docs/optimizations.md) | how the kernel works, every optimization and why |
| [docs/benchmarking.md](docs/benchmarking.md) | baselines, method, full results, tile tuning |
| [docs/testing.md](docs/testing.md) | pytest, GoogleTest building-block tests, sanitizers |
| [docs/profiling.md](docs/profiling.md) | Nsight Compute / Systems scripts and what to look for |
| [docs/setup.md](docs/setup.md) | toolchain, the two build systems, repo layout |
| [docs/worklog.md](docs/worklog.md) | what was done, in order, and open items |
