"""Benchmark fa.forward against production attention implementations.

    python bench/benchmark.py                 # full sweep -> bench/results/<gpu>.csv
    python bench/benchmark.py --n 4096 --d 128 --causal 1

Shapes follow the FlashAttention-2 paper: batch * seqlen = 16k tokens and hidden size 2048,
i.e. B = 16384 / N and H = 2048 / d, so every N does comparable total work per token.

Timing: CUDA events around each call, warmup first, median of many iterations.
FLOPs = 4 * B * H * N^2 * d (two matmuls of 2*N^2*d each), halved for causal.

Each implementation gets a setup step outside the timed region (layout conversion to its
native format, planning), so every library is timed on its own preferred input layout.
"""

import argparse
import csv
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import fa
from fa.reference import attention_naive_fp16


# FlashInfer compiles its kernels on first use with ninja, which it looks up on PATH; make the
# venv's ninja visible even when this script is run as .venv/bin/python without activating.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
try:
    import flashinfer
except ImportError:  # optional baseline
    flashinfer = None


def _identity(o):
    return o


# An implementation is a setup function: setup(q, k, v, causal) -> (run, to_bhnd), where
# run() is what gets timed and to_bhnd converts its output back to [B, H, N, d] for the
# correctness check.
def _simple(fn):
    def setup(q, k, v, causal):
        return (lambda: fn(q, k, v, causal)), _identity

    return setup


def _sdpa(backend):
    def run(q, k, v, causal):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return _simple(run)


def _ours(variant):
    return _simple(lambda q, k, v, causal: fa.forward_variant(q, k, v, causal=causal, variant=variant))


_FLASHINFER_WORKSPACE = None


def _flashinfer(fp16_qk):
    """FlashInfer batch prefill on its native ragged token-major layout [B*N, H, d].
    fp16_qk: FlashInfer's own fp16-accumulation option (for Q K^T)."""

    def setup(q, k, v, causal):
        global _FLASHINFER_WORKSPACE
        b, h, n, d = q.shape
        if _FLASHINFER_WORKSPACE is None:
            _FLASHINFER_WORKSPACE = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        def to_nhd(x):
            return x.transpose(1, 2).reshape(b * n, h, d).contiguous()

        qn, kn, vn = to_nhd(q), to_nhd(k), to_nhd(v)
        indptr = torch.arange(0, (b + 1) * n, n, device="cuda", dtype=torch.int32)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_FLASHINFER_WORKSPACE, "NHD")
        wrapper.plan(indptr, indptr, h, h, d, causal=causal, q_data_type=torch.half,
                     use_fp16_qk_reduction=fp16_qk)
        return (lambda: wrapper.run(qn, kn, vn)), (lambda o: o.reshape(b, n, h, d).transpose(1, 2))

    return setup


IMPLS = {
    "fa fp16-acc (ours)": _ours("fp16acc"),
    "fa opt (ours)": _ours("opt"),
    "fa baseline (ours)": _ours("baseline"),
    "sdpa-flash (FA2)": _sdpa(SDPBackend.FLASH_ATTENTION),
    "sdpa-cudnn": _sdpa(SDPBackend.CUDNN_ATTENTION),
    "sdpa-efficient": _sdpa(SDPBackend.EFFICIENT_ATTENTION),
    "torch naive": _simple(attention_naive_fp16),
}
if flashinfer is not None:
    # use_fp16_qk_reduction=True would be FlashInfer's fp16-accumulation mode, but its JIT build
    # rejects it unless compiled with -DFP16_QK_REDUCTION_SUPPORTED and Boost.Math.
    IMPLS["flashinfer"] = _flashinfer(fp16_qk=False)


def time_ms(fn, warmup=10, min_iters=20, max_iters=200, budget_ms=1500.0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    while len(times) < max_iters and (len(times) < min_iters or sum(times) < budget_ms):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def bench_one(setup, b, h, n, d, causal, ref_out):
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.half, generator=gen) for _ in range(3))
    try:
        run, to_bhnd = setup(q, k, v, causal)
        out = to_bhnd(run())
        if ref_out is not None:
            err = (out.float() - ref_out.float()).abs().max().item()
            if not err < 2e-2:
                return {"status": f"mismatch {err:.1e}"}
        del out
        ms = time_ms(run)
    except torch.OutOfMemoryError:
        return {"status": "OOM"}
    except RuntimeError as e:  # backend not available for this GPU / shape
        msg = str(e).splitlines()[0][:60]
        return {"status": f"unsupported: {msg}"}
    finally:
        torch.cuda.empty_cache()
    flops = 4 * b * h * n * n * d / (2 if causal else 1)
    return {"status": "ok", "ms": ms, "tflops": flops / ms / 1e9}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="*", default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--d", type=int, nargs="*", default=[64, 128])
    p.add_argument("--causal", type=int, nargs="*", default=[0, 1])
    p.add_argument("--impl", nargs="*", default=list(IMPLS))
    p.add_argument("--tokens", type=int, default=16384, help="B * N")
    p.add_argument("--hidden", type=int, default=2048, help="H * d")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    gpu = torch.cuda.get_device_name()
    out = args.out or Path(__file__).parent / "results" / (gpu.replace(" ", "_") + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"GPU: {gpu}   torch {torch.__version__}   CUDA {torch.version.cuda}")
    for d in args.d:
        for causal in map(bool, args.causal):
            for n in args.n:
                b, h = max(1, args.tokens // n), args.hidden // d
                # correctness cross-check of every impl against ours (cheap, same inputs)
                gen = torch.Generator(device="cuda").manual_seed(0)
                q, k, v = (
                    torch.randn(b, h, n, d, device="cuda", dtype=torch.half, generator=gen)
                    for _ in range(3)
                )
                ref_out = fa.forward_variant(q, k, v, causal=causal, variant="baseline")
                del q, k, v
                line = [f"d={d:3d} causal={int(causal)} N={n:5d} B={b:2d} H={h:2d} |"]
                for name in args.impl:
                    r = bench_one(IMPLS[name], b, h, n, d, causal, ref_out)
                    rows.append(
                        {"gpu": gpu, "impl": name, "B": b, "H": h, "N": n, "d": d,
                         "causal": int(causal), "ms": r.get("ms", math.nan),
                         "tflops": r.get("tflops", math.nan), "status": r["status"]}
                    )
                    cell = f"{r['tflops']:6.1f}" if r["status"] == "ok" else f"{r['status'][:6]:>6}"
                    line.append(f"{name.replace(' (ours)', '')}: {cell.strip()}")
                del ref_out
                torch.cuda.empty_cache()
                print("  ".join(line), flush=True)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
