"""Benchmark fa.forward against production attention implementations.

    python bench/benchmark.py                 # full sweep -> bench/results/<gpu>.csv
    python bench/benchmark.py --n 4096 --d 128 --causal 1

Shapes follow the FlashAttention-2 paper: batch * seqlen = 16k tokens and hidden size 2048,
i.e. B = 16384 / N and H = 2048 / d, so every N does comparable total work per token.

Timing: CUDA events around each call, warmup first, median of many iterations.
FLOPs = 4 * B * H * N^2 * d (two matmuls of 2*N^2*d each), halved for causal.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import fa
from fa.reference import attention_naive_fp16


def _sdpa(backend):
    def run(q, k, v, causal):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return run


def _ours(variant):
    def run(q, k, v, causal):
        return fa.forward_variant(q, k, v, causal=causal, variant=variant)

    return run


IMPLS = {
    "fa fp16-acc (ours)": _ours("fp16acc"),
    "fa opt (ours)": _ours("opt"),
    "fa baseline (ours)": _ours("baseline"),
    "sdpa-flash (FA2)": _sdpa(SDPBackend.FLASH_ATTENTION),
    "sdpa-cudnn": _sdpa(SDPBackend.CUDNN_ATTENTION),
    "sdpa-efficient": _sdpa(SDPBackend.EFFICIENT_ATTENTION),
    "torch naive": attention_naive_fp16,
}


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


def bench_one(name, fn, b, h, n, d, causal, ref_out):
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.half, generator=gen) for _ in range(3))
    try:
        out = fn(q, k, v, causal)
        if ref_out is not None:
            err = (out.float() - ref_out.float()).abs().max().item()
            if not err < 2e-2:
                return {"status": f"mismatch {err:.1e}"}
        ms = time_ms(lambda: fn(q, k, v, causal))
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
                    r = bench_one(name, IMPLS[name], b, h, n, d, causal, ref_out)
                    rows.append(
                        {"gpu": gpu, "impl": name, "B": b, "H": h, "N": n, "d": d,
                         "causal": int(causal), "ms": r.get("ms", math.nan),
                         "tflops": r.get("tflops", math.nan), "status": r["status"]}
                    )
                    cell = f"{r['tflops']:6.1f}" if r["status"] == "ok" else f"{r['status'][:6]:>6}"
                    line.append(f"{name.split()[0]} {cell}")
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
