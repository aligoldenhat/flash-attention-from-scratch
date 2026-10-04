"""FlashAttention-2 forward pass written from scratch in CUDA C++.

    import fa
    o = fa.forward(q, k, v, causal=False)   # q, k, v: [B, H, N, D] fp16 CUDA, D in {64, 128}
    o = fa.forward(q, k, v, fp16_accum=True)  # faster on GeForce GPUs, slightly less exact
"""

import torch

from fa import _C
from fa.reference import attention_reference


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    fp16_accum: bool = False,
) -> torch.Tensor:
    """O = softmax(Q K^T * scale) V.

    fp16_accum: accumulate P @ V in fp16 within each 64-key tile (folded into fp32 between
    tiles). GeForce GPUs run fp16-accumulate tensor-core math at twice the fp32 rate.
    """
    variant = "fp16acc" if fp16_accum else "opt"
    return _C.forward(q, k, v, causal, softmax_scale, variant)


def forward_variant(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, variant: str = "opt"
) -> torch.Tensor:
    """Run a specific kernel build: 'baseline', 'opt' or 'fp16acc' (benchmarks, tests)."""
    return _C.forward(q, k, v, causal, None, variant)


__all__ = ["forward", "forward_variant", "attention_reference"]
