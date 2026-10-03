"""FlashAttention-2 forward pass written from scratch in CUDA C++.

    import fa
    o = fa.forward(q, k, v, causal=False)   # q, k, v: [B, H, N, D] fp16 CUDA, D in {64, 128}
"""

import torch  # noqa: F401  (loads libtorch / libc10 before the extension)

from fa._C import forward
from fa.reference import attention_reference

__all__ = ["forward", "attention_reference"]
