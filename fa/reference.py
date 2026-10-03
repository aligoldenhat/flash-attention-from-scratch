"""Reference implementations used by the tests and the benchmark baselines."""

import math

import torch


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Exact attention computed in float32 (the ground truth for the tests)."""
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = (q32 @ k32.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    if causal:
        n = q.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        scores.masked_fill_(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v32


def attention_naive_fp16(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """What a straightforward PyTorch user writes: materializes the N x N matrix in fp16."""
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    if causal:
        n = q.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        scores.masked_fill_(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
