"""Correctness of fa.forward against a float32 PyTorch reference."""

import math

import pytest
import torch

import fa

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

ATOL = 1e-2
RTOL = 1e-2


def _qkv(b, h, n, d, seed=0, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (torch.randn(b, h, n, d, device="cuda", generator=g) * scale).half() for _ in range(3)
    )


# N covers: 1 row, below one tile, exactly one tile, odd sizes that leave partial Q and
# K/V tiles, multiples of the tile, and a longer sequence.
@pytest.mark.parametrize("n", [1, 17, 63, 64, 65, 100, 128, 257, 511, 1000, 2048])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_reference(n, d, causal):
    q, k, v = _qkv(2, 3, n, d)
    out = fa.forward(q, k, v, causal=causal)
    ref = fa.attention_reference(q, k, v, causal=causal)
    assert out.dtype == torch.float16 and out.shape == q.shape
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("b,h", [(1, 1), (3, 5), (8, 16)])
def test_batch_heads(b, h):
    q, k, v = _qkv(b, h, 300, 64, seed=1)
    torch.testing.assert_close(
        fa.forward(q, k, v).float(), fa.attention_reference(q, k, v), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("causal", [False, True])
def test_large_logits_are_stable(causal):
    """Scores of magnitude ~100s: a softmax without max-subtraction would overflow."""
    q, k, v = _qkv(1, 2, 512, 64, seed=2, scale=4.0)
    out = fa.forward(q, k, v, causal=causal)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(), fa.attention_reference(q, k, v, causal=causal), atol=ATOL, rtol=RTOL
    )


def test_custom_softmax_scale():
    q, k, v = _qkv(1, 2, 200, 128, seed=3)
    scale = 0.3
    out = fa.forward(q, k, v, softmax_scale=scale)
    # reference with a different scale = reference on rescaled q
    ref = fa.attention_reference((q.float() * scale * math.sqrt(128)), k, v)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_causal_first_row_is_v0():
    """Row 0 attends only to key 0, so its output must equal V[0]."""
    q, k, v = _qkv(1, 1, 130, 64, seed=4)
    out = fa.forward(q, k, v, causal=True)
    torch.testing.assert_close(out[0, 0, 0].float(), v[0, 0, 0].float(), atol=1e-3, rtol=0)


def test_deterministic():
    q, k, v = _qkv(2, 4, 777, 128, seed=5)
    assert torch.equal(fa.forward(q, k, v), fa.forward(q, k, v))


@pytest.mark.parametrize(
    "bad, msg",
    [
        (lambda q: q.float(), "float16"),
        (lambda q: q.cpu(), "CUDA"),
        (lambda q: q.transpose(2, 3), "contiguous"),
        (lambda q: q[..., :32].contiguous(), "same shape"),
    ],
)
def test_input_validation(bad, msg):
    q, k, v = _qkv(1, 1, 64, 64)
    with pytest.raises(RuntimeError, match=msg):
        fa.forward(bad(q), k, v)


def test_unsupported_head_dim():
    q, k, v = _qkv(1, 1, 64, 32)
    with pytest.raises((RuntimeError, ValueError), match="head_dim"):
        fa.forward(q, k, v)
