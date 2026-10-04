"""Correctness of fa.forward against a float32 PyTorch reference."""

import math

import pytest
import torch

import fa

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

ATOL = 1e-2
RTOL = 1e-2
# Every kernel build: the first tuned kernel, the default (opt), and the fp16-accumulate one.
VARIANTS = ["baseline", "opt", "fp16acc"]


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
@pytest.mark.parametrize("variant", VARIANTS)
def test_matches_reference(n, d, causal, variant):
    q, k, v = _qkv(2, 3, n, d)
    out = fa.forward_variant(q, k, v, causal=causal, variant=variant)
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
@pytest.mark.parametrize("variant", VARIANTS)
def test_large_logits_are_stable(causal, variant):
    """Scores of magnitude ~100s: a softmax without max-subtraction would overflow.
    Also exercises lazy rescaling (the max keeps growing by more than 2^8)."""
    q, k, v = _qkv(1, 2, 512, 64, seed=2, scale=4.0)
    out = fa.forward_variant(q, k, v, causal=causal, variant=variant)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(), fa.attention_reference(q, k, v, causal=causal), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("variant", VARIANTS)
def test_uniform_scores(d, variant):
    """q = 0: every score is equal, so every p = 1 and O = mean of V. The largest possible
    per-tile sums: the worst case for the fp16 partial sums of fp16acc."""
    _, k, v = _qkv(1, 2, 4096, d, seed=6)
    q = torch.zeros_like(k)
    out = fa.forward_variant(q, k, v, variant=variant)
    torch.testing.assert_close(
        out.float(), v.float().mean(dim=2, keepdim=True).expand_as(out), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("causal", [False, True])
def test_fp16acc_large_values_no_overflow(causal):
    """Large |V| over a long sequence: an fp16 accumulator over all of N would overflow
    (65504); fp16acc only sums one tile in fp16, then folds into fp32."""
    q, k, _ = _qkv(1, 2, 4096, 128, seed=7)
    v = (torch.randn(1, 2, 4096, 128, device="cuda") * 100.0).half()
    out = fa.forward(q, k, v, causal=causal, fp16_accum=True)
    assert torch.isfinite(out).all()
    ref = fa.attention_reference(q, k, v, causal=causal)
    torch.testing.assert_close(out.float(), ref, atol=100.0 * ATOL, rtol=RTOL)


def test_fp16_accum_flag_selects_variant():
    q, k, v = _qkv(1, 2, 256, 64, seed=8)
    assert torch.equal(fa.forward(q, k, v, fp16_accum=True), fa.forward_variant(q, k, v, variant="fp16acc"))
    assert torch.equal(fa.forward(q, k, v), fa.forward_variant(q, k, v, variant="opt"))


def test_unknown_variant():
    q, k, v = _qkv(1, 1, 64, 64)
    with pytest.raises(RuntimeError, match="variant"):
        fa.forward_variant(q, k, v, variant="nope")


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
