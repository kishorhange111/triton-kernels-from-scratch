"""Correctness against PyTorch references.

On CPU these run under ``TRITON_INTERPRET=1`` (set in conftest), which executes the same
kernel source with NumPy, so CI checks the real kernels without a GPU.
"""

import pytest
import torch

from tkern import flash_attention, reference_attention, reference_rms_norm, rms_norm, softmax

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [16, 37, 64])  # 37: not a multiple of the block size
def test_attention_forward_and_backward(causal, n):
    torch.manual_seed(n)
    q, k, v = (torch.randn(2, 2, n, 16, device=DEV, requires_grad=True) for _ in range(3))
    do = torch.randn(2, 2, n, 16, device=DEV)

    out = flash_attention(q, k, v, causal=causal, block_m=16, block_n=16)
    out.backward(do)
    got = (out.detach(), q.grad, k.grad, v.grad)

    for t in (q, k, v):
        t.grad = None
    ref = reference_attention(q, k, v, causal=causal)
    ref.backward(do)
    want = (ref.detach(), q.grad, k.grad, v.grad)

    for name, a, b in zip(("out", "dq", "dk", "dv"), got, want, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4, msg=name)


def test_attention_rectangular_blocks():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 48, 32, device=DEV, requires_grad=True) for _ in range(3))
    out = flash_attention(q, k, v, causal=True, block_m=32, block_n=16)
    out.sum().backward()
    g = q.grad.clone()
    q.grad = None
    reference_attention(q, k, v, causal=True).sum().backward()
    torch.testing.assert_close(out, reference_attention(q, k, v, causal=True), atol=2e-4, rtol=0)
    torch.testing.assert_close(g, q.grad, atol=2e-4, rtol=2e-4)


def test_attention_is_stable_with_large_logits():
    """The online softmax must not overflow where a naive exp would."""
    q = torch.full((1, 1, 32, 16), 30.0, device=DEV)
    k = torch.full((1, 1, 32, 16), 30.0, device=DEV)
    v = torch.randn(1, 1, 32, 16, device=DEV)
    out = flash_attention(q, k, v, block_m=16, block_n=16)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, v.mean(2, keepdim=True).expand_as(v), atol=1e-4, rtol=0)


def test_attention_rejects_bad_head_dim():
    x = torch.randn(1, 1, 8, 24, device=DEV)
    with pytest.raises(ValueError):
        flash_attention(x, x, x)


@pytest.mark.parametrize("cols", [7, 64, 1000])
def test_softmax(cols):
    x = torch.randn(5, cols, device=DEV) * 10
    torch.testing.assert_close(softmax(x), x.softmax(-1), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("cols", [16, 100])
def test_rms_norm_forward_and_backward(cols):
    torch.manual_seed(cols)
    x = torch.randn(3, 4, cols, device=DEV, requires_grad=True)
    w = torch.randn(cols, device=DEV, requires_grad=True)
    dy = torch.randn(3, 4, cols, device=DEV)

    y = rms_norm(x, w)
    y.backward(dy)
    got = (y.detach(), x.grad, w.grad)
    x.grad = w.grad = None
    ref = reference_rms_norm(x, w)
    ref.backward(dy)
    for name, a, b in zip(("y", "dx", "dw"), got, (ref.detach(), x.grad, w.grad), strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4, msg=name)
