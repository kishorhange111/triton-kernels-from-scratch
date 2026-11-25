"""Fused row-wise kernels: softmax (forward) and RMSNorm (forward and backward).

Both are memory-bound: each row is read once and written once, so the target is HBM
bandwidth, not FLOPs. Fusing avoids the intermediate tensors eager PyTorch writes
(``x.pow(2).mean()``, ``rsqrt``, the multiply...). One program handles one row, and the
row must fit in a block (``BLOCK >= n_cols``), which covers hidden sizes up to 64k.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd(X, Y, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    x = tl.load(X + row * stride + cols, mask=ok, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    tl.store(Y + row * stride + cols, (e / tl.sum(e, 0)).to(Y.dtype.element_ty), mask=ok)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(shape[-1])
    _softmax_fwd[(x2.shape[0],)](x2, y, shape[-1], x2.stride(0), block)
    return y.reshape(shape)


@triton.jit
def _rms_fwd(X, W, Y, RSTD, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=ok, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=ok, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 0) / n_cols + eps)
    tl.store(RSTD + row, rstd)
    tl.store(Y + row * n_cols + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=ok)


@triton.jit
def _rms_bwd(DY, X, W, RSTD, DX, DW_PARTIAL, n_cols, BLOCK: tl.constexpr):
    """dx = rstd * (g - xhat * mean(g * xhat)) with g = dy * w and xhat = x * rstd.

    dW needs a reduction over rows; each program writes its row's contribution and the
    host sums them. (A production kernel would reduce in-kernel over groups of rows.)
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    off = row * n_cols + cols
    x = tl.load(X + off, mask=ok, other=0.0).to(tl.float32)
    dy = tl.load(DY + off, mask=ok, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=ok, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    xhat = x * rstd
    g = dy * w
    dx = rstd * (g - xhat * (tl.sum(g * xhat, 0) / n_cols))
    tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=ok)
    tl.store(DW_PARTIAL + off, dy * xhat, mask=ok)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        y = torch.empty_like(x2)
        rstd = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(shape[-1])
        _rms_fwd[(x2.shape[0],)](x2, w, y, rstd, shape[-1], eps, block)
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape, ctx.block = shape, block
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        dy2 = dy.reshape(x2.shape).contiguous()
        dx = torch.empty_like(x2)
        dw_partial = torch.empty(x2.shape, device=x2.device, dtype=torch.float32)
        _rms_bwd[(x2.shape[0],)](dy2, x2, w, rstd, dx, dw_partial, x2.shape[1], ctx.block)
        return dx.reshape(ctx.shape), dw_partial.sum(0).to(w.dtype), None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _RMSNorm.apply(x, weight, eps)


def reference_rms_norm(x, weight, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
