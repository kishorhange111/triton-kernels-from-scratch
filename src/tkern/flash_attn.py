"""FlashAttention-2 forward and backward in Triton.

Layout: q, k, v are contiguous ``(B, H, N, D)``; ``D`` is a power of two >= 16.

Forward (one program per query block): stream key/value blocks through SRAM, keep a running
row max ``m`` and normaliser ``l`` (online softmax), and never materialise the N x N score
matrix. It stores the log-sum-exp ``L`` per row, which is all the backward pass needs to
recompute probabilities.

Backward (two kernels, no atomics):
- ``_bwd_dkdv``: one program per key block, loops over query blocks, accumulates dK and dV.
- ``_bwd_dq``: one program per query block, loops over key blocks, accumulates dQ.

Recomputing P costs extra FLOPs but saves O(N^2) memory traffic; that trade is the paper's
point. Scores are computed in base 2 (``exp2``) with ``log2(e)`` folded into the scale.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _fwd(Q, K, V, O, L, qk_scale, N, D: tl.constexpr, BLOCK_M: tl.constexpr,
         BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):  # fmt: skip
    start_m = tl.program_id(0)
    base = tl.program_id(1).to(tl.int64) * N * D
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < N

    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N)
    else:
        hi = N
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        col_ok = cols < N
        kv_off = base + cols[:, None] * D + offs_d[None, :]
        k = tl.load(K + kv_off, mask=col_ok[:, None], other=0.0)
        v = tl.load(V + kv_off, mask=col_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = col_ok[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= cols[None, :])
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)  # rows past N see only -inf
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=row_ok[:, None])  # fmt: skip
    # Log-sum-exp in base 2, in units of the scaled scores.
    tl.store(L + tl.program_id(1).to(tl.int64) * N + offs_m, m_i + tl.log2(l_safe), mask=row_ok)


@triton.jit
def _bwd_preprocess(O, DO, Delta, N, D: tl.constexpr, BLOCK_M: tl.constexpr):
    """Delta_i = sum_j P_ij dP_ij = rowsum(dO * O)."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    bh = tl.program_id(1).to(tl.int64)
    off = bh * N * D + offs_m[:, None] * D + offs_d[None, :]
    ok = offs_m[:, None] < N
    o = tl.load(O + off, mask=ok, other=0.0).to(tl.float32)
    do = tl.load(DO + off, mask=ok, other=0.0).to(tl.float32)
    tl.store(Delta + bh * N + offs_m, tl.sum(o * do, 1), mask=offs_m < N)


@triton.jit
def _bwd_dkdv(Q, K, V, DO, L, Delta, DK, DV, sm_scale, qk_scale, N, D: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):  # fmt: skip
    start_n = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * N * D
    cols = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    col_ok = cols < N
    kv_off = base + cols[:, None] * D + offs_d[None, :]
    k = tl.load(K + kv_off, mask=col_ok[:, None], other=0.0)
    v = tl.load(V + kv_off, mask=col_ok[:, None], other=0.0)
    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)

    if CAUSAL:
        lo = (start_n * BLOCK_N) // BLOCK_M * BLOCK_M  # earlier query rows never see these keys
    else:
        lo = 0
    for start_m in range(lo, N, BLOCK_M):
        rows = start_m + offs_m
        row_ok = rows < N
        q_off = base + rows[:, None] * D + offs_d[None, :]
        q = tl.load(Q + q_off, mask=row_ok[:, None], other=0.0)
        do = tl.load(DO + q_off, mask=row_ok[:, None], other=0.0)
        lse = tl.load(L + bh * N + rows, mask=row_ok, other=0.0)
        di = tl.load(Delta + bh * N + rows, mask=row_ok, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = row_ok[:, None] & col_ok[None, :]
        if CAUSAL:
            keep = keep & (rows[:, None] >= cols[None, :])
        p = tl.where(keep, tl.exp2(s - lse[:, None]), 0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - di[:, None]) * sm_scale
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    tl.store(DK + kv_off, dk.to(DK.dtype.element_ty), mask=col_ok[:, None])
    tl.store(DV + kv_off, dv.to(DV.dtype.element_ty), mask=col_ok[:, None])


@triton.jit
def _bwd_dq(Q, K, V, DO, L, Delta, DQ, sm_scale, qk_scale, N, D: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):  # fmt: skip
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * N * D
    rows = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    row_ok = rows < N
    q_off = base + rows[:, None] * D + offs_d[None, :]
    q = tl.load(Q + q_off, mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + q_off, mask=row_ok[:, None], other=0.0)
    lse = tl.load(L + bh * N + rows, mask=row_ok, other=0.0)
    di = tl.load(Delta + bh * N + rows, mask=row_ok, other=0.0)
    dq = tl.zeros((BLOCK_M, D), tl.float32)

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N)
    else:
        hi = N
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        col_ok = cols < N
        kv_off = base + cols[:, None] * D + offs_d[None, :]
        k = tl.load(K + kv_off, mask=col_ok[:, None], other=0.0)
        v = tl.load(V + kv_off, mask=col_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = row_ok[:, None] & col_ok[None, :]
        if CAUSAL:
            keep = keep & (rows[:, None] >= cols[None, :])
        p = tl.where(keep, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - di[:, None]) * sm_scale
        dq += tl.dot(ds.to(k.dtype), k)

    tl.store(DQ + q_off, dq.to(DQ.dtype.element_ty), mask=row_ok[:, None])


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not (q.shape == k.shape == v.shape) or q.dim() != 4:
        raise ValueError("q, k, v must share shape (B, H, N, D)")
    d = q.shape[-1]
    if d < 16 or d & (d - 1):
        raise ValueError("head dim must be a power of two >= 16")


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_m, block_n, launch):
        _check(q, k, v)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, H, N, D = q.shape
        o = torch.empty_like(q)
        lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(N, block_m), B * H)
        _fwd[grid](q, k, v, o, lse, sm_scale * LOG2E, N, D, block_m, block_n, causal, **launch)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.sm_scale, ctx.blocks, ctx.launch = (
            causal,
            sm_scale,
            (block_m, block_n),
            launch,
        )
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        bm, bn = ctx.blocks
        B, H, N, D = q.shape
        do = do.contiguous()
        delta = torch.empty_like(lse)
        _bwd_preprocess[(triton.cdiv(N, bm), B * H)](o, do, delta, N, D, bm)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        qk = ctx.sm_scale * LOG2E
        _bwd_dkdv[(triton.cdiv(N, bn), B * H)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk,
            dv,
            ctx.sm_scale,
            qk,
            N,
            D,
            bm,
            bn,
            ctx.causal,
            **ctx.launch,
        )
        _bwd_dq[(triton.cdiv(N, bm), B * H)](
            q, k, v, do, lse, delta, dq, ctx.sm_scale, qk, N, D, bm, bn, ctx.causal, **ctx.launch
        )
        return dq, dk, dv, None, None, None, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """softmax(q k^T * scale [+ causal mask]) v, differentiable in q, k and v.

    ``num_warps`` / ``num_stages`` are Triton launch parameters (None = Triton's default).
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    launch = {}
    if num_warps is not None:
        launch["num_warps"] = num_warps
    if num_stages is not None:
        launch["num_stages"] = num_stages
    return _FlashAttention.apply(q, k, v, causal, scale, block_m, block_n, launch)


def forward_ptx(q, k, v, causal=False, block_m=64, block_n=64, **launch) -> str:
    """Compile and run the forward kernel once; return its PTX (to check for ``mma``)."""
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    kern = _fwd[(triton.cdiv(N, block_m), B * H)](
        q, k, v, o, lse, LOG2E / math.sqrt(D), N, D, block_m, block_n, causal, **launch
    )
    return kern.asm.get("ptx", "")


def reference_attention(q, k, v, causal=False, sm_scale=None):
    scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(
            torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf")
        )
    return (s.softmax(-1) @ v.float()).to(q.dtype)
