"""Paged-attention decode in Triton: vLLM's block-table KV cache plus flash-decoding split-K.

The KV cache is a pool of fixed-size pages, ``k_cache, v_cache: (num_pages, H_kv, P, D)``.
Each sequence owns a list of pages, given by ``block_tables[b]``, so sequences grow without
reallocation and without fragmenting memory. The query for one decode step is
``q: (B, H, D)``, with ``H = G · H_kv`` (grouped-query attention: G query heads share a
KV head).

With one program per (sequence, head), a batch of a few long sequences leaves most SMs idle.
**Flash-decoding** therefore splits each sequence's pages over ``S`` programs:

1. ``_partial``, grid ``(B, H, S)``: each program streams its share of pages and keeps an
   online-softmax running max ``m``, normaliser ``l`` and unnormalised output ``acc``.
2. ``_combine``, grid ``(B, H)``: rescales the S partial results to a common max and sums
   them: ``o = Σ 2^(m_s − M) acc_s / Σ 2^(m_s − M) l_s``.

One query row against a page is a matrix-vector product, so the kernel uses elementwise
multiply and reduce (no ``tl.dot``). Decode attention reads every cached byte once, so it
is bound by memory bandwidth, not FLOPs.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _partial(
    Q,
    K,
    V,
    BT,
    SL,
    Mo,
    Lo,
    Acc,
    qk_scale,
    max_pages,
    pages_per_split,
    H: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
):
    b, h, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = h // G
    S = tl.num_programs(2)
    seq_len = tl.load(SL + b)
    offs_p, offs_d = tl.arange(0, P), tl.arange(0, D)
    q = tl.load(Q + (b * H + h) * D + offs_d).to(tl.float32)
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    n_pages = tl.cdiv(seq_len, P)
    first = s * pages_per_split
    last = tl.minimum(first + pages_per_split, n_pages)
    for page in range(first, last):
        blk = tl.load(BT + b * max_pages + page).to(tl.int64)
        tok = page * P + offs_p
        ok = tok < seq_len
        off = ((blk * (H // G) + hk) * P + offs_p[:, None]) * D + offs_d[None, :]
        k = tl.load(K + off, mask=ok[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + off, mask=ok[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(k * q[None, :], 1) * qk_scale
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m, tl.max(sc, 0))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(sc - m_new)
        l = l * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m = m_new
    row = (b * H + h) * S + s
    tl.store(Mo + row, m)
    tl.store(Lo + row, l)
    tl.store(Acc + row * D + offs_d, acc)


@triton.jit
def _combine(Mo, Lo, Acc, O, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    base = (b * H + h) * S
    offs_s, offs_d = tl.arange(0, S), tl.arange(0, D)
    m = tl.load(Mo + base + offs_s)
    l = tl.load(Lo + base + offs_s)
    acc = tl.load(Acc + (base + offs_s[:, None]) * D + offs_d[None, :])
    M = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - M))  # empty splits contribute nothing
    o = tl.sum(w[:, None] * acc, 0) / tl.sum(w * l, 0)
    tl.store(O + (b * H + h) * D + offs_d, o.to(O.dtype.element_ty))


def paged_attention(
    q, k_cache, v_cache, block_tables, seq_lens, sm_scale=None, num_splits: int | None = None
):
    """q (B, H, D); caches (pages, H_kv, P, D); block_tables (B, max_pages) int32;
    seq_lens (B,) int32 with every length >= 1. Returns (B, H, D)."""
    B, H, D = q.shape
    _, Hk, P, _ = k_cache.shape
    if H % Hk:
        raise ValueError("query heads must be a multiple of KV heads")
    max_pages = block_tables.shape[1]
    if num_splits is None:  # enough programs to fill the GPU, at least one page each
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count if q.is_cuda else 8
        num_splits = max(1, min(max_pages, triton.cdiv(2 * sms, B * H)))
    S = triton.next_power_of_2(num_splits)
    pages_per_split = triton.cdiv(max_pages, S)
    scale = (1.0 / math.sqrt(D) if sm_scale is None else sm_scale) * LOG2E
    m = torch.empty(B * H * S, device=q.device, dtype=torch.float32)
    l, acc = torch.empty_like(m), torch.empty(B * H * S, D, device=q.device, dtype=torch.float32)
    _partial[(B, H, S)](
        q.contiguous(),
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        m,
        l,
        acc,
        scale,
        max_pages,
        pages_per_split,
        H,
        H // Hk,
        P,
        D,
    )
    o = torch.empty_like(q)
    _combine[(B, H)](m, l, acc, o, H, S, D)
    return o


def gather_kv(cache, block_tables, seq_lens):
    """Contiguous (B, H_kv, max_len, D) copy of each sequence's pages (zero-padded)."""
    B = block_tables.shape[0]
    pages = cache[block_tables.long()]  # (B, max_pages, Hk, P, D)
    x = pages.permute(0, 2, 1, 3, 4).reshape(B, cache.shape[1], -1, cache.shape[3])
    return x[:, :, : int(seq_lens.max())]


def reference_paged_attention(q, k_cache, v_cache, block_tables, seq_lens, sm_scale=None):
    B, H, D = q.shape
    G = H // k_cache.shape[1]
    k = gather_kv(k_cache, block_tables, seq_lens).float().repeat_interleave(G, 1)
    v = gather_kv(v_cache, block_tables, seq_lens).float().repeat_interleave(G, 1)
    scale = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    s = torch.einsum("bhd,bhtd->bht", q.float(), k) * scale
    valid = torch.arange(k.shape[2], device=q.device)[None, :] < seq_lens[:, None].long()
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    return torch.einsum("bht,bhtd->bhd", s.softmax(-1), v).to(q.dtype)


def make_paged_cache(
    seq_lens,
    n_kv_heads,
    head_dim,
    page=16,
    dtype=torch.float16,
    device="cpu",
    generator=None,
    extra_pages: int = 3,
):
    """Random KV pool with each sequence's pages scattered in random order (as after many
    allocations and frees), plus ``extra_pages`` unused ones."""
    pages_per = [math.ceil(int(n) / page) for n in seq_lens]
    total = sum(pages_per) + extra_pages
    perm = torch.randperm(total, generator=generator)
    tables = torch.zeros(len(seq_lens), max(pages_per), dtype=torch.int32)
    i = 0
    for b, n in enumerate(pages_per):
        tables[b, :n] = perm[i : i + n].to(torch.int32)
        i += n
    shape = (total, n_kv_heads, page, head_dim)
    k = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    v = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    return k, v, tables.to(device), torch.tensor(seq_lens, dtype=torch.int32, device=device)
