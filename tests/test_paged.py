import pytest
import torch

from tkern.paged_attn import (
    gather_kv,
    make_paged_cache,
    paged_attention,
    reference_paged_attention,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _setup(lens, hk=2, g=2, d=32, page=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    k, v, bt, sl = make_paged_cache(lens, hk, d, page, torch.float32, DEV, gen)
    q = torch.randn(len(lens), hk * g, d, generator=gen).to(DEV)
    return q, k, v, bt, sl


@pytest.mark.parametrize("splits", [1, 2, 4])
def test_matches_reference_with_ragged_lengths(splits):
    q, k, v, bt, sl = _setup([1, 17, 40, 64])  # 1 token, page boundary +1, partial, exact
    out = paged_attention(q, k, v, bt, sl, num_splits=splits)
    torch.testing.assert_close(
        out, reference_paged_attention(q, k, v, bt, sl), atol=1e-4, rtol=1e-4
    )


def test_more_splits_than_pages_leaves_empty_splits_harmless():
    q, k, v, bt, sl = _setup([5, 20])  # at most 2 pages, 8 splits
    out = paged_attention(q, k, v, bt, sl, num_splits=8)
    torch.testing.assert_close(
        out, reference_paged_attention(q, k, v, bt, sl), atol=1e-4, rtol=1e-4
    )


def test_gqa_heads_share_their_kv_head():
    q, k, v, bt, sl = _setup([33], hk=1, g=4)
    q[:] = q[:, :1]  # identical queries in one group must give identical outputs
    out = paged_attention(q, k, v, bt, sl, num_splits=2)
    for h in range(1, 4):
        torch.testing.assert_close(out[:, h], out[:, 0])


def test_pages_are_used_through_the_block_table():
    q, k, v, bt, sl = _setup([32, 32])
    base = paged_attention(q, k, v, bt, sl, num_splits=1)
    k2, v2 = k.clone(), v.clone()
    unused = sorted(set(range(k.shape[0])) - set(bt.flatten().tolist()))
    k2[unused], v2[unused] = 1e3, 1e3  # pages nobody owns must not matter
    torch.testing.assert_close(paged_attention(q, k2, v2, bt, sl, num_splits=1), base)
    assert gather_kv(k, bt, sl).shape == (2, 2, 32, 32)


def test_head_count_validation():
    q, k, v, bt, sl = _setup([8], hk=2, g=1)
    with pytest.raises(ValueError):
        paged_attention(q[:, :1].repeat(1, 3, 1), k, v, bt, sl)
