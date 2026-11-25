"""GPU benchmarks. ``python -m tkern.bench --out results`` (needs CUDA).

Attention: forward and forward+backward time for the Triton kernel, PyTorch SDPA (which
dispatches to its own flash / memory-efficient kernels) and naive eager attention, in
TFLOP/s. FLOPs counted as 4·B·H·N²·D forward (halved when causal) and 2.5x that for the
backward, the FlashAttention-2 paper's convention.

Row kernels: effective bandwidth (bytes read + written / time) for softmax and RMSNorm
against eager PyTorch. Results go to markdown tables plus raw JSON.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
import triton

from .flash_attn import flash_attention, reference_attention
from .norms import reference_rms_norm, rms_norm, softmax


def _ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def _attn_flops(b, h, n, d, causal, mode):
    f = 4 * b * h * n * n * d * (0.5 if causal else 1.0)
    return f * (3.5 if mode == "fwd+bwd" else 1.0)


def _attention_impls(q, k, v, causal, blocks, naive):
    def tri(bm, bn):
        return lambda: flash_attention(q, k, v, causal, None, bm, bn)

    impls = {f"triton {bm}x{bn}": tri(bm, bn) for bm, bn in blocks}
    impls["torch sdpa"] = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if naive:  # materialises the N x N matrix, so only at sizes that fit
        impls["naive eager"] = lambda: reference_attention(q, k, v, causal)
    return impls


def bench_attention(n_list, b, h, d, causal, blocks):
    rows = []
    dt = torch.float16
    for n in n_list:
        q, k, v = [
            torch.randn(b, h, n, d, device="cuda", dtype=dt, requires_grad=True) for _ in range(3)
        ]
        do = torch.randn_like(q)
        impls = _attention_impls(q, k, v, causal, blocks, naive=b * h * n * n * 4 < 2e9)
        for name, fn in impls.items():
            for mode in ("fwd", "fwd+bwd"):
                try:
                    if mode == "fwd":
                        with torch.no_grad():
                            ms = _ms(fn)
                    else:
                        ms = _ms(lambda fn=fn: fn().backward(do))
                except Exception as e:  # e.g. shared-memory overflow for a block shape
                    rows.append({"n": n, "impl": name, "mode": mode, "error": type(e).__name__})
                    continue
                tf = _attn_flops(b, h, n, d, causal, mode) / ms / 1e9
                rows.append({"n": n, "impl": name, "mode": mode, "ms": ms, "tflops": tf})
                print(rows[-1], flush=True)
        q = k = v = do = impls = None  # free before the next size
        torch.cuda.empty_cache()
    return rows


def bench_rows(m, cols_list):
    rows = []
    for cols in cols_list:
        x = torch.randn(m, cols, device="cuda", dtype=torch.float16)
        w = torch.randn(cols, device="cuda", dtype=torch.float16)
        nbytes = 2 * x.numel() * x.element_size()  # read x once, write y once
        for name, fn in {
            "softmax triton": lambda: softmax(x),
            "softmax torch": lambda: torch.softmax(x, -1),
            "rmsnorm triton": lambda: rms_norm(x, w),
            "rmsnorm eager": lambda: reference_rms_norm(x, w),
        }.items():
            with torch.no_grad():
                ms = _ms(fn)
            rows.append({"cols": cols, "impl": name, "ms": ms, "gbps": nbytes / ms / 1e6})
            print(rows[-1], flush=True)
    return rows


def _attn_table(rows, mode):
    ns = sorted({r["n"] for r in rows})
    impls = list(dict.fromkeys(r["impl"] for r in rows))
    out = [f"| impl ({mode}, TFLOP/s) | " + " | ".join(f"N={n}" for n in ns) + " |",
           "|---|" + "---|" * len(ns)]  # fmt: skip
    for i in impls:
        cells = []
        for n in ns:
            r = next(
                (r for r in rows if r["impl"] == i and r["n"] == n and r["mode"] == mode), None
            )
            cells.append("—" if r is None else r.get("error") or f"{r['tflops']:.1f}")
        out.append(f"| {i} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    gpu = torch.cuda.get_device_name()
    blocks = [(64, 64), (128, 64), (64, 32), (128, 128)]
    report = [f"GPU: **{gpu}**, torch {torch.__version__}, triton {triton.__version__}, fp16, "
              f"B={a.batch} H={a.heads} D={a.head_dim}.\n"]  # fmt: skip
    blob = {"gpu": gpu}
    for causal in (False, True):
        rows = bench_attention(a.seqlens, a.batch, a.heads, a.head_dim, causal, blocks)
        blob[f"attention_causal={causal}"] = rows
        for mode in ("fwd", "fwd+bwd"):
            report += [f"### Attention, causal={causal}, {mode}\n", _attn_table(rows, mode), ""]
    rows = bench_rows(4096, [1024, 4096, 8192, 16384])
    blob["rows"] = rows
    report += ["### Row kernels (4096 rows, fp16), effective GB/s\n",
               "| cols | " + " | ".join(dict.fromkeys(r["impl"] for r in rows)) + " |",
               "|---|---|---|---|---|"]  # fmt: skip
    for c in sorted({r["cols"] for r in rows}):
        report.append(
            f"| {c} | " + " | ".join(f"{r['gbps']:.0f}" for r in rows if r["cols"] == c) + " |"
        )
    tag = gpu.replace(" ", "_")
    with open(os.path.join(a.out, f"bench_{tag}.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(a.out, f"bench_{tag}.json"), "w") as f:
        json.dump(blob, f, indent=1)
    print("\n".join(report))


if __name__ == "__main__":
    main()
