# triton-kernels-from-scratch

[![CI](https://github.com/kishorhange111/triton-kernels-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/triton-kernels-from-scratch/actions/workflows/ci.yml)

GPU kernels written from scratch in [Triton](https://triton-lang.org), each verified against a
PyTorch reference (values **and gradients**) and benchmarked on real GPUs:

| kernel | file | what it shows |
|---|---|---|
| **FlashAttention-2**, forward + backward, causal or not | [`flash_attn.py`](src/tkern/flash_attn.py) | online softmax, tiling through SRAM, log-sum-exp recomputation instead of storing N×N probabilities, a two-kernel backward with no atomics |
| fused **softmax** | [`norms.py`](src/tkern/norms.py) | numerically safe row reduction; memory-bound, so the target is bandwidth |
| fused **RMSNorm**, forward + backward | [`norms.py`](src/tkern/norms.py) | the full dx derivation in-kernel, dW as a cross-row reduction |

## How the attention kernel works

For each query block `Q_i` (one Triton program), stream key/value blocks `K_j, V_j` and keep
three running quantities per row: the max score `m`, the normaliser `l` and the unnormalised
output `acc`:

```
s     = Q_i K_jᵀ · scale·log2(e)          # base-2 scores: exp2 is cheaper than exp
m_new = max(m, rowmax(s))
p     = exp2(s − m_new)
l     = l · exp2(m − m_new) + rowsum(p)    # rescale the old sum to the new max
acc   = acc · exp2(m − m_new) + p V_j
```

Output is `acc / l`, and the kernel saves only `L = m + log2(l)` per row. The backward pass
recomputes `P = exp2(s − L)` block by block and uses `dS = P ∘ (dP − Δ)` with
`Δ = rowsum(dO ∘ O)`. That costs extra FLOPs but keeps memory traffic linear in N, the trade
the FlashAttention papers make. dK/dV are accumulated per key block and dQ per query block in
separate kernels, so no atomics are needed. With causal masking, blocks entirely above the
diagonal are skipped, not just masked.

## Paged-attention decode ([`paged_attn.py`](src/tkern/paged_attn.py))

This is the decode kernel inside vLLM-style serving.
- **Paged KV cache.** The cache is a pool of 16-token pages. Each sequence has a block table
  listing its pages, so sequences grow without reallocation and freed pages are reused
  anywhere.
- **Grouped-query attention.** G query heads read the same KV head.
- **Flash-decoding split-K.** With a small batch of long sequences, one program per
  (sequence, head) leaves most SMs idle. Each sequence's pages are therefore split over S
  programs, each keeping an online-softmax `(m, l, acc)`. A second kernel rescales them to a
  common max and merges them: `o = Σ 2^(mₛ−M)·accₛ / Σ 2^(mₛ−M)·lₛ`.

One query against a page is a matrix-vector product, so decode attention is bound by memory
bandwidth: every cached byte is read once. The kernel uses multiply-and-reduce rather than
`tl.dot`. The tests use page tables in random order, ragged lengths (1 token, page boundary
+ 1, partial and exact pages), more splits than pages, and GQA head sharing. They also check
that pages no sequence owns are never read. `python -m tkern.bench --paged` reports
effective GB/s against SDPA on a contiguous cache, with and without split-K.

**On a T4** ([`results/paged_Tesla_T4.md`](results/paged_Tesla_T4.md); fp16, 32 query /
8 KV heads, d = 128, 16-token pages; GB/s = K and V bytes read / time, peak ≈ 320):

| batch × context | paged, split-K (auto) | paged, 1 split | SDPA, contiguous cache |
|---|---|---|---|
| 32 × 1,024 | 57 | 73 | 89 |
| 8 × 4,096 | 96 | 93 | 92 |
| 8 × 16,384 | 62 | 83 | 87 |
| 1 × 32,768 | **97** | 48 | 66 |

- **Split-K pays off where it is meant to.** With one long sequence, one program per
  (sequence, head) occupies 32 programs on 40 SMs. Splitting the pages doubles bandwidth
  (48 → 97 GB/s) and beats SDPA on a contiguous cache by 1.5×, despite the block-table
  indirection.
- **It hurts when the batch already fills the GPU.** At 32 × 1,024 and 8 × 16,384 the
  automatic split count adds programs and a combine pass that aren't needed. A single
  split is faster there, so the heuristic should consider the work per program, not only
  the SM count.
- **Paging costs little.** With a single split, the paged kernel is within 10–20% of
  SDPA's contiguous reads, and level at 8 × 4,096. Every configuration stays at or below 30%
  of HBM peak. As in the attention diagnosis above, this kernel runs on CUDA-core FMAs on
  sm75; it does not saturate memory.

## Correctness

```
pytest -q        # 21 tests
```

The tests compare outputs and all gradients with PyTorch. They cover causal and non-causal
attention, sequence lengths that aren't block multiples (37), rectangular tiles (32×16), and
logits large enough to overflow a naive `exp`. On a machine without a GPU they run the
**same kernel source** in Triton's interpreter (`TRITON_INTERPRET=1`), which is how CI
checks them.

## Benchmarks

`python -m tkern.bench --out results` times the kernels against PyTorch SDPA (its own
flash and memory-efficient kernels) and naive eager attention. It reports TFLOP/s for
attention (forward, and forward+backward) and effective GB/s for the row kernels, and it
sweeps tile shapes, because the best tile depends on the GPU's shared memory.

### Tesla T4 (sm75): torch 2.14.1+cu130, triton 3.8.0, fp16

These are the full tables: [`results/bench_Tesla_T4.md`](results/bench_Tesla_T4.md), with
raw data in [`results/bench_Tesla_T4.json`](results/bench_Tesla_T4.json). All 14 tests also
pass on the GPU ([`results/tkern_gpu_tests.txt`](results/tkern_gpu_tests.txt)).

**Row kernels win clearly.** Effective bandwidth, 4,096 rows:

| cols | softmax: Triton | softmax: torch | RMSNorm: Triton | RMSNorm: eager |
|---|---|---|---|---|
| 1,024 | 221 GB/s | 206 GB/s | 220 GB/s | 23 GB/s |
| 4,096 | 229 GB/s | 149 GB/s | 226 GB/s | 24 GB/s |
| 16,384 | 233 GB/s | 215 GB/s | 233 GB/s | 24 GB/s |

The fused kernels run at about 230 GB/s, roughly 75% of the T4's 320 GB/s. That is the
practical ceiling for a one-read, one-write kernel. Eager RMSNorm launches several
elementwise and reduction kernels, each a full round trip to HBM, and is about 9× slower.

**Attention loses, and the diagnosis explains why.** Forward pass, non-causal, N = 4,096:
the best Triton tile reaches 1.2 TFLOP/s against 12.7 for PyTorch SDPA and 1.7 for naive
eager attention at N = 2,048. `python -m tkern.bench --diagnose` inspects the compiled PTX
and sweeps launch parameters
([`results/diagnose_Tesla_T4.md`](results/diagnose_Tesla_T4.md)):

- **The PTX contains no tensor-core (`mma`) instructions.** It does contain 2,052 scalar
  `fma.rn.f32` instructions. On this GPU, with this Triton build, `tl.dot` on fp16 tiles is
  lowered to CUDA-core FMAs, so the kernel is capped near the T4's 8 TFLOP/s fp32 peak, not
  its 65 TFLOP/s fp16 tensor-core peak. It reaches 1.35 TFLOP/s. The kernel source is not
  the cause: the same operands are fp16 and the accumulators fp32, which is the
  tensor-core-eligible pattern. Why Triton picks this lowering on sm75 is still an open
  question here.
- **Pipelining doesn't matter, but warps do.** Across 45 configurations, `num_stages`
  (1–3) changes throughput by under 2%. That fits a compute-bound FMA loop with nothing
  to hide behind async copies. `num_warps` matters a lot: 64×64 tiles run at 0.14 TFLOP/s
  with 2 warps and 1.27 with 8. The best configuration is 64×32 with 8 warps (1.35).
  128×128 tiles exceed the T4's 64 KB of shared memory (`OutOfResources`).
- **The causal 64×64 kernel is slower than non-causal (0.1 vs 0.5 TFLOP/s),** even though
  it skips half the blocks. A likely cause, not yet profiled: with the default 4 warps, the diagonal-block masking adds a
  scalar `where` over each tile to a loop that is already ALU-bound. 64×32 doesn't show the
  drop (1.1 causal vs 1.2).

This also explains a number from
[llm-capacity-planner](https://github.com/kishorhange111/llm-capacity-planner)'s vLLM
calibration on the same GPU. vLLM had selected its Triton attention backend there, and the
fitted attention efficiency came out at about 1% of peak, against 27% for the GEMMs.

The next step is to rerun the same diagnosis on an sm80+ GPU (A100, L4 or H100), where Triton's
tensor-core path is the documented default. That waits on Colab granting one.

## Layout

```
src/tkern/flash_attn.py   FlashAttention-2 fwd/bwd + autograd wrapper + reference
src/tkern/norms.py        softmax, RMSNorm fwd/bwd
src/tkern/paged_attn.py   paged-attention decode: block tables, GQA, split-K + combine
src/tkern/bench.py        GPU benchmark → results/bench_<gpu>.md/json
tests/                    correctness vs PyTorch (CPU interpreter or CUDA)
```
