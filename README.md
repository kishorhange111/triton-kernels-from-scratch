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

## Correctness

```
pytest -q        # 14 tests
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

**GPU results are pending.** They are being measured on Colab GPUs and will be added here
with the GPU model, library versions and raw JSON.

## Layout

```
src/tkern/flash_attn.py   FlashAttention-2 fwd/bwd + autograd wrapper + reference
src/tkern/norms.py        softmax, RMSNorm fwd/bwd
src/tkern/bench.py        GPU benchmark → results/bench_<gpu>.md/json
tests/                    correctness vs PyTorch (CPU interpreter or CUDA)
```
