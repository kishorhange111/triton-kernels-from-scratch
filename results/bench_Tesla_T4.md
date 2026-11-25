GPU: **Tesla T4**, torch 2.14.1+cu130, triton 3.8.0, fp16, B=4 H=16 D=64.

### Attention, causal=False, fwd

| impl (fwd, TFLOP/s) | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| triton 64x64 | 0.4 | 0.5 | 0.5 | 0.5 | 0.5 |
| triton 128x64 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| triton 64x32 | 0.5 | 1.3 | 1.2 | 1.2 | 1.1 |
| triton 128x128 | OutOfResources | OutOfResources | OutOfResources | OutOfResources | OutOfResources |
| torch sdpa | 6.3 | 14.5 | 13.1 | 12.7 | 12.2 |
| naive eager | 1.7 | 1.8 | 1.7 | — | — |

### Attention, causal=False, fwd+bwd

| impl (fwd+bwd, TFLOP/s) | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| triton 64x64 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| triton 128x64 | OutOfResources | OutOfResources | OutOfResources | OutOfResources | OutOfResources |
| triton 64x32 | 0.5 | 0.7 | 0.7 | 0.7 | 0.7 |
| triton 128x128 | OutOfResources | OutOfResources | OutOfResources | OutOfResources | OutOfResources |
| torch sdpa | 5.1 | 9.9 | 10.4 | 10.3 | 9.6 |
| naive eager | 2.1 | 2.2 | 2.1 | — | — |

### Attention, causal=True, fwd

| impl (fwd, TFLOP/s) | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| triton 64x64 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| triton 128x64 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| triton 64x32 | 0.4 | 1.1 | 1.1 | 1.1 | 1.0 |
| triton 128x128 | OutOfResources | OutOfResources | OutOfResources | OutOfResources | OutOfResources |
| torch sdpa | 5.0 | 12.5 | 11.8 | 11.7 | 11.9 |
| naive eager | 0.6 | 0.6 | 0.6 | — | — |

### Attention, causal=True, fwd+bwd

| impl (fwd+bwd, TFLOP/s) | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| triton 64x64 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| triton 128x64 | OutOfResources | OutOfResources | OutOfResources | OutOfResources | OutOfResources |
| triton 64x32 | 0.3 | 0.6 | 0.7 | 0.7 | 0.6 |
| triton 128x128 | OutOfResources | OutOfResources | OutOfResources | OutOfResources | OutOfResources |
| torch sdpa | 3.6 | 7.3 | 8.5 | 8.8 | 8.9 |
| naive eager | 0.8 | 0.8 | 0.8 | — | — |

### Row kernels (4096 rows, fp16), effective GB/s

| cols | softmax triton | softmax torch | rmsnorm triton | rmsnorm eager |
|---|---|---|---|---|
| 1024 | 221 | 206 | 220 | 23 |
| 4096 | 229 | 149 | 226 | 24 |
| 8192 | 230 | 165 | 229 | 24 |
| 16384 | 233 | 215 | 233 | 24 |
