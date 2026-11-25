Paged-attention decode, fp16, H=32 query heads / 8 KV heads (GQA), D=128, page=16. GB/s = bytes of K and V read / time; HBM peak ≈ 320 GB/s.

| batch × context | paged, split-K (auto) | paged, 1 split | SDPA, contiguous cache |
|---|---|---|---|
| 32 × 1,024 | 57 | 73 | 89 |
| 8 × 4,096 | 96 | 93 | 92 |
| 8 × 16,384 | 62 | 83 | 87 |
| 1 × 32,768 | 97 | 48 | 66 |
