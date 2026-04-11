# Arithmetic Intensity of Dominant Kernel (`aten::bmm`)

## Dominant kernel

From the profiler output, `aten::bmm` (batched matrix multiply) accounts for
**37.75% of self CPU time** (139.9 ms / 370.8 ms). It underlies the three
`torch.einsum` calls in the forward pass.

## Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| B      | 256   | Batch size |
| T      | 100   | Time steps |
| I      | 700   | Input units |
| H      | 200   | Hidden units |
| O      | 20    | Output classes |

All tensors are float32 (4 bytes/element).

## Per-einsum breakdown

For a matmul `(M, K) @ (K, N) -> (M, N)`:
- FLOPs = 2·M·K·N
- Bytes = (M·K + K·N + M·N) × 4  (no reuse — all operands loaded from DRAM)

| Einsum | Operation | Calls | M | K | N | FLOPs | Bytes |
|--------|-----------|-------|---|---|---|-------|-------|
| 1 — Input projection  | `(B·T, I) @ (I, H)` | 1   | 25600 | 700 | 200 | 7.168 G | 92.72 MB |
| 2 — Recurrent hidden  | `(B, H) @ (H, H)`   | 100 | 256   | 200 | 200 | 2.048 G | 56.96 MB |
| 3 — Output projection | `(B·T, H) @ (H, O)` | 1   | 25600 | 200 | 20  | 0.205 G | 22.54 MB |
| **Total** | | **102** | | | | **9.421 GFLOP** | **172.2 MB** |

Call count check: 102 calls/batch × 6 profiled batches = 612 (matches profiler).

## Arithmetic intensity

$$\text{AI} = \frac{9{,}420{,}800{,}000 \text{ FLOP}}{172{,}224{,}000 \text{ B}} \approx \boxed{54.7 \text{ FLOP/byte}}$$

This is well above typical DRAM bandwidth bottlenecks, indicating the bmm
kernel is **compute-bound**.
