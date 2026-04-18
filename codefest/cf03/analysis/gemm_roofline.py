"""Roofline plot for naive vs. tiled GEMM on NVIDIA A100."""

import numpy as np
import matplotlib.pyplot as plt

# A100 SXM4-40GB peak specs
PEAK_FP32_GFLOPS = 19500.0   # 19.5 TFLOP/s, FP32 CUDA cores (no tensor cores)
PEAK_BW_GBPS = 1555.0        # HBM2 bandwidth

# Problem size
N = 1024
TILE = 8
TOTAL_FLOPS = 2.0 * N * N * N      # 2 * N^3

# Theoretical (algorithmic) AI — assumes every read hits DRAM (no cache reuse)
#   Naive:      AI = 0.25 FLOP/B
#   Tiled (T):  AI = T/4 FLOP/B
ai_naive_theory = 0.25
ai_tiled_theory = TILE / 4.0

# Theoretical ceiling = performance the roofline model predicts at theoretical AI
ceil_naive = min(PEAK_BW_GBPS * ai_naive_theory, PEAK_FP32_GFLOPS)  # 389 GF/s
ceil_tiled = min(PEAK_BW_GBPS * ai_tiled_theory, PEAK_FP32_GFLOPS)  # 3110 GF/s

# Measured achieved performance (from gemm_analysis.md)
gflops_naive = 2833.99
gflops_tiled = 2647.92
dur_naive_s  = 896.42e-6
dur_tiled_s  = 968.64e-6
dram_pct_naive = 0.60 / 100.0
dram_pct_tiled = 0.56 / 100.0

# Effective AI from measured DRAM traffic (where the kernel actually sits)
dram_bytes_naive = dram_pct_naive * PEAK_BW_GBPS * 1e9 * dur_naive_s
dram_bytes_tiled = dram_pct_tiled * PEAK_BW_GBPS * 1e9 * dur_tiled_s
ai_naive_eff = TOTAL_FLOPS / dram_bytes_naive
ai_tiled_eff = TOTAL_FLOPS / dram_bytes_tiled

# --- Build the roofline ---
ai = np.logspace(-2, 3, 500)
roof = np.minimum(PEAK_BW_GBPS * ai, PEAK_FP32_GFLOPS)
ridge = PEAK_FP32_GFLOPS / PEAK_BW_GBPS

fig, ax = plt.subplots(figsize=(9, 6))
ax.loglog(ai, roof, 'k-', linewidth=2.2, label='A100 DRAM Roofline')
ax.axvline(ridge, color='gray', linestyle=':', alpha=0.5)

# Shade regimes at the ridge
ax.axvspan(1e-2, ridge, alpha=0.07, color='red')
ax.axvspan(ridge, 1e3, alpha=0.07, color='green')

# Theoretical ceilings: mark where (theoretical AI, predicted max) lies on the roof
ax.plot(ai_naive_theory, ceil_naive, 'v', color='C0', markersize=11,
        markerfacecolor='none', markeredgewidth=2,
        label=f'Naive theoretical ceiling: {ceil_naive:.0f} GF/s @ AI={ai_naive_theory:.2f}')
ax.plot(ai_tiled_theory, ceil_tiled, 'v', color='C1', markersize=11,
        markerfacecolor='none', markeredgewidth=2,
        label=f'Tiled theoretical ceiling: {ceil_tiled:.0f} GF/s @ AI={ai_tiled_theory:.2f}')

# Achieved (measured) points at effective AI
ax.plot(ai_naive_eff, gflops_naive, 'o', color='C0', markersize=13,
        label=f'Naive achieved: {gflops_naive:.0f} GF/s @ AI={ai_naive_eff:.0f}')
ax.plot(ai_tiled_eff, gflops_tiled, 's', color='C1', markersize=13,
        label=f'Tiled achieved: {gflops_tiled:.0f} GF/s @ AI={ai_tiled_eff:.0f}')

# Region labels (below the respective roofs)
ax.text(0.05, 20, 'Memory bound', fontsize=12, color='darkred',
        alpha=0.8, weight='bold')
ax.text(150, 6000, 'Compute bound', fontsize=12, color='darkgreen',
        alpha=0.8, weight='bold')

# Roof annotations
ax.text(60, PEAK_FP32_GFLOPS * 1.15,
        f'Peak FP32: {PEAK_FP32_GFLOPS/1000:.1f} TFLOP/s', fontsize=10)
ax.text(0.015, 80, f'HBM2: {PEAK_BW_GBPS:.0f} GB/s',
        fontsize=10, rotation=38)
ax.text(ridge * 1.15, 15, f'ridge AI = {ridge:.1f}', fontsize=9, color='gray')

ax.set_xlabel('Arithmetic Intensity (FLOP / byte)')
ax.set_ylabel('Performance (GFLOP/s)')
ax.set_title('Roofline: 1024x1024 FP32 GEMM on NVIDIA A100')
ax.set_xlim(1e-2, 1e3)
ax.set_ylim(1e1, 1e5)
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('gemm_roofline.png', dpi=150)
print('Saved gemm_roofline.png')

# --- Summary ---
print()
print(f'{"Kernel":<8} {"Achieved GF/s":>14} {"AI theory":>11} {"Ceil (GF/s)":>12} '
      f'{"AI eff":>8} {"Regime":>16}')
for name, gf, ai_t, ceil, ai_e, mem_pct, comp_pct in [
    ('Naive', gflops_naive, ai_naive_theory, ceil_naive, ai_naive_eff, 95.58, 63.68),
    ('Tiled', gflops_tiled, ai_tiled_theory, ceil_tiled, ai_tiled_eff, 91.58, 51.03),
]:
    regime = 'memory-bound' if mem_pct > comp_pct else 'compute-bound'
    print(f'{name:<8} {gf:>14.1f} {ai_t:>11.2f} {ceil:>12.0f} {ai_e:>8.1f} {regime:>16}')
