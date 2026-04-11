"""Generate a roofline plot for AMD Ryzen 5 5600X with the bmm kernel marked."""

import matplotlib.pyplot as plt
import numpy as np

# System specs
peak_flops = 883.2  # GFLOP/s
peak_bw = 57.6  # GB/s
ridge_point = peak_flops / peak_bw  # ~15.3 FLOP/byte

# Dominant kernel (aten::bmm)
bmm_ai = 54.7  # FLOP/byte

# Measured performance: 9.421 GFLOP per batch, bmm took 139.9 ms over 6 batches
bmm_perf = 9.421 * 6 / 0.1399  # GFLOP/s

# Plot range
ai = np.logspace(-1, 3, 500)
roofline = np.minimum(peak_flops, peak_bw * ai)

fig, ax = plt.subplots(figsize=(8, 5))

# Roofline
ax.loglog(ai, roofline, "k-", linewidth=2, label="Roofline")

# Label the compute ceiling
ax.annotate(
    f"Peak compute: {peak_flops:.1f} GFLOP/s",
    xy=(500, peak_flops),
    fontsize=9,
    va="bottom",
)

# Label the memory-bound slope
ax.annotate(
    f"DRAM BW: {peak_bw:.1f} GB/s",
    xy=(1.5, peak_bw * 1.5),
    fontsize=9,
    rotation=38,
    va="bottom",
)

# Ridge point
ax.axvline(ridge_point, color="gray", linestyle=":", alpha=0.5)
ax.annotate(
    f"Ridge: {ridge_point:.1f} FLOP/byte",
    xy=(ridge_point, 2),
    fontsize=8,
    color="gray",
    ha="center",
)

# bmm kernel
ax.plot(bmm_ai, bmm_perf, "ro", markersize=10, zorder=5, label=f"Software kernel (AI={bmm_ai:.1f})")
ax.annotate(
    f"Software kernel \n{bmm_perf:.0f} GFLOP/s\nAI = {bmm_ai:.1f} FLOP/byte",
    xy=(bmm_ai, bmm_perf),
    xytext=(bmm_ai * 2.5, bmm_perf * 0.3),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=9,
    color="red",
)

# Target hardware accelerator (INT8, 8 MACs @ 50 MHz, 64-bit SRAM port)
hw_peak = 0.8  # GOPS
hw_bw = 0.4  # GB/s (on-chip SRAM)
hw_ridge = hw_peak / hw_bw  # 2.0 OPS/byte
# At AI=54.7, compute-bound: perf = hw_peak
ax.plot(bmm_ai, hw_peak, "bs", markersize=10, zorder=5,
        label=f"Hardware target ({hw_peak} GOPS, {hw_bw} GB/s SRAM)")
ax.annotate(
    f"Hardware target\n{hw_peak} GOPS\nridge = {hw_ridge:.1f} OPS/byte",
    xy=(bmm_ai, hw_peak),
    xytext=(bmm_ai * 2.5, hw_peak * 3),
    arrowprops=dict(arrowstyle="->", color="blue"),
    fontsize=9,
    color="blue",
)

ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
ax.set_ylabel("Performance (GFLOP/s)")
ax.set_title("Roofline — AMD Ryzen 5 5600X (FP32, dual-channel DDR4-3600)")
ax.set_xlim(0.1, 1000)
ax.set_ylim(0.1, 2000)
ax.legend()
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("roofline_project.png", dpi=150)
print(f"Ridge point: {ridge_point:.1f} FLOP/byte")
print(f"bmm AI: {bmm_ai:.1f} FLOP/byte -> compute-bound")
print(f"bmm measured: {bmm_perf:.1f} GFLOP/s ({bmm_perf / peak_flops * 100:.1f}% of peak)")
print(f"Hardware target: {hw_peak} GOPS at AI={bmm_ai}, compute-bound (ridge={hw_ridge:.1f})")
