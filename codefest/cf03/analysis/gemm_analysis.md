# GEMM CUDA Analysis

The naive kernel is memory-bound, as expected for GEMM. However, the roofline can be deceiving as the achieved points are for DRAM throughput. Both kernels are memory-bound (as shown by the theoretical kernel points), but not by the DRAM. In the naive case, the memory throughput is ~95.6% of available, but only 0.60% of the DRAM throughput is used. Most accesses (>95% for both kernels) go through the caches, which are saturating. The L1 cache has 96.13% usage for the naive case, and a slight improvement to 92.61% in the tiling version. The tiling attempts to further reduce traffic by lowering reuse, which is a negligable improvement for the DRAM (0.04%). There is little overall improvement between the naive and tiling attempts, and the tiling version requires additional computation The DRAM throughput usage is already so low that any improvement is unlikely. The memory-boundness of the caches is the real bottleneck.

## Data
### Naive
Naive GEMM 1024x1024
  Time:    0.758 ms
  GFLOP/s: 2833.99

```
[8160] gemm_naive@127.0.0.1
  gemm_naive(const float *, const float *, float *, int) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         1.21
    SM Frequency                    Ghz         1.09
    Elapsed Cycles                cycle      981,275
    Memory Throughput                 %        95.58
    DRAM Throughput                   %         0.60
    Duration                         us       896.42
    L1/TEX Cache Throughput           %        96.13
    L2 Cache Throughput               %        16.11
    SM Active Cycles              cycle   971,583.36
    Compute (SM) Throughput           %        63.68
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing L1 in the Memory Workload Analysis section.
```

### Tiled
Tiled GEMM 1024x1024 (tile=8)
  Time:    0.811 ms
  GFLOP/s: 2647.92

```
[8364] gemm_tiled@127.0.0.1
  gemm_tiled(const float *, const float *, float *, int) (128, 128, 1)x(8, 8, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         1.21
    SM Frequency                    Ghz         1.10
    Elapsed Cycles                cycle    1,075,944
    Memory Throughput                 %        91.58
    DRAM Throughput                   %         0.56
    Duration                         us       968.64
    L1/TEX Cache Throughput           %        92.61
    L2 Cache Throughput               %        33.56
    SM Active Cycles              cycle 1,055,434.05
    Compute (SM) Throughput           %        51.03
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing L1 in the Memory Workload Analysis section.
```
