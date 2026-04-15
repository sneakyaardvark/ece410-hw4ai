# CF03 CMAN: DRAM TRAFFIC

1. Naive triple loop
    For each output element, one row of A and one column of B is accessed. So, elements of both B and A are accessed N=32 times. Total DRAM traffic $= 2N^3 \times 4 = 2 \times 32^3 \times 4 = 131,072$ bytes $\times 2 = 262,144$ bytes

2. Tiled
T = 8, N = 32

A: $N^2 = 32^2 \times 4 = 4096$ bytes
B: same as A

Total traffic = 8192 bytes

3. Ratio
    $262,144 / 8192 = 32$x reduction

4. Compute Time

Naive: T(mem) = 262 KB / 320e6 KB/s = 81.875 us, T(FLOP) = 65536 FLOPS / 10e12 FLOP/s = 6.5536 ns. Memory bound!

Tiled: T(mem) = 4 KB / 320e6 KB/s = 1.25 us, T(FLOP) is unchanged = 6.5536 ns. Closer to compute bound.

