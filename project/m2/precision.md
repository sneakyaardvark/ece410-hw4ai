# Precision and Numeric Format Analysis

## Weight Format: INT8

Weights are stored and used as signed 8-bit integers (INT8, range −128 to +127).
The choice is motivated by memory bandwidth: the recurrent weight matrix for a
200-neuron hidden layer contains 200 × 200 = 40,000 parameters. At INT8 each
parameter occupies one byte, giving 40 KB — a quantity that fits in a modest
on-chip SRAM without compression. Floating-point (FP32) would require 160 KB,
four times the area and power.

Multiply-accumulate operations use a 32-bit signed accumulator (`INT32`).
Accumulating 200 products of INT8 × INT8 produces values up to
127 × 127 × 200 = 3,226,600, well within the INT32 range of ±2 147 483 647,
so overflow cannot occur during a single MAC pass.

## Synaptic and Membrane Decay: Q1.15

The decay constants `alpha` (synaptic) and `beta` (membrane) are represented
in Q1.15 fixed-point format: one sign bit and 15 fractional bits.
The representable range is [−2, +2) with a resolution of 2⁻¹⁵ ≈ 0.0000305.

For LIF neurons the decay coefficients are biological rates in (0, 1), so the
full range [−2, 2) is overkill, but Q1.15 is the natural 16-bit fixed-point
format supported by many DSP multiply instructions (the `>> 15` shift maps
directly to a signed arithmetic right-shift). The default values are:

- `alpha = 19875 / 32768 ≈ 0.6063` (Q1.15 → roughly 60 % synaptic retention per step)
- `beta  = 29635 / 32768 ≈ 0.9043` (Q1.15 → roughly 90 % membrane retention per step)

Multiplication is implemented as `(coeff × value) >> 15`. The intermediate
product occupies up to 31 bits (16-bit coefficient × 32-bit state), so the
hardware accumulator must be at least 48 bits to avoid overflow mid-multiply.
In `compute_core.sv` the LIF state registers (`syn`, `mem`) are INT32; the
multiply in `snn_lif_cell.sv` promotes both operands to 64 bits before
shifting, avoiding any intermediate overflow.

## Threshold Format: INT32 Signed

The spike threshold is a signed 32-bit integer compared directly against the
membrane potential register (also INT32). Using the same format for both
eliminates any format-conversion step at the comparator. The default threshold
of 32768 (= 2¹⁵) was chosen to be reachable by a neuron receiving moderate
positive synaptic input over several timesteps, ensuring neurons can spike in
the reference test cases.

## Quantization Error Analysis

The dominant quantization error source is the INT8 weight representation.
The maximum rounding error per weight is ±0.5 LSB, or ±0.5 on the integer
scale. After accumulating 200 such errors, the worst-case error in a single
MAC result is ±100 (assuming errors are correlated, which they are not in
practice). In the random-weight inference test (`test_full_inference_random`)
with threshold = 100, a raw accumulator error of ±100 could flip a borderline
neuron. The test passes reproducibly because the random seed is fixed
(`rng = np.random.default_rng(42)`) and the reference model uses the same
INT8 representation — both model and hardware quantize identically, so
mismatches due to weight quantization cancel exactly.

The Q1.15 decay multiplication introduces an additional rounding error of at
most 1 LSB (1/32768 ≈ 3 × 10⁻⁵) per multiply. Over NB_STEPS = 5 timesteps
this accumulates to at most 5 LSB ≈ 0.00015 on the INT32 membrane scale —
negligible relative to the threshold of 32768.

The practical implication is that the INT8/Q1.15 scheme is accurate enough
for the spiking workload tested here, while offering a 4× weight-memory
reduction and the ability to implement every MAC with a single 8-bit × 8-bit
multiplier.
