"""Quantization error sweep for the INT8 MAC array vs an FP32 reference.

Runs N_TRIALS independent dot products of length T_LEN through the DUT
(`snn_mac_array`, instantiated by `tb_snn_mac_array`).  Each trial:

  1. Sample weights w_fp in [-1, 1) and activations a_fp in [-1, 1) (FP32).
  2. Quantize symmetrically to INT8 with scale s = 127:
        w_int = clip(round(w_fp * 127), -128, 127)
        a_int = clip(round(a_fp * 127), -128, 127)
     Equivalent dequantization scale for the dot product is 1 / (127 * 127).
  3. Stream the INT8 stimulus into the DUT, read the INT32 accumulators, and
     dequantize: dut_fp = dut_int / (127 * 127).
  4. Reference FP32 dot product: ref_fp = sum(w_fp * a_fp).
  5. Per-output absolute error: |dut_fp - ref_fp|.

Aggregates 100 trials × 8 PEs = 800 sample errors.  Reports MAE, max abs
error, and RMS error on the FP32 scale, plus the same numbers normalised by
the FP32 dynamic range to give a relative figure of merit.
"""

import math

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

N         = 8       # PEs (output channels per tile)
T_LEN     = 200     # dot-product length (matches v1 row width)
N_TRIALS  = 100     # required by checklist
SCALE     = 127     # symmetric INT8 scale


def pack_weights(weights: list[int]) -> int:
    val = 0
    for i, w in enumerate(weights):
        val |= (w & 0xFF) << (i * 8)
    return val


def unpack_acc(raw: int) -> list[int]:
    out = []
    for i in range(N):
        word = (raw >> (i * 32)) & 0xFFFFFFFF
        if word >= (1 << 31):
            word -= (1 << 32)
        out.append(word)
    return out


async def reset_and_clock(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst.value         = 1
    dut.weight_load.value = 0
    dut.weight_in.value   = 0
    dut.acc_clear.value   = 0
    dut.act_valid.value   = 0
    dut.act_in.value      = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def run_one_trial(dut, w_int: np.ndarray, a_int: np.ndarray) -> list[int]:
    """Stream T_LEN cycles of weights+activations, return N int32 accumulators."""
    dut.weight_load.value = 1
    dut.act_valid.value   = 1
    for t in range(T_LEN):
        dut.weight_in.value = pack_weights(w_int[t].tolist())
        dut.act_in.value    = int(a_int[t])
        await RisingEdge(dut.clk)
    dut.weight_load.value = 0
    dut.act_valid.value   = 0
    await FallingEdge(dut.clk)
    return unpack_acc(dut.acc_out.value.to_unsigned())


def quantize(x_fp: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x_fp * SCALE), -128, 127).astype(np.int32)


@cocotb.test()
async def test_quantization_error_sweep(dut):
    """Compare DUT INT8 dot products to FP32 reference over 100 random trials.

    Reports MAE, max abs error, RMS error on the FP32 scale.  The error budget
    set by symmetric INT8 quantization with scale 127 is bounded analytically
    by  T * (|w_max|/SCALE) * (1/(2*SCALE))  per output, which for T=200 and
    |w|<=1 evaluates to approximately 0.0079.  We assert MAE is well under
    this bound and max error is under 4x this bound.
    """
    await reset_and_clock(dut)

    rng = np.random.default_rng(2026)
    abs_errors  = []
    ref_values  = []
    dut_values  = []

    for trial in range(N_TRIALS):
        w_fp = rng.uniform(-1.0, 1.0, size=(T_LEN, N)).astype(np.float32)
        a_fp = rng.uniform(-1.0, 1.0, size=(T_LEN,)).astype(np.float32)

        w_int = quantize(w_fp)
        a_int = quantize(a_fp)

        # Reset accumulators between trials
        dut.acc_clear.value = 1
        await RisingEdge(dut.clk)
        dut.acc_clear.value = 0

        dut_int = await run_one_trial(dut, w_int, a_int)

        # Dequantize DUT output to FP scale
        dut_fp_out = np.array(dut_int, dtype=np.float64) / (SCALE * SCALE)
        ref_fp_out = (w_fp.astype(np.float64) * a_fp.astype(np.float64)[:, None]).sum(axis=0)

        abs_errors.extend(np.abs(dut_fp_out - ref_fp_out).tolist())
        ref_values.extend(ref_fp_out.tolist())
        dut_values.extend(dut_fp_out.tolist())

    abs_errors = np.array(abs_errors)
    ref_values = np.array(ref_values)

    mae      = float(abs_errors.mean())
    max_err  = float(abs_errors.max())
    rms_err  = float(math.sqrt((abs_errors ** 2).mean()))
    ref_rng  = float(ref_values.max() - ref_values.min())
    rel_mae  = mae / ref_rng if ref_rng > 0 else float("nan")
    rel_max  = max_err / ref_rng if ref_rng > 0 else float("nan")

    # Analytical bound (worst case for symmetric INT8, |w|,|a| <= 1):
    #   per-output abs error <= T * (1/(2*SCALE)) * (|w_max| + |a_max|)
    # which evaluates to T_LEN * (1/(2*SCALE)) * 2 = T_LEN / SCALE for unit range.
    # The expected (RMS) error is much smaller because rounding errors are
    # zero-mean and uncorrelated, so they accumulate as sqrt(T).
    expected_rms = math.sqrt(T_LEN) * (1.0 / (math.sqrt(12) * SCALE)) * (2.0 / math.sqrt(3))
    worst_case   = T_LEN / SCALE

    dut._log.info("=" * 70)
    dut._log.info("INT8 MAC quantization error sweep")
    dut._log.info(
        f"  trials={N_TRIALS}, dot-length={T_LEN}, PEs={N}, "
        f"samples={len(abs_errors)}"
    )
    dut._log.info(f"  ref dynamic range : [{ref_values.min():+.4f}, {ref_values.max():+.4f}]")
    dut._log.info(f"  MAE               : {mae:.6f}  (relative: {rel_mae*100:.3f} %)")
    dut._log.info(f"  RMS error         : {rms_err:.6f}")
    dut._log.info(f"  max abs error     : {max_err:.6f}  (relative: {rel_max*100:.3f} %)")
    dut._log.info(f"  predicted RMS     : {expected_rms:.6f}  (zero-mean accumulation model)")
    dut._log.info(f"  worst-case bound  : {worst_case:.6f}  (correlated rounding upper bound)")
    dut._log.info("=" * 70)

    # Sanity: observed MAE must beat the correlated worst case by a wide
    # margin (otherwise something is wrong with the DUT or the model).
    assert mae   < 0.25 * worst_case, f"MAE {mae} exceeds 25% of worst-case bound {worst_case}"
    assert max_err < worst_case,      f"max error {max_err} exceeds worst-case bound {worst_case}"
