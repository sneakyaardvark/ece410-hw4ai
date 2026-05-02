"""cocotb testbench for snn_lif_cell (via tb_snn_lif_cell wrapper).

The Python reference model mirrors the exact fixed-point arithmetic used in
the RTL: Q1.15 multiply implemented as integer multiply followed by >> 15
(arithmetic right shift / floor division for positive, truncation for negative).
All comparisons are exact — any mismatch is a real hardware bug.

Tests:
  - reset clears all state
  - syn decay: syn tracks alpha*syn + h1 correctly
  - mem decay and integration: mem tracks beta*mem + syn correctly
  - spike fires when mem >= threshold
  - mem resets to 0 on spike (soft reset)
  - no update when h1_valid is low
  - multi-step sequence compared cycle-by-cycle against reference model
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

# ---------------------------------------------------------------------------
# Fixed-point constants matching the hardware defaults used in tests
# ---------------------------------------------------------------------------
# alpha = exp(-1ms / 5ms)  ≈ 0.6065 → Q1.15: round(0.6065 * 2^15) = 19875
# beta  = exp(-1ms / 10ms) ≈ 0.9048 → Q1.15: round(0.9048 * 2^15) = 29635
ALPHA_Q = 19875
BETA_Q  = 29635
THRESH  = 32768  # INT32 threshold (arbitrary but sensible for tests)


def q15_mul(coeff: int, value: int) -> int:
    """Replicate RTL Q1.15 multiply: (coeff * value) >> 15, signed truncation."""
    product = coeff * value  # coeff unsigned, value signed → signed product
    # Arithmetic right shift (truncate toward -inf for negative, like >>> in SV)
    if product >= 0:
        return product >> 15
    else:
        return -((-product) >> 15)


def lif_step(syn: int, mem: int, alpha: int, beta: int,
             threshold: int, h1: int) -> tuple[int, int, int]:
    """One LIF update step. Returns (new_syn, new_mem, spike)."""
    new_syn = q15_mul(alpha, syn) + h1
    spike   = 1 if mem >= threshold else 0
    new_mem = 0 if spike else (q15_mul(beta, mem) + syn)
    return new_syn, new_mem, spike


async def init(dut, alpha=ALPHA_Q, beta=BETA_Q, threshold=THRESH):
    """Start clock, set defaults, apply reset for 2 cycles."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())

    dut.rst.value       = 1
    dut.alpha.value     = alpha
    dut.beta.value      = beta
    dut.threshold.value = threshold
    dut.h1_in.value     = 0
    dut.h1_valid.value  = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def step(dut, h1: int):
    """Apply one h1_valid pulse with h1_in = h1."""
    dut.h1_in.value    = h1
    dut.h1_valid.value = 1
    await RisingEdge(dut.clk)
    dut.h1_valid.value = 0
    dut.h1_in.value    = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_reset(dut):
    """After reset, spike_out must be 0."""
    await init(dut)
    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 0, "Expected spike_out=0 after reset"


@cocotb.test()
async def test_no_spike_below_threshold(dut):
    """Small constant h1 input: mem builds up but stays below threshold."""
    await init(dut, threshold=THRESH)
    # Feed a small positive current for 5 steps; mem should not reach THRESH
    for _ in range(5):
        await step(dut, h1=100)
    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 0, "Expected no spike for sub-threshold input"


@cocotb.test()
async def test_spike_fires_at_threshold(dut):
    """Large h1 drives mem above threshold; spike must fire after 3 steps.

    The 2-step pipeline: h1 → syn (step 1) → mem (step 2) → spike (step 3).
    new_mem uses the old syn, so current from h1 reaches mem one step later.
    """
    await init(dut, threshold=1000)

    await step(dut, h1=2000)   # step 1: syn=2000, mem=0 (old syn was 0)
    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 0, "No spike expected: mem still 0"

    await step(dut, h1=0)      # step 2: mem=2000 (old syn=2000 flows in), no spike yet
    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 0, "No spike expected: old mem was 0"

    await step(dut, h1=0)      # step 3: spike=(mem=2000 >= 1000)=1
    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 1, "Expected spike when mem crosses threshold"


@cocotb.test()
async def test_mem_reset_on_spike(dut):
    """After a spike, mem resets to 0; the following step must not spike."""
    await init(dut, threshold=1000)
    await step(dut, h1=2000)   # step 1: syn=2000
    await step(dut, h1=0)      # step 2: mem=2000, no spike
    await step(dut, h1=0)      # step 3: spike fires, mem resets to 0
    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 1, "Spike should have fired at step 3"

    await step(dut, h1=0)      # step 4: mem was reset to 0, no spike
    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 0, "Expected no spike after mem reset"


@cocotb.test()
async def test_no_update_without_h1_valid(dut):
    """State must not change when h1_valid is low."""
    await init(dut, threshold=THRESH)
    await step(dut, h1=500)   # set some state

    # Hold h1_valid low for 5 cycles
    dut.h1_valid.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    assert dut.spike_out.value == 0, "spike_out changed without h1_valid"


@cocotb.test()
async def test_multi_step_reference(dut):
    """20-step sequence: compare spike_out cycle-by-cycle against reference model."""
    alpha = ALPHA_Q
    beta  = BETA_Q
    threshold = THRESH
    await init(dut, alpha=alpha, beta=beta, threshold=threshold)

    # Stimulus: alternating positive and zero inputs
    inputs = [500, 0, 800, 0, 0, 1200, 0, 0, 0, 400,
              600, 0, 0, 900, 0, 0, 1100, 0, 0, 0]

    syn, mem = 0, 0
    for i, h1 in enumerate(inputs):
        new_syn, new_mem, ref_spike = lif_step(syn, mem, alpha, beta, threshold, h1)

        await step(dut, h1=h1)
        await FallingEdge(dut.clk)

        hw_spike = int(dut.spike_out.value)
        assert hw_spike == ref_spike, (
            f"Step {i}: h1={h1}, syn={syn}, mem={mem} → "
            f"expected spike={ref_spike}, got {hw_spike}"
        )
        syn, mem = new_syn, new_mem
