"""cocotb testbench for snn_lif_bank (via tb_snn_lif_bank wrapper).

The reference model is the same fixed-point LIF step used in test_snn_lif_cell,
applied independently to each of the NB neurons.

Tests:
  - reset clears all spike bits in spike_vec
  - only the addressed cell updates; all others hold state
  - all cells update correctly when streamed sequentially
  - spike_vec reflects each cell's independent spike after a full step
  - representative full time step: 200 neurons, random h1 inputs, compared
    cycle-by-cycle against the reference model
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

NB      = 200
ALPHA_Q = 19875
BETA_Q  = 29635
THRESH  = 32768


def q15_mul(coeff: int, value: int) -> int:
    product = coeff * value
    return product >> 15 if product >= 0 else -((-product) >> 15)


def lif_step(syn, mem, alpha, beta, threshold, h1):
    new_syn = q15_mul(alpha, syn) + h1
    spike   = 1 if mem >= threshold else 0
    new_mem = 0 if spike else (q15_mul(beta, mem) + syn)
    return new_syn, new_mem, spike


async def init(dut, alpha=ALPHA_Q, beta=BETA_Q, threshold=THRESH):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst.value       = 1
    dut.alpha.value     = alpha
    dut.beta.value      = beta
    dut.threshold.value = threshold
    dut.h1_in.value     = 0
    dut.h1_idx.value    = 0
    dut.h1_valid.value  = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def update_cell(dut, idx: int, h1: int):
    """Send one h1_valid pulse to cell idx."""
    dut.h1_idx.value   = idx
    dut.h1_in.value    = h1
    dut.h1_valid.value = 1
    await RisingEdge(dut.clk)
    dut.h1_valid.value = 0


async def stream_step(dut, h1_vec: list[int]):
    """Update all NB cells serially with h1 values from h1_vec."""
    assert len(h1_vec) == NB
    dut.h1_valid.value = 1
    for idx, h1 in enumerate(h1_vec):
        dut.h1_idx.value = idx
        dut.h1_in.value  = h1
        await RisingEdge(dut.clk)
    dut.h1_valid.value = 0


def read_spike_vec(dut) -> list[int]:
    raw = dut.spike_vec.value.to_unsigned()
    return [(raw >> i) & 1 for i in range(NB)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_reset(dut):
    """After reset, all spike bits must be 0."""
    await init(dut)
    await FallingEdge(dut.clk)
    spikes = read_spike_vec(dut)
    assert all(s == 0 for s in spikes), f"Expected all zeros after reset, got {spikes}"


@cocotb.test()
async def test_only_addressed_cell_updates(dut):
    """Update cell 5 with a large h1; no other cell should spike."""
    await init(dut, threshold=100)
    # Prime cell 5: step 1 loads syn, step 2 loads mem, step 3 spikes
    for _ in range(3):
        await update_cell(dut, idx=5, h1=500)
        await RisingEdge(dut.clk)  # settle

    await FallingEdge(dut.clk)
    spikes = read_spike_vec(dut)
    for i, s in enumerate(spikes):
        if i == 5:
            assert s == 1, f"Cell 5 should have spiked, got spike_vec[5]={s}"
        else:
            assert s == 0, f"Cell {i} should not have spiked, got spike_vec[{i}]={s}"


@cocotb.test()
async def test_representative_full_step(dut):
    """One complete time step: stream h1 to all 200 cells, compare spike_vec
    against the reference model after 3 steps (pipeline fill included)."""
    import numpy as np
    rng = np.random.default_rng(7)

    alpha, beta, threshold = ALPHA_Q, BETA_Q, THRESH
    await init(dut, alpha=alpha, beta=beta, threshold=threshold)

    # Reference state for all NB cells
    syn_ref = [0] * NB
    mem_ref = [0] * NB

    # Run 3 time steps so the pipeline fills and spikes can occur
    h1_steps = [
        rng.integers(-2000, 4000, NB, dtype=np.int32).tolist(),
        rng.integers(-500,  500,  NB, dtype=np.int32).tolist(),
        rng.integers(-500,  500,  NB, dtype=np.int32).tolist(),
    ]

    for step_idx, h1_vec in enumerate(h1_steps):
        await stream_step(dut, h1_vec)
        await FallingEdge(dut.clk)  # all NB cells have been updated

        # Advance reference model
        new_syn = [0] * NB
        new_mem = [0] * NB
        ref_spikes = [0] * NB
        for k in range(NB):
            new_syn[k], new_mem[k], ref_spikes[k] = lif_step(
                syn_ref[k], mem_ref[k], alpha, beta, threshold, h1_vec[k]
            )
        syn_ref, mem_ref = new_syn, new_mem

        hw_spikes = read_spike_vec(dut)
        mismatches = [k for k in range(NB) if hw_spikes[k] != ref_spikes[k]]
        assert not mismatches, (
            f"Step {step_idx}: spike mismatch at cells {mismatches}. "
            f"HW={[hw_spikes[k] for k in mismatches]}, "
            f"REF={[ref_spikes[k] for k in mismatches]}"
        )
