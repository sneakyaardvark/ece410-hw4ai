"""cocotb testbench for compute_core (via tb_compute_core wrapper).

DUT is parametrized: NB_HIDDEN=8, NB_MACS=8, NB_STEPS=5.
The weight matrix is 8×8 INT8 (64 bytes).  Byte address layout:
  byte_addr = in_neuron * NB_HIDDEN + out_neuron
  h1[out] = Σ_in  W[in][out] * spike_vec[in]   (i.e. W^T @ spike_vec)

Tests:
  - reset: busy/done low, spike_out zero
  - busy stays high throughout inference, drops on done
  - zero weights: no spikes
  - full inference: random INT8 weights, compared against Python reference model
  - diagonal weights: each output driven only by corresponding input neuron
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
import numpy as np

NB       = 8
NB_STEPS = 5
ALPHA_Q  = 19875
BETA_Q   = 29635
THRESH   = 32768


def q15_mul(coeff: int, value: int) -> int:
    product = coeff * value
    return product >> 15 if product >= 0 else -((-product) >> 15)


def lif_step(syn: int, mem: int, alpha: int, beta: int,
             threshold: int, h1: int):
    new_syn = q15_mul(alpha, syn) + h1
    spike   = 1 if mem >= threshold else 0
    new_mem = 0 if spike else (q15_mul(beta, mem) + syn)
    return new_syn, new_mem, spike


def reference_inference(weights_flat: list, spike_in: list,
                        alpha: int, beta: int, threshold: int,
                        nb_steps: int = NB_STEPS) -> list:
    """Fixed-point reference matching compute_core.sv.

    weights_flat[in*NB + out] = signed INT8 weight from input to output neuron.
    Returns spike_out list[NB] after nb_steps recurrent steps.
    """
    W = np.array(weights_flat, dtype=np.int8).reshape(NB, NB)  # W[in][out]
    spike_vec = list(spike_in)
    syn = [0] * NB
    mem = [0] * NB

    for _ in range(nb_steps):
        sv = np.array(spike_vec, dtype=np.int32)
        h1 = [int(np.dot(W[:, j].astype(np.int32), sv)) for j in range(NB)]

        new_syn, new_mem, new_spk = [0]*NB, [0]*NB, [0]*NB
        for k in range(NB):
            new_syn[k], new_mem[k], new_spk[k] = lif_step(
                syn[k], mem[k], alpha, beta, threshold, h1[k])
        syn, mem, spike_vec = new_syn, new_mem, new_spk

    return spike_vec


async def init(dut, alpha=ALPHA_Q, beta=BETA_Q, threshold=THRESH):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst.value            = 1
    dut.weight_wr_en.value   = 0
    dut.weight_wr_addr.value = 0
    dut.weight_wr_data.value = 0
    dut.alpha.value          = alpha
    dut.beta.value           = beta
    dut.threshold.value      = threshold
    dut.start.value          = 0
    dut.spike_in.value       = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def load_weights(dut, weights_flat: list):
    """Write NB*NB bytes to v1 SRAM, one byte per cycle."""
    dut.weight_wr_en.value = 1
    for addr, byte_val in enumerate(weights_flat):
        dut.weight_wr_addr.value = addr
        dut.weight_wr_data.value = int(byte_val) & 0xFF
        await RisingEdge(dut.clk)
    dut.weight_wr_en.value = 0


async def run_inference(dut, spike_in_bits: list) -> list:
    """Pulse start, wait for done, return spike_out as list[NB]."""
    spike_in_int = sum(b << i for i, b in enumerate(spike_in_bits))
    dut.spike_in.value = spike_in_int
    dut.start.value    = 1
    await RisingEdge(dut.clk)
    dut.start.value    = 0

    for _ in range(2000):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break
    else:
        raise AssertionError("done never asserted within 2000-cycle timeout")

    raw = dut.spike_out.value.to_unsigned()
    return [(raw >> i) & 1 for i in range(NB)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_reset(dut):
    """After reset, busy=0, done=0, spike_out=0."""
    await init(dut)
    await FallingEdge(dut.clk)
    assert dut.busy.value     == 0, "busy should be 0 after reset"
    assert dut.done.value     == 0, "done should be 0 after reset"
    assert dut.spike_out.value.to_unsigned() == 0, "spike_out should be 0 after reset"


@cocotb.test()
async def test_busy_lifecycle(dut):
    """busy asserted from start, held through inference, cleared on done."""
    await init(dut)
    await load_weights(dut, [0] * (NB * NB))

    dut.spike_in.value = 0xFF
    dut.start.value    = 1
    await RisingEdge(dut.clk)
    dut.start.value    = 0

    await FallingEdge(dut.clk)
    assert dut.busy.value == 1, "busy should be 1 immediately after start"

    # Sample at FallingEdge so always_ff delta cycles have settled.
    # done=1 and busy=0 fire on the same rising edge; break on done before
    # asserting busy to avoid the simultaneous-drop false failure.
    for _ in range(2000):
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        if dut.done.value == 1:
            break
        assert dut.busy.value == 1, "busy dropped before done"
    else:
        raise AssertionError("done never asserted")

    # done and busy both registered at the same edge; busy already 0 here
    assert dut.busy.value == 0, "busy should be 0 on done cycle"


@cocotb.test()
async def test_zero_weights_no_spikes(dut):
    """All-zero weights and spike_in=0 → spike_out=0 after NB_STEPS steps."""
    await init(dut)
    await load_weights(dut, [0] * (NB * NB))
    result = await run_inference(dut, [0] * NB)
    assert all(s == 0 for s in result), f"Expected all zeros, got {result}"


@cocotb.test()
async def test_full_inference_random(dut):
    """Random INT8 weights and spike_in; compare spike_out to reference model.

    Uses a low threshold (100) so neurons spike, exercising the spike path.
    """
    rng = np.random.default_rng(42)
    alpha, beta, threshold = ALPHA_Q, BETA_Q, 100
    await init(dut, alpha=alpha, beta=beta, threshold=threshold)

    weights_signed = rng.integers(-64, 64, NB * NB, dtype=np.int8).tolist()
    spike_in_bits  = rng.integers(0, 2, NB, dtype=np.int32).tolist()

    await load_weights(dut, [b & 0xFF for b in weights_signed])
    hw_spikes  = await run_inference(dut, spike_in_bits)
    ref_spikes = reference_inference(
        weights_signed, spike_in_bits, alpha, beta, threshold)

    mismatches = [k for k in range(NB) if hw_spikes[k] != ref_spikes[k]]
    assert not mismatches, (
        f"Mismatch at neurons {mismatches}. "
        f"HW={[hw_spikes[k] for k in mismatches]}, "
        f"REF={[ref_spikes[k] for k in mismatches]}"
    )


@cocotb.test()
async def test_full_inference_diagonal(dut):
    """Diagonal weight matrix: output k driven only by input k.

    Also verifies correctness against reference model with standard threshold.
    """
    alpha, beta, threshold = ALPHA_Q, BETA_Q, 100
    await init(dut, alpha=alpha, beta=beta, threshold=threshold)

    weights_flat  = [50 if i == j else 0 for i in range(NB) for j in range(NB)]
    spike_in_bits = [1, 0, 1, 0, 1, 0, 1, 0]

    await load_weights(dut, [b & 0xFF for b in weights_flat])
    hw_spikes  = await run_inference(dut, spike_in_bits)
    ref_spikes = reference_inference(
        weights_flat, spike_in_bits, alpha, beta, threshold)

    assert hw_spikes == ref_spikes, (
        f"Diagonal test mismatch: HW={hw_spikes}, REF={ref_spikes}"
    )
