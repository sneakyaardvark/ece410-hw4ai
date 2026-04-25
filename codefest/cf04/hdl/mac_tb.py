import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def rising_edge(dut):
    """Wait for a rising clock edge and let non-blocking assignments settle."""
    await RisingEdge(dut.clk)
    await Timer(1, unit="step")


@cocotb.test()
async def test_mac(dut):
    cocotb.start_soon(Clock(dut.clk, 2, unit="step").start())

    # Synchronous reset
    dut.rst.value = 1
    dut.a.value = 0
    dut.b.value = 0
    await rising_edge(dut)
    assert dut.out.value.to_signed() == 0, (
        f"After reset: expected 0, got {dut.out.value.to_signed()}"
    )

    # a=3, b=4 for 3 cycles: accumulator should reach 12, 24, 36
    dut.rst.value = 0
    dut.a.value = 3
    dut.b.value = 4
    for cycle, expected in enumerate((12, 24, 36), start=1):
        await rising_edge(dut)
        got = dut.out.value.to_signed()
        assert got == expected, f"a=3,b=4 cycle {cycle}: expected {expected}, got {got}"

    # Assert reset: accumulator must clear to 0
    dut.rst.value = 1
    await rising_edge(dut)
    assert dut.out.value.to_signed() == 0, (
        f"After mid-run reset: expected 0, got {dut.out.value.to_signed()}"
    )

    # a=-5, b=2 for 2 cycles: accumulator should reach -10, -20
    dut.rst.value = 0
    dut.a.value = -5
    dut.b.value = 2
    for cycle, expected in enumerate((-10, -20), start=1):
        await rising_edge(dut)
        got = dut.out.value.to_signed()
        assert got == expected, f"a=-5,b=2 cycle {cycle}: expected {expected}, got {got}"
