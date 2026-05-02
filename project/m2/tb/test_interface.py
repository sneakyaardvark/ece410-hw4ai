"""cocotb testbench for spi_interface (via tb_interface wrapper).

DUT parametrized: NB_HIDDEN=8, NB_BYTES=1 spike byte.

Protocol: SPI Mode 0 (CPOL=0, CPHA=0), 4-byte transactions:
  [CMD:8][ADDR_HI:8][ADDR_LO:8][DATA:8]   CMD 0x02=WRITE, 0x03=READ

SCK driven from falling core-clock edges so SCK transitions never coincide
with rising core-clock edges, giving the 2-stage synchronizer full margin.

Register map (NB_HIDDEN=8, NB_BYTES=1):
  0x0000..0x0007   Weight SRAM bytes (write)
  0xA000           spike_in byte 0 (write)
  0xA019           Control: write 0x01 = start
  0xA01A           Status: bit1=done, bit0=busy (read)
  0xA01B           spike_out byte 0 (read, latched on done)
  0xA034/0xA035    alpha lo/hi (write)
  0xA036/0xA037    beta lo/hi (write)

Tests:
  - reset: outputs at defaults, strobes low
  - write_spike_in: complete SPI write to 0xA000 → spike_in register correct
  - write_weight: SPI write to SRAM → captures weight_wr_en strobe and data
  - start_pulse: SPI write 0x01 to control → captures one-cycle start pulse
  - read_status: SPI read 0xA01A with busy/done set → MISO byte correct
  - read_spike_out: SPI read 0xA01B after done latch → MISO byte correct
  - write_alpha: SPI write to alpha registers → alpha port updated
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

CMD_WRITE = 0x02
CMD_READ  = 0x03


async def init(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst.value       = 1
    dut.sck.value       = 0
    dut.cs_n.value      = 1
    dut.mosi.value      = 0
    dut.busy.value      = 0
    dut.done.value      = 0
    dut.spike_out.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def spi_transaction(dut, cmd: int, addr: int, data: int = 0x00) -> int:
    """Drive one 4-byte SPI transaction and return the MISO byte.

    SCK edges are set on falling core-clock edges so they are never coincident
    with rising core-clock edges (at which the DUT's synchronizers sample).
    Each SCK half-period = 4 core clocks = 80 ns.
    """
    bytes_out = [cmd, (addr >> 8) & 0xFF, addr & 0xFF, data]
    miso_byte = 0

    # CS_N goes low on a falling edge (mid-cycle, avoids coincidence)
    await FallingEdge(dut.clk)
    dut.cs_n.value = 0

    # CS_N-to-SCK setup: 4 rising edges
    for _ in range(4):
        await RisingEdge(dut.clk)

    for byte_idx, tx_byte in enumerate(bytes_out):
        for bit in range(7, -1, -1):          # MSB first
            # Set MOSI on falling edge (mid-cycle)
            await FallingEdge(dut.clk)
            dut.mosi.value = (tx_byte >> bit) & 1

            # MOSI-to-SCK setup: 3 rising edges
            for _ in range(3):
                await RisingEdge(dut.clk)

            # SCK rises on falling edge
            await FallingEdge(dut.clk)
            dut.sck.value = 1

            # SCK high: 4 rising edges
            for _ in range(4):
                await RisingEdge(dut.clk)

            # Sample MISO during data byte while SCK is still high
            if byte_idx == 3:
                miso_byte = (miso_byte << 1) | int(dut.miso.value)

            # SCK falls on falling edge
            await FallingEdge(dut.clk)
            dut.sck.value = 0

            # SCK-low hold: 2 rising edges before next MOSI/SCK
            for _ in range(2):
                await RisingEdge(dut.clk)

    # CS_N goes high on a falling edge
    await FallingEdge(dut.clk)
    dut.cs_n.value = 1
    dut.mosi.value = 0

    # Let the synchronizer see the CS_N edge (4 rising edges)
    for _ in range(4):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    return miso_byte


async def capture_strobe(dut, signal, timeout_cycles: int = 600) -> bool:
    """Return True if signal goes high within timeout_cycles rising edges."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        if int(signal.value) == 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_reset(dut):
    """After reset: strobes low, alpha/beta/threshold at defaults."""
    await init(dut)
    await FallingEdge(dut.clk)
    assert dut.weight_wr_en.value == 0,  "weight_wr_en should be 0 after reset"
    assert dut.start.value        == 0,  "start should be 0 after reset"
    assert dut.alpha.value.to_unsigned()     == 19875, "alpha default wrong"
    assert dut.beta.value.to_unsigned()      == 29635, "beta default wrong"
    assert dut.threshold.value.to_unsigned() == 32768, "threshold default wrong"


@cocotb.test()
async def test_write_spike_in(dut):
    """Complete SPI WRITE to 0xA000 updates spike_in[7:0].

    This is the primary write-transaction correctness test.
    """
    await init(dut)
    await spi_transaction(dut, CMD_WRITE, 0xA000, 0xC5)
    assert dut.spike_in.value.to_unsigned() == 0xC5, \
        f"spike_in mismatch: expected 0xC5, got {dut.spike_in.value}"


@cocotb.test()
async def test_write_weight(dut):
    """SPI WRITE to weight SRAM addr 0x0005 fires weight_wr_en with correct addr/data."""
    await init(dut)

    # Launch the SPI transaction in background; monitor weight_wr_en concurrently
    cocotb.start_soon(spi_transaction(dut, CMD_WRITE, 0x0005, 0x7F))

    fired = await capture_strobe(dut, dut.weight_wr_en, timeout_cycles=600)
    assert fired, "weight_wr_en never asserted"

    assert dut.weight_wr_addr.value.to_unsigned() == 0x0005, \
        f"weight_wr_addr wrong: {dut.weight_wr_addr.value}"
    assert dut.weight_wr_data.value.to_unsigned() == 0x7F, \
        f"weight_wr_data wrong: {dut.weight_wr_data.value}"

    # Strobe is one cycle; verify it cleared
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert dut.weight_wr_en.value == 0, "weight_wr_en should self-clear"


@cocotb.test()
async def test_start_pulse(dut):
    """Writing 0x01 to control register (0xA019) produces a one-cycle start pulse."""
    await init(dut)

    cocotb.start_soon(spi_transaction(dut, CMD_WRITE, 0xA019, 0x01))

    fired = await capture_strobe(dut, dut.start, timeout_cycles=600)
    assert fired, "start never asserted"

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert dut.start.value == 0, "start should self-clear after one cycle"


@cocotb.test()
async def test_read_status(dut):
    """SPI READ from 0xA01A returns {done, busy} correctly.

    This is the primary read-transaction correctness test.
    """
    await init(dut)

    # Set busy=1, done=0 directly (combinatorial inputs)
    dut.busy.value = 1
    dut.done.value = 0
    await RisingEdge(dut.clk)  # let the synchronizer in the DUT see it

    miso_byte = await spi_transaction(dut, CMD_READ, 0xA01A)
    assert miso_byte & 0x01 == 1, \
        f"busy=1 not reflected in status (0x{miso_byte:02X})"
    assert miso_byte & 0x02 == 0, \
        f"done=0 not reflected in status (0x{miso_byte:02X})"

    # Flip: busy=0, done=1
    dut.busy.value = 0
    dut.done.value = 1
    await RisingEdge(dut.clk)

    miso_byte = await spi_transaction(dut, CMD_READ, 0xA01A)
    assert miso_byte & 0x02 == 2, \
        f"done=1 not reflected in status (0x{miso_byte:02X})"
    assert miso_byte & 0x01 == 0, \
        f"busy=0 not reflected in status (0x{miso_byte:02X})"

    dut.done.value = 0


@cocotb.test()
async def test_read_spike_out(dut):
    """SPI READ from 0xA01B returns spike_out byte latched on done pulse."""
    await init(dut)

    # Latch spike_out by asserting done for one core clock
    dut.spike_out.value = 0xD7
    dut.done.value      = 1
    await RisingEdge(dut.clk)
    dut.done.value      = 0
    await FallingEdge(dut.clk)

    miso_byte = await spi_transaction(dut, CMD_READ, 0xA01B)
    assert miso_byte == 0xD7, \
        f"spike_out read wrong: expected 0xD7, got 0x{miso_byte:02X}"


@cocotb.test()
async def test_write_alpha(dut):
    """SPI WRITEs to alpha lo/hi registers update the alpha output port."""
    await init(dut)

    await spi_transaction(dut, CMD_WRITE, 0xA034, 0x34)   # alpha[7:0]
    await spi_transaction(dut, CMD_WRITE, 0xA035, 0x12)   # alpha[15:8]

    assert dut.alpha.value.to_unsigned() == 0x1234, \
        f"alpha mismatch: expected 0x1234, got {dut.alpha.value.to_unsigned():#06x}"
