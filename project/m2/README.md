# Milestone 2 — SNN Hardware Accelerator

## RTL Modules

| File | Module | Description |
|------|--------|-------------|
| `rtl/snn_mac.sv` | `snn_mac` | Signed INT8 multiply-accumulate unit |
| `rtl/snn_mac_array.sv` | `snn_mac_array` | Parallel array of NB_MACS MACs |
| `rtl/snn_lif_cell.sv` | `snn_lif_cell` | Single LIF neuron (Q1.15 decay, INT32 state) |
| `rtl/snn_lif_bank.sv` | `snn_lif_bank` | Bank of NB_HIDDEN LIF cells |
| `rtl/compute_core.sv` | `compute_core` | Tiled inference engine FSM |
| `rtl/interface.sv` | `spi_interface` | SPI slave (Mode 0) register interface |

## Simulator

**Icarus Verilog 12.0 (stable)** via cocotb 2.0.1.

```
iverilog -V
# Icarus Verilog version 12.0 (stable)
```

## Running Tests

Tests require the project virtual environment (created with `uv`):

```bash
cd project/m2/tb

# Individual targets
make mac
make mac_array
make lif_cell
make lif_bank
make compute_core
make interface

# All targets
make all
```

Each target compiles the RTL with iverilog and runs the cocotb Python testbench.
Results print a PASS/FAIL summary table; logs for `compute_core` and `interface`
are captured in `sim/`.

## Simulation Logs

- `sim/compute_core_run.log` — 5 tests, all PASS
- `sim/interface_run.log` — 7 tests, all PASS
- `sim/waveform.png` — annotated waveform showing one SPI WRITE and one SPI READ transaction

## Deviations from Milestone 1 Plan

1. **Tiled MAC execution.** The M1 plan described a single-pass dot product.
   `compute_core` instead uses a tiled scheme: the NB_MACS-wide array makes
   NB_TILES passes over the input vector, accumulating partial sums into an
   INT32 accumulator SRAM before feeding the LIF bank. This was necessary to
   support NB_HIDDEN=200 with a small MAC array (NB_MACS=8) without exposing
   a 200-wide multiply in the critical path.

2. **Module naming.** The SPI interface file is named `interface.sv` but the
   module is `spi_interface` because `interface` is a reserved keyword in
   SystemVerilog.

3. **Register-file extension.** The M1 interface sketch did not include
   alpha/beta/threshold configuration registers. These were added at
   `0xA034`–`0xA03B` to allow runtime tuning of LIF parameters without
   recompilation.
