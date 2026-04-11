# Target ASIC Specs (Edge SNN Accelerator)

## Design constraints

- Must synthesize on OpenLane 2 (sky130 PDK)
- SPI external interface (low-bandwidth, serial)
- Optimized for efficiency over accuracy — small, low-power edge/embedded target
- INT8 arithmetic chosen for area efficiency; SNNs are robust to quantization

## Weight storage

The SNN model has 184K parameters:
- Input projection: 700 × 200 = 140,000
- Recurrent hidden: 200 × 200 = 40,000
- Output projection: 200 × 20 = 4,000

At INT8 this is ~184 KB, small enough to store entirely on-chip in SRAM.
Since weights are on-chip, the relevant roofline bandwidth is SRAM, not SPI.

## Specs

| Parameter | Value |
|-----------|-------|
| Data type | INT8 |
| MACs | 8 |
| Clock | 50 MHz |
| Peak compute | 8 × 2 × 50M = 0.8 GOPS |
| SRAM port width | 64 bits |
| On-chip BW | 50M × 8B = 0.4 GB/s |
| Ridge point | 0.8 / 0.4 = 2.0 OPS/byte |
| External interface | SPI |
| Weight storage | On-chip SRAM (~184 KB) |

## Roofline placement

The dominant kernel (bmm) has an arithmetic intensity of ~55 OPS/byte,
well above the ridge point of 2.0 OPS/byte. The design is solidly
compute-bound, as expected for a small edge accelerator with limited
MAC units but fast on-chip memory access.
