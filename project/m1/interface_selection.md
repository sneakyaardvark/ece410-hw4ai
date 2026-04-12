# Interface Selection

Selection: SPI

## Bandwidth requirement
The input data, if densely packed, would be 8.75 KB per audio sample (each 1.4 seconds long). The output spikes are 2.5 KB, so a single sample will be a transfer of 11.25 KB total. The compute time, given the hardware target of 0.8 GOPS, would be 45 ms. Required bandwidth $= 11.25 KB / 45 ms = 250 KB/s$. Even with a 1 MHz SPI connection this would be satisfied, and the simplest of MCUs can handle this datarate.
