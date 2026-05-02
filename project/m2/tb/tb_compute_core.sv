`default_nettype none

// Module: tb_compute_core
// Description: Simulation wrapper for compute_core. Exposes all DUT ports as
//              top-level signals for cocotb. Parametrized with NB_HIDDEN=8,
//              NB_MACS=8, NB_STEPS=5 for fast simulation; the 8-wide MAC
//              array covers the full weight matrix in a single tile (NB_TILES=1).
//
// Ports:
//   Name             Dir    Width   Purpose
//   -----------------------------------------------------------------------
//   clk              in     1       Clock driven by cocotb
//   rst              in     1       Synchronous reset driven by cocotb
//   weight_wr_en     in     1       Write enable for v1 weight SRAM
//   weight_wr_addr   in     16      Byte address in v1 SRAM (row*8 + col)
//   weight_wr_data   in     8       INT8 weight byte to write
//   alpha            in     16      Q1.15 synaptic decay constant
//   beta             in     16      Q1.15 membrane decay constant
//   threshold        in     32      INT32 spike threshold
//   start            in     1       Pulse to begin inference; latches spike_in
//   spike_in         in     8       Initial hidden spike vector (NB_HIDDEN=8)
//   busy             out    1       High while inference is in progress
//   done             out    1       One-cycle pulse; spike_out valid on this cycle
//   spike_out        out    8       Final hidden spike vector (NB_HIDDEN=8)

module tb_compute_core (
    input  logic        clk,
    input  logic        rst,

    input  logic        weight_wr_en,
    input  logic [15:0] weight_wr_addr,
    input  logic [7:0]  weight_wr_data,

    input  logic [15:0] alpha,
    input  logic [15:0] beta,
    input  logic signed [31:0] threshold,

    input  logic        start,
    input  logic [7:0]  spike_in,

    output logic        busy,
    output logic        done,
    output logic [7:0]  spike_out
);

    compute_core #(
        .NB_HIDDEN (8),
        .NB_MACS   (8),
        .NB_STEPS  (5),
        .STATE_W   (32),
        .DECAY_W   (16),
        .WEIGHT_W  (8),
        .ACC_W     (32)
    ) dut (
        .clk            (clk),
        .rst            (rst),
        .weight_wr_en   (weight_wr_en),
        .weight_wr_addr (weight_wr_addr),
        .weight_wr_data (weight_wr_data),
        .alpha          (alpha),
        .beta           (beta),
        .threshold      (threshold),
        .start          (start),
        .spike_in       (spike_in),
        .busy           (busy),
        .done           (done),
        .spike_out      (spike_out)
    );

endmodule

`default_nettype wire
