`default_nettype none

// Module: tb_snn_lif_bank
// Description: Thin simulation wrapper for snn_lif_bank. Exposes all DUT ports
//              as top-level signals so cocotb can drive and sample them directly.
//              NB fixed at 8 for simulation speed; full-size (200) is tested in
//              test_snn_lif_bank.py via the representative test.
//
// Ports:
//   Name          Dir    Width   Purpose
//   -----------------------------------------------------------------------
//   clk           in     1       Clock driven by cocotb
//   rst           in     1       Synchronous reset driven by cocotb
//   alpha         in     16      Q1.15 synaptic decay constant
//   beta          in     16      Q1.15 membrane decay constant
//   threshold     in     32      INT32 spike threshold
//   h1_in         in     32      INT32 input current for selected cell
//   h1_idx        in     8       Index of cell to update (supports up to 256)
//   h1_valid      in     1       Update strobe
//   spike_vec     out    200     One spike bit per cell

module tb_snn_lif_bank (
    input  logic        clk,
    input  logic        rst,
    input  logic [15:0] alpha,
    input  logic [15:0] beta,
    input  logic signed [31:0] threshold,
    input  logic signed [31:0] h1_in,
    input  logic [7:0]  h1_idx,
    input  logic        h1_valid,
    output logic [199:0] spike_vec
);

    snn_lif_bank #(
        .NB      (200),
        .STATE_W (32),
        .DECAY_W (16),
        .ACC_W   (32)
    ) dut (
        .clk       (clk),
        .rst       (rst),
        .alpha     (alpha),
        .beta      (beta),
        .threshold (threshold),
        .h1_in     (h1_in),
        .h1_idx    (h1_idx),
        .h1_valid  (h1_valid),
        .spike_vec (spike_vec)
    );

endmodule

`default_nettype wire
