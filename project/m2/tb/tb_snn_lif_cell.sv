`default_nettype none

// Module: tb_snn_lif_cell
// Description: Thin simulation wrapper for snn_lif_cell. Exposes all DUT ports
//              as top-level signals so cocotb can drive and sample them directly.
//
// Ports:
//   Name          Dir    Width   Purpose
//   -----------------------------------------------------------------------
//   clk           in     1       Clock driven by cocotb
//   rst           in     1       Synchronous reset driven by cocotb
//   alpha         in     16      Q1.15 synaptic decay constant
//   beta          in     16      Q1.15 membrane decay constant
//   threshold     in     32      INT32 spike threshold
//   h1_in         in     32      INT32 input current
//   h1_valid      in     1       Update strobe
//   spike_out     out    1       Registered spike output

module tb_snn_lif_cell (
    input  logic        clk,
    input  logic        rst,
    input  logic [15:0] alpha,
    input  logic [15:0] beta,
    input  logic signed [31:0] threshold,
    input  logic signed [31:0] h1_in,
    input  logic        h1_valid,
    output logic        spike_out
);

    snn_lif_cell #(
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
        .h1_valid  (h1_valid),
        .spike_out (spike_out)
    );

endmodule

`default_nettype wire
