`default_nettype none

// Module: tb_interface
// Description: Simulation wrapper for spi_interface. Exposes all DUT ports as
//              top-level signals so cocotb can drive and sample them directly.
//              NB_HIDDEN fixed at 8 for simulation speed (byte-aligned).
//
// Ports:
//   Name             Dir    Width   Purpose
//   -----------------------------------------------------------------------
//   clk              in     1       Clock driven by cocotb
//   rst              in     1       Synchronous reset driven by cocotb
//   sck              in     1       SPI clock from cocotb host model
//   cs_n             in     1       SPI chip select (active-low)
//   mosi             in     1       Host-to-slave data
//   miso             out    1       Slave-to-host data
//   weight_wr_en     out    1       Write enable to weight SRAM
//   weight_wr_addr   out    16      Byte address in weight SRAM
//   weight_wr_data   out    8       INT8 weight byte to write
//   alpha            out    16      Q1.15 synaptic decay
//   beta             out    16      Q1.15 membrane decay
//   threshold        out    32      INT32 spike threshold
//   start            out    1       One-cycle start pulse
//   spike_in         out    8       Initial spike vector (NB_HIDDEN=8)
//   busy             in     1       Inference busy flag
//   done             in     1       Inference done flag
//   spike_out        in     8       Final spike vector (NB_HIDDEN=8)

module tb_interface (
    input  logic        clk,
    input  logic        rst,

    input  logic        sck,
    input  logic        cs_n,
    input  logic        mosi,
    output logic        miso,

    output logic        weight_wr_en,
    output logic [15:0] weight_wr_addr,
    output logic [7:0]  weight_wr_data,
    output logic [15:0] alpha,
    output logic [15:0] beta,
    output logic signed [31:0] threshold,
    output logic        start,
    output logic [7:0]  spike_in,

    input  logic        busy,
    input  logic        done,
    input  logic [7:0]  spike_out
);

    spi_interface #(
        .NB_HIDDEN   (8),
        .STATE_W     (32),
        .DECAY_W     (16),
        .ALPHA_INIT  (19875),
        .BETA_INIT   (29635),
        .THRESH_INIT (32768)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .sck           (sck),
        .cs_n          (cs_n),
        .mosi          (mosi),
        .miso          (miso),
        .weight_wr_en  (weight_wr_en),
        .weight_wr_addr(weight_wr_addr),
        .weight_wr_data(weight_wr_data),
        .alpha         (alpha),
        .beta          (beta),
        .threshold     (threshold),
        .start         (start),
        .spike_in      (spike_in),
        .busy          (busy),
        .done          (done),
        .spike_out     (spike_out)
    );

endmodule

`default_nettype wire
