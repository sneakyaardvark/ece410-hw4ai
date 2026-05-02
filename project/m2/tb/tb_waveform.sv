`default_nettype none
// Standalone waveform generator for spi_interface.
// Drives one WRITE (0xA019 start) and one READ (0xA01A status) transaction.
// SCK half-period = 4 core clocks = 80ns; core clock period = 20ns.
module tb_waveform;
    parameter int CLK_HALF = 10;  // ns
    parameter int SCK_HOLD = 4;   // core clocks per SCK half

    logic clk=0, rst=1, sck=0, cs_n=1, mosi=0;
    logic miso;
    logic weight_wr_en, start;
    logic [15:0] weight_wr_addr;
    logic [7:0]  weight_wr_data, spike_in;
    logic [15:0] alpha, beta;
    logic signed [31:0] threshold;
    logic busy=0, done=0;
    logic [7:0] spike_out=8'hA5;

    spi_interface #(.NB_HIDDEN(8)) dut (
        .clk(clk), .rst(rst), .sck(sck), .cs_n(cs_n), .mosi(mosi), .miso(miso),
        .weight_wr_en(weight_wr_en), .weight_wr_addr(weight_wr_addr),
        .weight_wr_data(weight_wr_data), .alpha(alpha), .beta(beta),
        .threshold(threshold), .start(start), .spike_in(spike_in),
        .busy(busy), .done(done), .spike_out(spike_out)
    );

    always #(CLK_HALF) clk = ~clk;

    task automatic drive_byte(input [7:0] b);
        integer bit_i;
        for (bit_i = 7; bit_i >= 0; bit_i = bit_i - 1) begin
            @(negedge clk); mosi = b[bit_i];
            repeat(3) @(posedge clk);
            @(negedge clk); sck = 1;
            repeat(4) @(posedge clk);
            @(negedge clk); sck = 0;
            repeat(2) @(posedge clk);
        end
    endtask

    task automatic spi_write(input [7:0] cmd, input [15:0] addr, input [7:0] data);
        @(negedge clk); cs_n = 0;
        repeat(4) @(posedge clk);
        drive_byte(cmd);
        drive_byte(addr[15:8]);
        drive_byte(addr[7:0]);
        drive_byte(data);
        @(negedge clk); cs_n = 1; mosi = 0;
        repeat(4) @(posedge clk);
        @(negedge clk);
    endtask

    task automatic spi_read(input [7:0] cmd, input [15:0] addr);
        @(negedge clk); cs_n = 0;
        repeat(4) @(posedge clk);
        drive_byte(cmd);
        drive_byte(addr[15:8]);
        drive_byte(addr[7:0]);
        drive_byte(8'h00);
        @(negedge clk); cs_n = 1; mosi = 0;
        repeat(4) @(posedge clk);
        @(negedge clk);
    endtask

    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, tb_waveform);

        repeat(2) @(posedge clk);
        rst = 0;
        @(posedge clk);

        // Write 0xC5 to spike_in (0xA000)
        spi_write(8'h02, 16'hA000, 8'hC5);

        // Latch spike_out via done pulse
        done = 1; @(posedge clk); done = 0;

        // Set busy for status read
        busy = 1;
        // Read status register (0xA01A)
        spi_read(8'h03, 16'hA01A);
        busy = 0;

        // Write start pulse (0xA019)
        spi_write(8'h02, 16'hA019, 8'h01);

        repeat(4) @(posedge clk);
        $finish;
    end
endmodule
`default_nettype wire
