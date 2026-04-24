module mac (
    input  logic              clk,  // 1-bit clock
    input  logic              rst,  // 1-bit active-high synchronous reset
    input  logic signed [7:0] a,    // 8-bit signed input
    input  logic signed [7:0] b,    // 8-bit signed input
    output logic signed [31:0] out  // 32-bit signed accumulator
);

    // Sequential logic for the MAC operation
    always_ff @(posedge clk) begin
        if (rst) begin
            // Synchronous reset: clear the accumulator
            out <= 32'sd0;
        end else begin
            // Accumulate: out = out + (a * b)
            // The compiler handles sign extension automatically 
            // due to the 'signed' keywords.
            out <= out + (a * b);
        end
    end

endmodule
