// Composition semantics exemplar, case B: STATEFUL glue.
// Same two verified parts, but the data now passes through a buffer
// register that HOLDS its value while step is low — so induction can
// start with garbage in buf_d, idle on it for the whole window, then
// deliver it to b. The parents' invariants say nothing about the glue;
// the invariant list must grow to cover it. Expected: NECESSARY with
// the buf_d one-hot clause, NOT_INDUCTIVE without.
module hot_src (
    input wire clk,
    input wire rst,
    input wire step,
    output wire [3:0] hot
);
    reg [1:0] sel;
    initial sel = 2'd0;
    always @(posedge clk)
        if (rst) sel <= 2'd0;
        else if (step) sel <= sel + 2'd1;
    assign hot = 4'b0001 << sel;
endmodule

module hot_latch (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [3:0] in_data,
    output reg [3:0] held
);
    initial held = 4'b0001;
    always @(posedge clk)
        if (rst) held <= 4'b0001;
        else if (in_valid) held <= in_data;

    always @(posedge clk)
        assert (held == 4'b0001 || held == 4'b0010 ||
                held == 4'b0100 || held == 4'b1000);
endmodule

module compose_buffer (
    input wire clk,
    input wire rst,
    input wire step,
    input wire in_valid,
    output wire [3:0] held_o,
    output reg  [3:0] buf_d,   // glue state, port-exposed for invariants
    output reg        buf_v
);
    wire [3:0] hot;
    hot_src a (.clk(clk), .rst(rst), .step(step), .hot(hot));

    initial buf_d = 4'b0001;
    initial buf_v = 1'b0;
    always @(posedge clk)
        if (rst) begin
            buf_d <= 4'b0001;
            buf_v <= 1'b0;
        end else begin
            if (step) buf_d <= hot;   // data refills only on step
            buf_v <= in_valid;        // valid follows the consumer side
        end

    hot_latch b (.clk(clk), .rst(rst), .in_valid(buf_v),
                 .in_data(buf_d), .held(held_o));
endmodule
