// Composition semantics exemplar, case A: plain-wire glue.
// hot_src's output is STRUCTURALLY one-hot (combinational shift of sel),
// so for every state induction can invent, b latches a one-hot value.
// b's property is therefore inductive with no invariants at all: the
// composition destroyed necessity. Expected verdict: DECORATIVE.
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

module compose_wire (
    input wire clk,
    input wire rst,
    input wire step,
    input wire in_valid,
    output wire [3:0] held_o
);
    wire [3:0] hot;
    hot_src  a (.clk(clk), .rst(rst), .step(step), .hot(hot));
    hot_latch b (.clk(clk), .rst(rst), .in_valid(in_valid),
                 .in_data(hot), .held(held_o));
endmodule
